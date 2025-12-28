"""유틸리티 함수들"""
import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from typing import Optional, Type, Union
import os
from datetime import datetime

from config import (
    TrainingConfig,
    get_default_params,
    is_algo_env_compatible,
    load_config_from_json,
    merge_configs
)


def create_env(config: TrainingConfig):
    """환경 생성"""
    # render_mode 설정
    render_mode = config.render_mode if config.render else None

    # Atari 환경인 경우
    if "ALE/" in config.env_name:
        env = gym.make(config.env_name, render_mode=render_mode)
        env = AtariWrapper(env)
        if config.n_envs > 1:
            raise ValueError("Atari 환경은 현재 단일 환경만 지원합니다.")
        return env

    # 병렬 환경 생성 (렌더링은 병렬 환경에서 지원하지 않음)
    if config.n_envs > 1:
        if config.render:
            print("경고: 병렬 환경에서는 렌더링을 지원하지 않습니다. 렌더링을 비활성화합니다.")
        # A2C, PPO는 병렬 환경 지원
        if config.algorithm in ["a2c", "ppo"]:
            env = make_vec_env(
                config.env_name,
                n_envs=config.n_envs,
                seed=config.seed,
            )
            return env
        else:
            print(f"경고: {config.algorithm}은 병렬 환경을 지원하지 않습니다. 단일 환경으로 생성합니다.")

    # 단일 환경
    env = gym.make(config.env_name, render_mode=render_mode)
    if config.seed is not None:
        env.reset(seed=config.seed)

    return env


def get_algorithm_class(algorithm: str) -> Type[Union[DQN, A2C, PPO, SAC]]:
    """알고리즘 이름으로 클래스 반환"""
    algorithms = {
        "dqn": DQN,
        "a2c": A2C,
        "ppo": PPO,
        "sac": SAC,
    }

    if algorithm not in algorithms:
        raise ValueError(f"지원하지 않는 알고리즘: {algorithm}")

    return algorithms[algorithm]


def create_model(config: TrainingConfig, env, use_json_config: bool = True):
    """모델 생성

    Args:
        config: TrainingConfig 객체
        env: Gymnasium 환경
        use_json_config: JSON 설정 파일 사용 여부 (기본값: True)
    """
    # 호환성 확인
    if not is_algo_env_compatible(config.algorithm, config.env_name):
        raise ValueError(
            f"{config.algorithm}은 {config.env_name} 환경을 지원하지 않습니다.\n"
            f"지원 알고리즘: {', '.join(config.supported_algos)}"
        )

    # 알고리즘 클래스 가져오기
    AlgoClass = get_algorithm_class(config.algorithm)

    # 파라미터 우선순위:
    # 1. JSON 설정 파일 (use_json_config=True인 경우)
    # 2. 코드 기본값 (ALGORITHM_DEFAULTS)
    # 3. 사용자 지정 (config.algo_params)

    if use_json_config:
        # JSON에서 로드
        json_params = load_config_from_json(config.algorithm, config.env_name)
        # 코드 기본값과 병합
        default_params = get_default_params(config.algorithm, config.env_name)
        base_params = merge_configs(default_params, json_params)
    else:
        # 코드 기본값만 사용
        base_params = get_default_params(config.algorithm, config.env_name)

    # 사용자 지정 파라미터와 병합 (최우선)
    algo_params = merge_configs(base_params, config.algo_params)

    # Policy 타입 설정 (JSON에서 policy 지정 가능)
    if "policy" in algo_params:
        policy_type = algo_params.pop("policy")
    elif "ALE/" in config.env_name:
        policy_type = "CnnPolicy"
    else:
        policy_type = config.policy_type

    # 네트워크 구조 설정
    if config.net_arch is not None:
        policy_kwargs = {"net_arch": config.net_arch}
    else:
        policy_kwargs = {}

    # total_timesteps는 config에서 관리 (algo_params에서 제거)
    algo_params.pop("total_timesteps", None)

    # 명시적으로 설정할 파라미터 (중복 방지를 위해 algo_params에서 제거)
    if "learning_rate" not in algo_params:
        algo_params["learning_rate"] = config.learning_rate
    if "verbose" not in algo_params:
        algo_params["verbose"] = 1
    if "tensorboard_log" not in algo_params:
        algo_params["tensorboard_log"] = config.tb_log_dir
    if config.seed is not None:
        algo_params["seed"] = config.seed
    if policy_kwargs:
        algo_params["policy_kwargs"] = policy_kwargs

    # 모델 생성
    model = AlgoClass(
        policy=policy_type,
        env=env,
        **algo_params,
    )

    return model


def create_callbacks(config: TrainingConfig, eval_env=None):
    """콜백 생성"""
    callbacks = []

    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, config.total_timesteps // 10),
        save_path=config.results_dir,
        name_prefix=f"{config.algorithm}_checkpoint",
        save_replay_buffer=config.algorithm in ["dqn", "sac"],
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # 평가 콜백 (선택사항)
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=config.results_dir,
            log_path=config.results_dir,
            eval_freq=max(5000, config.total_timesteps // 20),
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    return callbacks


def train_model(config: TrainingConfig, use_json_config: bool = True):
    """모델 학습

    Args:
        config: TrainingConfig 객체
        use_json_config: JSON 설정 파일 사용 여부 (기본값: True)
    """
    # TensorBoard 로그 이름 생성 (날짜/시간만, 알고리즘 이름은 폴더 구조에 이미 포함)
    tb_log_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print(f"🚀 {config.algorithm.upper()} 학습 시작: {config.env_name}")
    print("=" * 70)
    print(f"📁 결과 저장 위치: {config.results_dir}")
    print(f"⏱️  총 Timesteps: {config.total_timesteps:,}")
    print(f"🔧 병렬 환경 수: {config.n_envs}")
    if config.tensorboard_log:
        print(f"📊 TensorBoard 로그: {config.tb_log_dir}/{tb_log_name}")
    if use_json_config:
        print(f"⚙️  JSON 설정 사용: configs/{config.algorithm}.json")
    print("=" * 70)

    # 환경 생성
    env = create_env(config)

    # 평가 환경 생성 (병렬 환경이 아닌 경우)
    eval_env = None
    if config.n_envs == 1 and "ALE/" not in config.env_name:
        eval_env = gym.make(config.env_name)

    # 모델 생성
    model = create_model(config, env, use_json_config=use_json_config)

    # 콜백 생성
    callbacks = create_callbacks(config, eval_env)

    # 학습
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            progress_bar=True,
            tb_log_name=tb_log_name,
        )
    except KeyboardInterrupt:
        print("\n⚠️  학습이 중단되었습니다.")

    # 모델 저장
    model_path = config.get_model_path("final")
    model.save(model_path)
    print(f"\n✅ 모델 저장 완료: {model_path}.zip")

    # 평가
    print("\n" + "=" * 70)
    print("📊 최종 평가 중...")
    print("=" * 70)

    if eval_env is None:
        eval_env = gym.make(config.env_name)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
    )

    print(f"\n평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 환경 정리
    env.close()
    if eval_env is not None:
        eval_env.close()

    return model, mean_reward, std_reward


def test_model(config: TrainingConfig, model_path: Optional[str] = None, n_episodes: int = 3, n_steps: Optional[int] = None):
    """학습된 모델 테스트

    Args:
        config: TrainingConfig 객체
        model_path: 모델 파일 경로 (None이면 자동으로 찾음)
        n_episodes: 테스트할 에피소드 수
        n_steps: 각 에피소드당 최대 스텝 수 (None이면 제한 없음)

    Returns:
        mean_reward, std_reward: 평균 보상과 표준편차
    """
    if model_path is None:
        model_path = config.get_model_path("final")

    # 모델 로드
    AlgoClass = get_algorithm_class(config.algorithm)
    model = AlgoClass.load(model_path)

    # 환경 생성 (렌더링 모드)
    render_mode = "human" if config.render else None
    if "ALE/" in config.env_name:
        env = gym.make(config.env_name, render_mode=render_mode)
        env = AtariWrapper(env)
    else:
        env = gym.make(config.env_name, render_mode=render_mode)

    print(f"\n🎮 학습된 모델로 {n_episodes}개 에피소드 실행 중...")
    if n_steps:
        print(f"   (각 에피소드 최대 {n_steps} 스텝으로 제한)")
    else:
        print(f"   (스텝 제한 없음 - 에피소드가 자연스럽게 종료될 때까지)")
    print("=" * 70)

    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        force_stopped = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            # 최대 스텝 제한 체크
            if n_steps and steps >= n_steps:
                done = True
                force_stopped = True
            else:
                done = terminated or truncated

        episode_rewards.append(total_reward)

        # 종료 원인 표시
        if force_stopped:
            status = f" (강제 종료: {n_steps} 스텝 도달)"
        elif steps < (n_steps or float('inf')):
            status = " (자연 종료)"
        else:
            status = ""

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}{status}")

    env.close()

    # 평균과 표준편차 계산
    import numpy as np
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("=" * 70)
    print(f"📊 평균 보상: {mean_reward:.2f} ± {std_reward:.2f}")
    print("=" * 70)

    return mean_reward, std_reward


def print_available_configs():
    """사용 가능한 환경과 알고리즘 출력"""
    from config import ENVIRONMENT_CONFIGS

    print("\n" + "=" * 70)
    print("📋 사용 가능한 환경 및 알고리즘")
    print("=" * 70)

    for env_name, env_config in ENVIRONMENT_CONFIGS.items():
        print(f"\n🎮 {env_name}")
        print(f"   - Action Space: {env_config['action_space']}")
        print(f"   - State Space: {env_config['state_space']}")
        print(f"   - Success Threshold: {env_config['success_threshold']}")
        print(f"   - Supported Algorithms: {', '.join(env_config['supported_algos'])}")

    print("\n" + "=" * 70)
