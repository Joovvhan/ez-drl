"""공통 설정 파일

이 파일은 모든 알고리즘과 환경에 대한 설정을 정의합니다.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """학습 설정"""

    # 환경 설정
    env_name: str = "CartPole-v1"
    n_envs: int = 1  # 병렬 환경 수 (A2C, PPO에서 사용)

    # 알고리즘 설정
    algorithm: str = "ppo"  # dqn, a2c, ppo, sac

    # 학습 설정
    total_timesteps: int = 50000
    learning_rate: float = 3e-4

    # 네트워크 설정
    policy_type: str = "MlpPolicy"  # MlpPolicy, CnnPolicy
    net_arch: Optional[list] = None  # None이면 기본값 사용

    # 저장 및 로깅
    save_dir: str = "results"
    tensorboard_log: bool = True
    log_interval: int = 10

    # 렌더링
    render: bool = False
    render_mode: Optional[str] = None  # "human", "rgb_array", None

    # 평가
    eval_episodes: int = 10

    # 시드
    seed: Optional[int] = None

    # 추가 알고리즘별 파라미터
    algo_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """설정 검증 및 기본값 설정"""
        # 환경별 기본 timesteps 설정
        if self.total_timesteps == 50000:  # 기본값인 경우
            self.total_timesteps = self._get_default_timesteps()

        # 결과 저장 디렉토리 생성: results/{env}/{algo}
        self.results_dir = os.path.join(
            self.save_dir,
            self.env_name.lower().replace("-", "_").replace("/", "_"),
            self.algorithm
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # TensorBoard 로그 디렉토리: results/{env}/{algo}
        if self.tensorboard_log:
            self.tb_log_dir = self.results_dir
        else:
            self.tb_log_dir = None

    def _get_default_timesteps(self) -> int:
        """환경별 기본 timesteps 반환"""
        env_timesteps = {
            "CartPole-v1": 50000,
            "LunarLander-v3": 300000,
            "BipedalWalker-v3": 2000000,
            "Ant-v4": 1000000,
            "ALE/Breakout-v5": 1000000,
        }
        return env_timesteps.get(self.env_name, 50000)

    def get_model_path(self, suffix: str = "") -> str:
        """모델 저장 경로 반환"""
        base_name = f"{self.algorithm}_{self.env_name.lower().replace('-', '_').replace('/', '_')}"
        if suffix:
            base_name += f"_{suffix}"
        return os.path.join(self.results_dir, base_name)


# 환경별 설정
ENVIRONMENT_CONFIGS = {
    "CartPole-v1": {
        "action_space": "discrete",
        "state_space": "vector",
        "success_threshold": 475,
        "supported_algos": ["dqn", "a2c", "ppo"],
    },
    "LunarLander-v3": {
        "action_space": "discrete",
        "state_space": "vector",
        "success_threshold": 200,
        "supported_algos": ["dqn", "a2c", "ppo"],
    },
    "BipedalWalker-v3": {
        "action_space": "continuous",
        "state_space": "vector",
        "success_threshold": 300,
        "supported_algos": ["a2c", "ppo", "sac"],
    },
    "Ant-v4": {
        "action_space": "continuous",
        "state_space": "vector",
        "success_threshold": 6000,
        "supported_algos": ["a2c", "ppo", "sac"],
    },
    "ALE/Breakout-v5": {
        "action_space": "discrete",
        "state_space": "image",
        "success_threshold": 400,
        "supported_algos": ["dqn", "a2c", "ppo"],
    },
}


# 알고리즘별 기본 하이퍼파라미터
ALGORITHM_DEFAULTS = {
    "dqn": {
        "CartPole-v1": {
            "learning_rate": 1e-3,
            "buffer_size": 50000,
            "batch_size": 32,
            "target_update_interval": 500,
        },
        "LunarLander-v3": {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "batch_size": 128,
            "target_update_interval": 1000,
        },
        "ALE/Breakout-v5": {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "batch_size": 32,
            "target_update_interval": 10000,
        },
    },
    "a2c": {
        "CartPole-v1": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
        },
        "LunarLander-v3": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
        },
        "BipedalWalker-v3": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
        },
    },
    "ppo": {
        "CartPole-v1": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
        },
        "LunarLander-v3": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
        },
        "BipedalWalker-v3": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
        },
        "Ant-v4": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
        },
    },
    "sac": {
        "BipedalWalker-v3": {
            "learning_rate": 3e-4,
            "buffer_size": 300000,
            "batch_size": 256,
            "ent_coef": "auto",
        },
        "Ant-v4": {
            "learning_rate": 3e-4,
            "buffer_size": 1000000,
            "batch_size": 256,
            "ent_coef": "auto",
        },
    },
}


def get_default_params(algorithm: str, env_name: str) -> Dict[str, Any]:
    """알고리즘과 환경에 맞는 기본 파라미터 반환"""
    return ALGORITHM_DEFAULTS.get(algorithm, {}).get(env_name, {})


def is_algo_env_compatible(algorithm: str, env_name: str) -> bool:
    """알고리즘과 환경의 호환성 확인"""
    env_config = ENVIRONMENT_CONFIGS.get(env_name, {})
    supported = env_config.get("supported_algos", [])
    return algorithm in supported


def load_config_from_json(algorithm: str, env_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """JSON 파일에서 하이퍼파라미터 로드

    Args:
        algorithm: 알고리즘 이름 (dqn, a2c, ppo, sac)
        env_name: 환경 이름
        config_path: 커스텀 설정 파일 경로 (None이면 기본 경로 사용)

    Returns:
        하이퍼파라미터 딕셔너리 (common + environment 병합)
    """
    # 기본 설정 파일 경로
    if config_path is None:
        config_dir = Path(__file__).parent / "configs"
        config_path = config_dir / f"{algorithm}.json"
    else:
        config_path = Path(config_path)

    # 파일이 없으면 빈 딕셔너리 반환
    if not config_path.exists():
        print(f"⚠️  설정 파일을 찾을 수 없습니다: {config_path}")
        print(f"   기본 파라미터를 사용합니다.")
        return {}

    # JSON 로드
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {config_path}")
        print(f"   {e}")
        return {}

    # common 파라미터 가져오기
    params = config.get("common", {}).copy()

    # 환경별 파라미터 병합 (덮어쓰기)
    env_params = config.get("environments", {}).get(env_name, {})
    params.update(env_params)

    return params


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """두 설정을 병합 (override가 우선순위)

    Args:
        base_config: 기본 설정
        override_config: 오버라이드할 설정

    Returns:
        병합된 설정
    """
    merged = base_config.copy()

    # None이 아닌 값만 덮어쓰기
    for key, value in override_config.items():
        if value is not None:
            merged[key] = value

    return merged
