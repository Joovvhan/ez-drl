#!/usr/bin/env python3
"""전체 알고리즘 x 환경 조합 학습

모든 가능한 알고리즘과 환경 조합을 짧은 시간(3분)동안 학습하여
코드가 정상적으로 작동하는지 확인합니다.

사용법:
    python train_all.py                    # 모든 조합 학습
    python train_all.py --quick            # 더 빠른 학습 (1분)
    python train_all.py --env CartPole-v1  # 특정 환경만 학습
    python train_all.py --algo ppo         # 특정 알고리즘만 학습
"""
import argparse
import time
from datetime import datetime
from config import TrainingConfig, ENVIRONMENT_CONFIGS, is_algo_env_compatible
from utils import train_model
import traceback


# 테스트 설정
TEST_DURATION_SECONDS = 180  # 3분
QUICK_TEST_DURATION_SECONDS = 60  # 1분


def get_test_timesteps(env_name: str, duration_seconds: int = TEST_DURATION_SECONDS) -> int:
    """환경별 테스트 timesteps 계산

    각 환경의 평균 step 시간을 고려하여 설정
    """
    # 환경별 대략적인 step/sec (CPU 기준)
    env_speed = {
        # "CartPole-v1": 2000,  # 매우 빠름
        "CartPole-v1": 500,  # 매우 빠름
        "LunarLander-v3": 1000,  # 빠름
        # "BipedalWalker-v3": 300,  # 중간
        "BipedalWalker-v3": 500,  # 중간
        "Ant-v4": 200,  # 느림 (MuJoCo)
        "ALE/Breakout-v5": 100,  # 매우 느림 (Atari)
    }

    steps_per_sec = env_speed.get(env_name, 500)
    return int(steps_per_sec * duration_seconds)


def run_single_training(env_name: str, algorithm: str, duration_seconds: int) -> dict:
    """단일 조합 학습"""
    print("\n" + "=" * 70)
    print(f"🧪 학습: {algorithm.upper()} x {env_name}")
    print("=" * 70)

    result = {
        "env": env_name,
        "algo": algorithm,
        "success": False,
        "error": None,
        "duration": 0,
        "mean_reward": None,
    }

    # 호환성 체크
    if not is_algo_env_compatible(algorithm, env_name):
        result["error"] = "Incompatible combination"
        print(f"⏭️  건너뛰기: {algorithm}은 {env_name}을 지원하지 않습니다.")
        return result

    try:
        start_time = time.time()

        # 학습 설정 생성
        config = TrainingConfig(
            env_name=env_name,
            algorithm=algorithm,
            total_timesteps=get_test_timesteps(env_name, duration_seconds),
            tensorboard_log=True,  # TensorBoard 로그는 항상 생성
            log_interval=100,  # 로그 출력 줄이기
            n_envs=1,  # 단일 환경
        )

        # 학습 실행
        model, mean_reward, std_reward = train_model(config)

        end_time = time.time()
        duration = end_time - start_time

        result["success"] = True
        result["duration"] = duration
        result["mean_reward"] = mean_reward

        print(f"✅ 성공! (소요 시간: {duration:.1f}초, 보상: {mean_reward:.2f})")

    except Exception as e:
        end_time = time.time()
        result["duration"] = end_time - start_time
        result["error"] = str(e)
        print(f"❌ 실패: {e}")
        traceback.print_exc()

    return result


def print_summary(results: list):
    """학습 결과 요약 출력"""
    print("\n\n" + "=" * 70)
    print("📊 학습 결과 요약")
    print("=" * 70)

    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"] and r["error"] != "Incompatible combination")
    skipped = sum(1 for r in results if r["error"] == "Incompatible combination")

    print(f"\n총 학습: {total}")
    print(f"✅ 성공: {success}")
    print(f"❌ 실패: {failed}")
    print(f"⏭️  건너뜀: {skipped}")

    if success > 0:
        print("\n✅ 성공한 학습:")
        for r in results:
            if r["success"]:
                print(f"   - {r['algo'].upper():4s} x {r['env']:25s} "
                      f"({r['duration']:.1f}초, 보상: {r['mean_reward']:.2f})")

    if failed > 0:
        print("\n❌ 실패한 학습:")
        for r in results:
            if not r["success"] and r["error"] != "Incompatible combination":
                print(f"   - {r['algo'].upper():4s} x {r['env']:25s} - {r['error']}")

    print("\n" + "=" * 70)

    # TensorBoard 및 테스트 안내
    print("\n💡 TensorBoard로 모든 결과를 비교하세요:")
    print("   tensorboard --logdir results/")
    print("\n💡 특정 환경만 보려면:")
    print("   tensorboard --logdir results/cartpole_v1/")
    print("\n💡 학습된 모델을 테스트하려면:")
    print("   python test_all.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ez-drl 전체 학습 스크립트")
    parser.add_argument(
        "--env",
        type=str,
        help="특정 환경만 학습 (예: CartPole-v1)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "a2c", "ppo", "sac"],
        help="특정 알고리즘만 학습"
    )

    parser.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="빠른 학습 모드"
    )

    parser.add_argument(
        "--exclude-atari",
        action="store_true",
        help="Atari 환경 제외 (시간 절약)"
    )

    args = parser.parse_args()

    # 학습 지속 시간 설정
    duration = QUICK_TEST_DURATION_SECONDS if args.quick else TEST_DURATION_SECONDS

    print("=" * 70)
    print("🧪 ez-drl 전체 학습 시작")
    print("=" * 70)
    print(f"⏱️  각 학습 시간: {duration}초")
    print(f"📅 시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 학습할 환경과 알고리즘 결정
    if args.env:
        environments = [args.env]
    else:
        environments = list(ENVIRONMENT_CONFIGS.keys())
        if args.exclude_atari:
            environments = [env for env in environments if "ALE/" not in env]

    if args.algo:
        algorithms = [args.algo]
    else:
        algorithms = ["dqn", "a2c", "ppo", "sac"]

    # 학습 실행
    results = []
    total_trainings = len(environments) * len(algorithms)
    current_training = 0

    for env_name in environments:
        for algorithm in algorithms:
            current_training += 1
            print(f"\n진행: {current_training}/{total_trainings}")

            result = run_single_training(env_name, algorithm, duration)
            results.append(result)

    # 결과 요약
    print_summary(results)

    # 종료 코드
    failed_count = sum(1 for r in results if not r["success"] and r["error"] != "Incompatible combination")
    exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
