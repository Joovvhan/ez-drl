#!/usr/bin/env python3
"""알고리즘별 통합 학습 스크립트

이 스크립트는 각 알고리즘 폴더의 개별 스크립트를 대체합니다.
JSON 설정 파일을 사용하여 하이퍼파라미터를 관리합니다.

사용법:
    python algorithms/run_algorithm.py --algo dqn --env CartPole-v1
    python algorithms/run_algorithm.py --algo ppo --env LunarLander-v3 --n-envs 4
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TrainingConfig, load_config_from_json
from utils import train_model, test_model
import argparse


def main():
    parser = argparse.ArgumentParser(description="알고리즘별 통합 학습 스크립트")

    # 필수 인자
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["dqn", "a2c", "ppo", "sac"],
        help="알고리즘 (dqn, a2c, ppo, sac)"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="환경 이름 (예: CartPole-v1)"
    )

    # 선택 인자
    parser.add_argument("--n-envs", type=int, default=1, help="병렬 환경 수")
    parser.add_argument("--seed", type=int, help="랜덤 시드")
    parser.add_argument("--test", action="store_true", help="학습 후 테스트")
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="JSON 설정 사용 안 함 (코드 기본값)"
    )

    args = parser.parse_args()

    # JSON에서 total_timesteps 로드
    json_config = load_config_from_json(args.algo, args.env)
    total_timesteps = json_config.get("total_timesteps", 50000)

    # 설정 생성
    config = TrainingConfig(
        env_name=args.env,
        algorithm=args.algo,
        total_timesteps=total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    # 학습
    print("\n" + "=" * 70)
    print(f"알고리즘: {args.algo.upper()}")
    print(f"환경: {args.env}")
    print(f"JSON 설정: {'비활성화' if args.no_json else '활성화'}")
    print("=" * 70 + "\n")

    model, mean_reward, std_reward = train_model(
        config,
        use_json_config=not args.no_json
    )

    print("\n" + "=" * 70)
    print("✅ 학습 완료!")
    print(f"📊 최종 평균 보상: {mean_reward:.2f} ± {std_reward:.2f}")
    print("=" * 70)

    # 테스트
    if args.test:
        print("\n🎮 학습된 모델을 테스트합니다...")
        test_model(config, n_episodes=3)


if __name__ == "__main__":
    main()
