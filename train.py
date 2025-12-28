#!/usr/bin/env python3
"""공통 학습 진입점

모든 알고리즘과 환경에 대한 통합 학습 스크립트입니다.

사용 예시:
    # 기본 실행 (대화형 모드)
    python train.py

    # 명령줄 인자로 실행
    python train.py --env CartPole-v1 --algo ppo --timesteps 100000

    # 병렬 환경 사용
    python train.py --env CartPole-v1 --algo ppo --n-envs 4

    # 네트워크 구조 커스터마이징
    python train.py --env CartPole-v1 --algo ppo --net-arch 256 256

    # 시드 고정
    python train.py --env CartPole-v1 --algo ppo --seed 42
"""
import argparse
import sys
from config import TrainingConfig, ENVIRONMENT_CONFIGS
from utils import train_model, test_model, print_available_configs


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="ez-drl 통합 학습 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python train.py --env CartPole-v1 --algo ppo
  python train.py --env LunarLander-v3 --algo dqn --timesteps 500000
  python train.py --env BipedalWalker-v3 --algo sac --n-envs 4
  python train.py --list  # 사용 가능한 환경 목록 보기
        """
    )

    # 기본 인자
    parser.add_argument(
        "--env",
        type=str,
        help="환경 이름 (예: CartPole-v1, LunarLander-v3)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "a2c", "ppo", "sac"],
        help="알고리즘 (dqn, a2c, ppo, sac)"
    )

    # 학습 설정
    parser.add_argument(
        "--timesteps",
        type=int,
        help="총 학습 timesteps (기본값: 환경별 자동 설정)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="학습률 (기본값: 3e-4)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="병렬 환경 수 (기본값: 1, A2C/PPO만 지원)"
    )

    # 네트워크 설정
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        help="네트워크 구조 (예: --net-arch 256 256)"
    )

    # 로깅 및 저장
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="결과 저장 디렉토리 (기본값: results)"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="TensorBoard 로깅 비활성화"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="로그 출력 간격 (기본값: 10)"
    )

    # 평가
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="평가 에피소드 수 (기본값: 10)"
    )

    # 시드
    parser.add_argument(
        "--seed",
        type=int,
        help="랜덤 시드"
    )

    # 유틸리티
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 환경 및 알고리즘 목록 출력"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="학습 후 모델 테스트 (렌더링)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="학습 중 환경을 화면에 렌더링 (시각화)"
    )

    # 고급 옵션
    parser.add_argument(
        "--policy",
        type=str,
        default="MlpPolicy",
        choices=["MlpPolicy", "CnnPolicy"],
        help="정책 타입 (기본값: MlpPolicy)"
    )

    return parser.parse_args()


def interactive_mode():
    """대화형 모드로 설정 입력"""
    print("\n" + "=" * 70)
    print("🎮 ez-drl 대화형 학습 모드")
    print("=" * 70)

    # 환경 선택
    print("\n📋 사용 가능한 환경:")
    envs = list(ENVIRONMENT_CONFIGS.keys())
    for i, env in enumerate(envs, 1):
        env_config = ENVIRONMENT_CONFIGS[env]
        print(f"  {i}. {env}")
        print(f"     - Action: {env_config['action_space']}, State: {env_config['state_space']}")
        print(f"     - Algorithms: {', '.join(env_config['supported_algos'])}")

    while True:
        try:
            env_idx = int(input(f"\n환경 선택 (1-{len(envs)}): ")) - 1
            if 0 <= env_idx < len(envs):
                env_name = envs[env_idx]
                break
            else:
                print("❌ 잘못된 선택입니다. 다시 입력해주세요.")
        except (ValueError, KeyboardInterrupt):
            print("\n\n⚠️  취소되었습니다.")
            sys.exit(0)

    # 알고리즘 선택
    supported_algos = ENVIRONMENT_CONFIGS[env_name]["supported_algos"]
    print(f"\n🤖 {env_name}에서 사용 가능한 알고리즘:")
    for i, algo in enumerate(supported_algos, 1):
        print(f"  {i}. {algo.upper()}")

    while True:
        try:
            algo_idx = int(input(f"\n알고리즘 선택 (1-{len(supported_algos)}): ")) - 1
            if 0 <= algo_idx < len(supported_algos):
                algorithm = supported_algos[algo_idx]
                break
            else:
                print("❌ 잘못된 선택입니다. 다시 입력해주세요.")
        except (ValueError, KeyboardInterrupt):
            print("\n\n⚠️  취소되었습니다.")
            sys.exit(0)

    # 추가 설정
    print("\n⚙️  추가 설정 (Enter를 누르면 기본값 사용):")

    # Timesteps
    default_timesteps = TrainingConfig(env_name=env_name)._get_default_timesteps()
    timesteps_input = input(f"  총 timesteps (기본값: {default_timesteps:,}): ").strip()
    total_timesteps = int(timesteps_input) if timesteps_input else default_timesteps

    # 병렬 환경
    if algorithm in ["a2c", "ppo"]:
        n_envs_input = input("  병렬 환경 수 (기본값: 1): ").strip()
        n_envs = int(n_envs_input) if n_envs_input else 1
    else:
        n_envs = 1

    # 시드
    seed_input = input("  랜덤 시드 (기본값: None): ").strip()
    seed = int(seed_input) if seed_input else None

    print("\n" + "=" * 70)

    # 설정 생성
    config = TrainingConfig(
        env_name=env_name,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        seed=seed,
    )

    return config


def main():
    """메인 함수"""
    args = parse_args()

    # 목록 출력 모드
    if args.list:
        print_available_configs()
        return

    # 대화형 모드 또는 명령줄 모드
    if args.env is None or args.algo is None:
        # 인자가 없으면 대화형 모드
        config = interactive_mode()
    else:
        # 렌더링과 병렬 환경 충돌 체크
        n_envs = args.n_envs
        render = args.render
        if args.render and args.n_envs > 1:
            print("\n⚠️  경고: 병렬 환경에서는 렌더링을 지원하지 않습니다.")
            print("   렌더링을 활성화하려면 --n-envs를 1로 설정하거나,")
            print("   병렬 환경을 사용하려면 --render를 제거하세요.")
            user_choice = input("\n선택: (1) 렌더링 비활성화 (2) n-envs=1로 변경 (3) 취소 [1/2/3]: ").strip()
            if user_choice == "2":
                n_envs = 1
                print("✓ n-envs를 1로 변경했습니다.\n")
            elif user_choice == "3":
                print("취소되었습니다.")
                sys.exit(0)
            else:
                render = False
                print("✓ 렌더링을 비활성화했습니다.\n")

        # 명령줄 인자로 설정 생성
        config = TrainingConfig(
            env_name=args.env,
            algorithm=args.algo,
            total_timesteps=args.timesteps if args.timesteps else 50000,
            learning_rate=args.learning_rate,
            n_envs=n_envs,
            net_arch=args.net_arch,
            save_dir=args.save_dir,
            tensorboard_log=not args.no_tensorboard,
            log_interval=args.log_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            policy_type=args.policy,
            render=render,
            render_mode="human" if render else None,
        )

    # 학습 실행
    try:
        model, mean_reward, std_reward = train_model(config)

        print("\n" + "=" * 70)
        print("✅ 학습 완료!")
        print(f"📊 최종 평균 보상: {mean_reward:.2f} ± {std_reward:.2f}")
        print("=" * 70)

        # 테스트 모드
        if args.test:
            print("\n🎮 학습된 모델을 테스트합니다...")
            test_model(config, n_episodes=3)

        # TensorBoard 안내
        if config.tensorboard_log:
            print(f"\n💡 TensorBoard로 학습 결과를 확인하세요:")
            print(f"   tensorboard --logdir {config.tb_log_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  학습이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
