#!/usr/bin/env python3
"""환경을 직접 플레이하는 스크립트

사용자가 키보드로 환경을 직접 제어하여 RL 문제를 체험할 수 있습니다.

사용법:
    python play.py                          # 환경 선택 메뉴 표시
    python play.py --env CartPole-v1        # 특정 환경 바로 플레이
    python play.py --env ALE/Breakout-v5    # Atari 게임 플레이
    python play.py --fps 15                 # 느린 속도로 플레이 (기본: 30 FPS)
    python play.py --fps 60                 # 빠른 속도로 플레이
"""
import argparse
import gymnasium as gym
import numpy as np
from config import ENVIRONMENT_CONFIGS

# ALE 환경 등록
try:
    import ale_py
    from gymnasium.envs import registration
    registration.register_envs(ale_py)
except ImportError:
    pass


# 환경별 키 매핑
KEY_MAPPINGS = {
    # CartPole: 좌우로 카트 이동
    "CartPole-v1": {
        "description": "막대기가 쓰러지지 않도록 카트를 좌우로 움직이세요",
        "keys": {
            "a": {"action": 0, "description": "왼쪽으로 밀기"},
            "d": {"action": 1, "description": "오른쪽으로 밀기"},
        },
        "default_action": 0,
    },

    # LunarLander: 로켓 엔진 제어
    "LunarLander-v3": {
        "description": "달 착륙선을 안전하게 착륙시키세요",
        "keys": {
            "w": {"action": 2, "description": "메인 엔진 (위)"},
            "a": {"action": 1, "description": "왼쪽 엔진"},
            "d": {"action": 3, "description": "오른쪽 엔진"},
            "s": {"action": 0, "description": "아무것도 안 함"},
        },
        "default_action": 0,
    },

    # BipedalWalker: 연속 제어 (간단화)
    "BipedalWalker-v3": {
        "description": "2족 보행 로봇이 걷도록 관절을 제어하세요",
        "keys": {
            "w": {"action": [1.0, 1.0, 1.0, 1.0], "description": "전진 (모든 관절 앞으로)"},
            "s": {"action": [-1.0, -1.0, -1.0, -1.0], "description": "후진 (모든 관절 뒤로)"},
            "a": {"action": [1.0, -1.0, 1.0, -1.0], "description": "왼쪽 다리 강화"},
            "d": {"action": [-1.0, 1.0, -1.0, 1.0], "description": "오른쪽 다리 강화"},
            "space": {"action": [0.0, 0.0, 0.0, 0.0], "description": "중립"},
        },
        "default_action": [0.0, 0.0, 0.0, 0.0],
        "note": "⚠️  연속 제어는 어렵습니다. 키보드로는 제한적입니다.",
    },

    # Ant: MuJoCo 연속 제어 (매우 복잡, 간단화)
    "Ant-v4": {
        "description": "4족 로봇이 걷도록 8개 관절을 제어하세요",
        "keys": {
            "w": {"action": [1.0] * 8, "description": "전진 (모든 관절 앞으로)"},
            "s": {"action": [-1.0] * 8, "description": "후진 (모든 관절 뒤로)"},
            "a": {"action": [1.0, -1.0] * 4, "description": "왼쪽 회전"},
            "d": {"action": [-1.0, 1.0] * 4, "description": "오른쪽 회전"},
            "space": {"action": [0.0] * 8, "description": "중립"},
        },
        "default_action": [0.0] * 8,
        "note": "⚠️  매우 복잡한 제어입니다. AI가 학습하기도 어려운 환경입니다.",
    },

    # Atari Breakout: 패들 좌우 이동
    "ALE/Breakout-v5": {
        "description": "패들을 움직여 공으로 벽돌을 깨세요",
        "keys": {
            "space": {"action": 1, "description": "게임 시작 (FIRE)"},
            "a": {"action": 3, "description": "왼쪽으로 이동"},
            "d": {"action": 2, "description": "오른쪽으로 이동"},
        },
        "default_action": 0,
    },
}


def print_banner():
    """배너 출력"""
    print("=" * 70)
    print("🎮 ez-drl 환경 플레이어")
    print("=" * 70)
    print()


def print_controls(env_name: str):
    """환경별 조작법 출력"""
    if env_name not in KEY_MAPPINGS:
        print("⚠️  이 환경은 수동 플레이를 지원하지 않습니다.")
        return False

    mapping = KEY_MAPPINGS[env_name]

    print()
    print("=" * 70)
    print(f"🎯 목표: {mapping['description']}")
    print("=" * 70)
    print()
    print("⌨️  조작법:")
    for key, info in mapping["keys"].items():
        print(f"   [{key}]: {info['description']}")
    print(f"   [q]: 종료")
    print()

    if "note" in mapping:
        print(mapping["note"])
        print()

    return True


def select_environment():
    """환경 선택 메뉴"""
    print("사용 가능한 환경:")
    print()

    envs = list(ENVIRONMENT_CONFIGS.keys())
    for i, env_name in enumerate(envs, 1):
        # 플레이 가능 여부 표시
        playable = "✅" if env_name in KEY_MAPPINGS else "❌"
        print(f"  {i}. {playable} {env_name}")

    print()
    print("(✅ = 플레이 가능, ❌ = 플레이 불가)")
    print()

    while True:
        try:
            choice = input("환경을 선택하세요 (번호 입력, q=종료): ").strip()

            if choice.lower() == 'q':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(envs):
                env_name = envs[idx]
                if env_name in KEY_MAPPINGS:
                    return env_name
                else:
                    print(f"❌ {env_name}은(는) 플레이를 지원하지 않습니다. 다시 선택하세요.\n")
            else:
                print("❌ 잘못된 번호입니다. 다시 시도하세요.\n")
        except ValueError:
            print("❌ 숫자를 입력하세요.\n")
        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            return None


def get_keyboard_action(env_name: str, key_state: dict) -> any:
    """키보드 입력을 액션으로 변환"""
    mapping = KEY_MAPPINGS[env_name]

    for key, info in mapping["keys"].items():
        if key_state.get(key, False):
            return info["action"]

    return mapping["default_action"]


def play_environment(env_name: str, max_steps: int = 1000, fps: int = 30):
    """환경 플레이

    Args:
        env_name: 환경 이름
        max_steps: 최대 스텝 수
        fps: 초당 프레임 수 (게임 속도 제어)
    """
    # 환경 생성
    print(f"\n🚀 {env_name} 환경을 시작합니다...\n")

    try:
        env = gym.make(env_name, render_mode="human")
    except Exception as e:
        print(f"❌ 환경 생성 실패: {e}")
        return

    # 조작법 출력
    if not print_controls(env_name):
        env.close()
        return

    print("💡 게임을 시작하려면 Enter를 누르세요...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n취소되었습니다.")
        env.close()
        return

    # pygame이 필요한 환경의 경우
    try:
        import pygame
        pygame.init()
        clock = pygame.time.Clock()
        use_pygame = True
    except ImportError:
        print("⚠️  pygame이 설치되지 않아 키보드 입력이 제한적입니다.")
        print("    pip install pygame 를 설치하면 더 나은 경험을 할 수 있습니다.")
        use_pygame = False
        clock = None

    # 게임 루프
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    mapping = KEY_MAPPINGS[env_name]

    print("\n게임 시작!")
    print(f"⚙️  게임 속도: {fps} FPS (--fps 옵션으로 조절 가능)")
    print("=" * 70)

    try:
        import time
        start_time = time.time()

        while not done and steps < max_steps:
            # 키보드 입력 받기 (pygame 사용)
            action = mapping["default_action"]

            if use_pygame:
                # pygame 이벤트 처리
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break

                # 현재 눌린 키 확인
                keys = pygame.key.get_pressed()
                key_state = {
                    'w': keys[pygame.K_w],
                    'a': keys[pygame.K_a],
                    's': keys[pygame.K_s],
                    'd': keys[pygame.K_d],
                    'space': keys[pygame.K_SPACE],
                    'q': keys[pygame.K_q],
                }

                if key_state['q']:
                    print("\n종료합니다.")
                    break

                action = get_keyboard_action(env_name, key_state)

            # 액션 실행
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # FPS 제어 (pygame clock 사용)
            if use_pygame and clock:
                actual_fps = clock.get_fps()
                clock.tick(fps)
            else:
                # pygame이 없으면 time.sleep으로 대체
                time.sleep(1.0 / fps)
                actual_fps = 0

            # 정보 출력 (10 스텝마다)
            if steps % 10 == 0:
                elapsed = time.time() - start_time
                if actual_fps > 0:
                    print(f"Step: {steps:4d} | Reward: {total_reward:8.2f} | FPS: {actual_fps:5.1f} | Time: {elapsed:5.1f}s", end="\r")
                else:
                    print(f"Step: {steps:4d} | Reward: {total_reward:8.2f} | Time: {elapsed:5.1f}s", end="\r")

        print()  # 줄바꿈
        print("=" * 70)
        print(f"\n🏁 게임 종료!")
        print(f"   총 스텝: {steps}")
        print(f"   총 보상: {total_reward:.2f}")

        if terminated:
            print(f"   종료 사유: 환경 목표 달성 또는 실패")
        elif truncated:
            print(f"   종료 사유: 최대 스텝 도달")
        else:
            print(f"   종료 사유: 사용자 중단")

        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 게임을 중단했습니다.")
    finally:
        env.close()
        if use_pygame:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="환경을 직접 플레이하는 스크립트")
    parser.add_argument(
        "--env",
        type=str,
        help="플레이할 환경 이름 (예: CartPole-v1, ALE/Breakout-v5)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="최대 스텝 수 (기본: 1000)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        # default=30,
        default=5,
        help="게임 속도 (FPS, 기본: 5). 낮을수록 느림 (예: 15=느리게, 60=빠르게)"
    )

    args = parser.parse_args()

    print_banner()

    # 환경 선택
    if args.env:
        env_name = args.env
        if env_name not in KEY_MAPPINGS:
            print(f"❌ {env_name}은(는) 플레이를 지원하지 않습니다.")
            print("\n지원되는 환경:")
            for name in KEY_MAPPINGS.keys():
                print(f"  - {name}")
            return
    else:
        env_name = select_environment()
        if env_name is None:
            return

    # 환경 플레이
    play_environment(env_name, max_steps=args.steps, fps=args.fps)

    print("\n감사합니다! 🎮")


if __name__ == "__main__":
    main()
