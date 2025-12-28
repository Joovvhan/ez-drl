#!/usr/bin/env python3
"""학습된 모델 테스트 스크립트

results/ 폴더를 순회하여 학습된 모델을 찾고,
각 모델을 렌더링하며 테스트합니다.

사용법:
    python test_all.py                    # 모든 학습된 모델 테스트
    python test_all.py --env CartPole-v1  # 특정 환경만 테스트
    python test_all.py --algo ppo         # 특정 알고리즘만 테스트
    python test_all.py --steps 500        # 각 모델당 테스트 스텝 수 (기본: 1000)
"""
import argparse
import time
from pathlib import Path
from typing import List, Dict
from config import TrainingConfig
from utils import test_model
import traceback


def find_trained_models(results_dir: str = "results") -> List[Dict[str, str]]:
    """results 폴더에서 학습된 모델을 찾습니다.

    폴더 구조: results/{env}/{algo}/
    모델 파일은 algo 폴더에 직접 저장되어 있음

    Returns:
        List of dict with keys: env, algo, timestamp, model_path
    """
    models = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"⚠️  결과 폴더가 없습니다: {results_dir}")
        return models

    # results/{env}/{algo}/ 구조 탐색
    for env_dir in sorted(results_path.iterdir()):
        if not env_dir.is_dir():
            continue

        env_name = env_dir.name

        for algo_dir in sorted(env_dir.iterdir()):
            if not algo_dir.is_dir():
                continue

            algo_name = algo_dir.name

            # algo 폴더에서 직접 .zip 모델 파일 찾기
            # 우선순위: *_final.zip > best_model.zip > 첫 번째 .zip 파일
            final_model = list(algo_dir.glob("*_final.zip"))
            best_model = list(algo_dir.glob("best_model.zip"))
            all_models = list(algo_dir.glob("*.zip"))

            model_file = None
            if final_model:
                model_file = final_model[0]
            elif best_model:
                model_file = best_model[0]
            elif all_models:
                model_file = all_models[0]

            if model_file:
                # 타임스탬프 폴더가 있으면 가져오기 (표시용)
                timestamp_dirs = [d for d in algo_dir.iterdir() if d.is_dir()]
                timestamp = max(timestamp_dirs, key=lambda d: d.name).name if timestamp_dirs else "unknown"

                models.append({
                    "env": env_name,
                    "algo": algo_name,
                    "timestamp": timestamp,
                    "model_path": str(model_file),
                })

    return models


def get_default_test_steps(env_name: str) -> int:
    """환경별 기본 테스트 스텝 수 반환

    Args:
        env_name: 표준 환경 이름 (예: CartPole-v1, Ant-v4)

    Returns:
        적절한 테스트 스텝 수
    """
    # 환경별 기본 스텝 수
    if "Ant" in env_name or "Humanoid" in env_name:
        return 300  # MuJoCo 복잡한 환경 - 매우 느림
    elif "BipedalWalker" in env_name or "LunarLander" in env_name:
        return 500  # 중간 복잡도
    elif "CartPole" in env_name:
        return 500  # 간단한 환경 (최대 500 스텝 제한)
    elif "ALE/" in env_name or "Atari" in env_name:
        return 1000  # Atari 게임
    else:
        return 1000  # 기본값


def env_name_to_standard(env_dir_name: str) -> str:
    """폴더 이름을 표준 환경 이름으로 변환

    cartpole_v1 -> CartPole-v1
    lunarlander_v3 -> LunarLander-v3
    bipedalwalker_v3 -> BipedalWalker-v3
    ale_breakout_v5 -> ALE/Breakout-v5
    """
    # 환경 이름 매핑 (폴더 이름 -> 표준 이름)
    env_mapping = {
        "cartpole": "CartPole",
        "lunarlander": "LunarLander",
        "bipedalwalker": "BipedalWalker",
        "ant": "Ant",
        "breakout": "Breakout",
        "pong": "Pong",
        "spaceinvaders": "SpaceInvaders",
    }

    # 특수 케이스: ALE
    if env_dir_name.startswith("ale_"):
        # ale_breakout_v5 -> ALE/Breakout-v5
        parts = env_dir_name.split("_")
        if len(parts) >= 3:
            game_base = "_".join(parts[1:-1])  # breakout
            game_name = env_mapping.get(game_base, game_base.capitalize())
            version = parts[-1].upper().replace("V", "-v")
            return f"ALE/{game_name}{version}"

    # 일반 케이스
    parts = env_dir_name.split("_")
    if len(parts) >= 2:
        # 마지막 부분이 버전 (v1, v2, v3 등)
        env_base = "_".join(parts[:-1])  # cartpole, lunarlander, etc.
        env_name = env_mapping.get(env_base, env_base.capitalize())
        version = parts[-1].upper().replace("V", "-v")
        return f"{env_name}{version}"

    return env_dir_name


def test_single_model(model_info: Dict[str, str], n_steps: int = 1000) -> Dict:
    """단일 모델 테스트"""
    env_name = env_name_to_standard(model_info["env"])
    algo_name = model_info["algo"]
    model_path = model_info["model_path"]

    print("\n" + "=" * 70)
    print(f"🎮 테스트: {algo_name.upper()} x {env_name}")
    print(f"   모델: {model_info['timestamp']}")
    print("=" * 70)

    result = {
        "env": env_name,
        "algo": algo_name,
        "timestamp": model_info["timestamp"],
        "success": False,
        "error": None,
        "mean_reward": None,
        "std_reward": None,
    }

    try:
        # 설정 생성
        config = TrainingConfig(
            env_name=env_name,
            algorithm=algo_name,
            render=True,
            render_mode="human",
        )

        # 모델 테스트 (렌더링 포함, n_steps만큼 실행)
        mean_reward, std_reward = test_model(
            config,
            model_path=model_path,
            n_episodes=1,
            n_steps=n_steps,
        )

        result["success"] = True
        result["mean_reward"] = mean_reward
        result["std_reward"] = std_reward

        print(f"✅ 평균 보상: {mean_reward:.2f} ± {std_reward:.2f}")

    except KeyboardInterrupt:
        print("\n⚠️  사용자가 테스트를 중단했습니다.")
        raise
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ 실패: {e}")
        traceback.print_exc()

    return result


def print_model_list(models: List[Dict[str, str]]):
    """찾은 모델 목록 출력"""
    print("\n📋 발견된 학습 모델:")
    print("=" * 70)

    if not models:
        print("   (없음)")
        return

    for i, model in enumerate(models, 1):
        env_name = env_name_to_standard(model["env"])
        print(f"{i:2d}. {model['algo'].upper():4s} x {env_name:25s} ({model['timestamp']})")

    print("=" * 70)


def print_summary(results: List[Dict]):
    """테스트 결과 요약"""
    print("\n\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)

    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success

    print(f"\n총 테스트: {total}")
    print(f"✅ 성공: {success}")
    print(f"❌ 실패: {failed}")

    if success > 0:
        print("\n✅ 성공한 테스트:")
        for r in results:
            if r["success"]:
                print(f"   - {r['algo'].upper():4s} x {r['env']:25s} "
                      f"평균 보상: {r['mean_reward']:7.2f} ± {r['std_reward']:.2f}")

    if failed > 0:
        print("\n❌ 실패한 테스트:")
        for r in results:
            if not r["success"]:
                print(f"   - {r['algo'].upper():4s} x {r['env']:25s} - {r['error']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="학습된 모델 테스트 스크립트")
    parser.add_argument(
        "--env",
        type=str,
        help="특정 환경만 테스트 (예: CartPole-v1)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "a2c", "ppo", "sac"],
        help="특정 알고리즘만 테스트"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="각 에피소드당 최대 스텝 수 (기본: 환경별 자동 설정 - Ant/Humanoid:300, BipedalWalker/LunarLander:500, CartPole:500, Atari:1000)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="결과 폴더 경로 (기본: results)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎮 ez-drl 모델 테스트")
    print("=" * 70)

    # 학습된 모델 찾기
    print(f"\n🔍 {args.results_dir} 폴더에서 모델을 검색 중...")
    models = find_trained_models(args.results_dir)

    # 필터링
    if args.env:
        env_dir_name = args.env.lower().replace("-", "_").replace("/", "_")
        models = [m for m in models if m["env"] == env_dir_name]

    if args.algo:
        models = [m for m in models if m["algo"] == args.algo]

    # 모델 목록 출력
    print_model_list(models)

    if not models:
        print("\n⚠️  테스트할 모델이 없습니다.")
        print("💡 먼저 train.py 또는 train_all.py로 모델을 학습하세요.")
        return

    print(f"\n총 {len(models)}개의 모델을 순서대로 테스트합니다.")
    if args.steps is not None:
        print(f"각 모델당 1개 에피소드, 최대 {args.steps} 스텝씩 실행됩니다.")
    else:
        print(f"각 모델당 1개 에피소드, 환경별 최적 스텝 수로 실행됩니다.")
        print(f"  (Ant/Humanoid: 300, BipedalWalker/LunarLander: 500, CartPole: 500, Atari: 1000)")
    print("\n💡 테스트를 중단하려면 Ctrl+C를 누르세요.")

    try:
        input("\nEnter를 눌러 시작하세요...")
    except KeyboardInterrupt:
        print("\n\n취소되었습니다.")
        return

    # 테스트 실행
    results = []

    try:
        for i, model in enumerate(models, 1):
            print(f"\n\n진행: {i}/{len(models)}")

            # 환경별 기본 스텝 수 결정
            env_name = env_name_to_standard(model["env"])
            n_steps = args.steps if args.steps is not None else get_default_test_steps(env_name)

            print(f"💡 {env_name}: 최대 {n_steps} 스텝으로 테스트")

            result = test_single_model(model, n_steps=n_steps)
            results.append(result)

            # 다음 모델로 넘어가기 전 잠시 대기
            if i < len(models):
                print("\n다음 모델로 넘어갑니다...")
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 테스트를 중단했습니다.")

    # 결과 요약
    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
