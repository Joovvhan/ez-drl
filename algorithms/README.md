# 알고리즘 학습 스크립트

이 디렉토리에는 JSON 설정 기반의 통합 학습 스크립트가 포함되어 있습니다.

## 파일 구조

```
algorithms/
├── run_algorithm.py   # JSON 기반 통합 학습 스크립트
└── __init__.py
```

## 사용법

### 기본 실행

```bash
# JSON 설정 자동 로드
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1
python algorithms/run_algorithm.py --algo dqn --env LunarLander-v2
python algorithms/run_algorithm.py --algo sac --env BipedalWalker-v3

# 학습 후 자동 테스트
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1 --test

# 병렬 환경 사용 (A2C, PPO)
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1 --n-envs 4

# 랜덤 시드 고정 (재현성)
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1 --seed 42
```

### JSON 설정 관리

하이퍼파라미터는 `configs/` 디렉토리의 JSON 파일에서 관리됩니다:

```
configs/
├── dqn.json    # DQN 하이퍼파라미터
├── a2c.json    # A2C 하이퍼파라미터
├── ppo.json    # PPO 하이퍼파라미터
└── sac.json    # SAC 하이퍼파라미터
```

자세한 내용은 [JSON 설정 가이드](../configs/README.md)를 참고하세요.

## 통합 학습 스크립트

더 편리한 사용을 위해 프로젝트 루트의 `train.py`를 사용할 수 있습니다:

```bash
# 대화형 모드 (초보자 추천)
python train.py

# 명령줄 모드
python train.py --env CartPole-v1 --algo ppo
python train.py --env BipedalWalker-v3 --algo sac --n-envs 4

# 사용 가능한 환경 목록
python train.py --list
```

## 알고리즘별 지원 환경

### DQN (Discrete Action만)
- ✅ CartPole-v1
- ✅ LunarLander-v2
- ✅ ALE/Breakout-v5 (Atari)
- ❌ BipedalWalker-v3 (연속 제어 불가)
- ❌ Ant-v4 (연속 제어 불가)

### A2C (모든 환경)
- ✅ CartPole-v1
- ✅ LunarLander-v2
- ✅ BipedalWalker-v3
- ✅ Ant-v4
- ✅ ALE/Breakout-v5

### PPO (모든 환경)
- ✅ CartPole-v1
- ✅ LunarLander-v2
- ✅ BipedalWalker-v3
- ✅ Ant-v4
- ✅ ALE/Breakout-v5

### SAC (Continuous Action만)
- ❌ CartPole-v1 (이산 제어 불가)
- ❌ LunarLander-v2 (이산 제어 불가)
- ✅ BipedalWalker-v3
- ✅ Ant-v4
- ❌ ALE/Breakout-v5 (이산 제어 불가)

## 학습 결과

학습 결과는 자동으로 저장됩니다:

```
models/              # 학습된 모델
└── {algo}/
    └── {env}/
        └── final_model.zip

logs/                # TensorBoard 로그
└── {algo}/
    └── {env}/
        └── {timestamp}/
```

## TensorBoard로 학습 모니터링

```bash
# 전체 결과 보기
tensorboard --logdir logs/

# 특정 알고리즘만 보기
tensorboard --logdir logs/ppo/

# 특정 환경만 보기
tensorboard --logdir logs/ppo/CartPole-v1/
```

브라우저에서 http://localhost:6006 열기

## 예상 학습 시간

| 알고리즘 | 환경 | Timesteps | 예상 시간 (CPU) |
|---------|------|-----------|----------------|
| DQN | CartPole-v1 | 50K | ~5분 |
| DQN | LunarLander-v2 | 300K | ~30분 |
| DQN | ALE/Breakout-v5 | 1M | ~2시간 |
| A2C | CartPole-v1 | 50K | ~3분 |
| PPO | CartPole-v1 | 50K | ~5분 |
| PPO | LunarLander-v2 | 300K | ~30분 |
| PPO | BipedalWalker-v3 | 2M | ~3시간 |
| PPO | Ant-v4 | 2M | ~4시간 |
| SAC | BipedalWalker-v3 | 300K | ~1시간 |
| SAC | Ant-v4 | 1M | ~2시간 |

GPU 사용 시 약 2~3배 빠를 수 있습니다.

## 권장 학습 순서

1. **DQN + CartPole** → Value-based 기초
2. **A2C + CartPole** → Actor-Critic 기초
3. **PPO + LunarLander** → 안정적 정책 학습
4. **PPO vs SAC + BipedalWalker** → 샘플 효율성 비교
5. **SAC + Ant** → 고차원 제어

## 하이퍼파라미터 커스터마이징

### 방법 1: JSON 파일 직접 편집

`configs/ppo.json` 예시:
```json
{
  "algorithm": "ppo",
  "common": {
    "gamma": 0.99,
    "gae_lambda": 0.95
  },
  "environments": {
    "CartPole-v1": {
      "total_timesteps": 50000,
      "learning_rate": 0.0003,
      "n_steps": 2048
    }
  }
}
```

### 방법 2: 명령줄에서 오버라이드

```bash
# learning_rate만 변경
python train.py --env CartPole-v1 --algo ppo --learning-rate 0.001

# 여러 파라미터 변경
python train.py --env CartPole-v1 --algo ppo \
  --learning-rate 0.001 \
  --n-steps 1024 \
  --batch-size 32
```

## 문제 해결

### 모듈을 찾을 수 없음
```bash
# 프로젝트 루트에서 실행하세요
cd /path/to/ez-drl
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1
```

### JSON 설정 사용 안 함
```bash
# --no-json 옵션 사용 (코드 기본값 사용)
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1 --no-json
```

### 학습이 너무 느림
```bash
# 병렬 환경 사용 (A2C, PPO만 지원)
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1 --n-envs 8
```

### 성능이 낮음
1. TensorBoard로 학습 곡선 확인
2. JSON 설정 파일에서 하이퍼파라미터 조절
3. 더 긴 학습 시간 시도
4. 환경과 알고리즘 호환성 확인 (위 표 참고)

## 추가 리소스

### 알고리즘 가이드
- [DQN 가이드](../docs/dqn.md)
- [A2C 가이드](../docs/a2c.md)
- [PPO 가이드](../docs/ppo.md)
- [SAC 가이드](../docs/sac.md)

### 비교 및 설정
- [알고리즘 비교](../docs/comparison.md)
- [JSON 설정 가이드](../configs/README.md)

### 외부 자료
- [Stable-Baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
