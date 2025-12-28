# Easy Deep Reinforcement Learning

> **DQN → PPO → SAC**의 흐름은 "성능 경쟁"이 아니라
> **"강화학습이 어떤 문제를 해결해 왔는가"**를 보여줍니다.

## 📚 저장소 목적

본 저장소는 **Stable-Baselines3(SB3)**를 활용하여 대표적인 Deep Reinforcement Learning 알고리즘을 다양한 Gymnasium 환경에 적용하고, **각 알고리즘의 한계와 진화를 실험적으로 이해**하는 것을 목표로 합니다.

- ✅ 개념 이해 + 재현 가능한 실험
- ❌ 최고 성능(SOTA) 달성

## 🎯 구현 알고리즘

모든 알고리즘은 **Stable-Baselines3** 기반으로 구현됩니다.

| 알고리즘 | 계열 | 핵심 메시지 |
|---------|------|-----------|
| **DQN** | Value-based | 값 기반 방법의 장점과 한계 이해<br>→ "왜 정책 기반 방법이 등장했는가?" |
| **A2C** | Actor-Critic<br>(Synchronous) | Actor-Critic의 기초 이해<br>→ "왜 Baseline이 중요한가?" |
| **PPO** | Actor-Critic<br>(On-policy) | 안정적인 정책 학습의 표준<br>→ "왜 PPO가 기본값(default)이 되었는가?" |
| **SAC** | Actor-Critic<br>(Off-policy) | 탐색과 샘플 효율성의 진화<br>→ "왜 PPO 이후에도 SAC가 필요한가?" |

### 알고리즘별 핵심 특징

#### 1. DQN (Deep Q-Network)
- **장점**: Discrete action 환경에서 안정적
- **한계**:
  - Continuous action 불가
  - 고차원 상태/불안정한 환경에서 성능 저하

#### 2. A2C (Advantage Actor-Critic)
- **장점**:
  - Baseline으로 분산 감소
  - TD learning으로 온라인 학습
  - 병렬 환경 지원
- **특징**: 동기식 업데이트, 빠른 프로토타이핑에 적합

#### 3. PPO (Proximal Policy Optimization)
- **장점**:
  - Clipping 기반 안정성
  - Discrete / Continuous 모두 지원
  - 업계 표준(default choice)
- **특징**: On-policy (현재 정책으로 수집한 데이터만 사용)

#### 4. SAC (Soft Actor-Critic)
- **장점**:
  - Entropy maximization으로 탐색 강화
  - Replay Buffer로 샘플 효율성 극대화
  - Continuous control에 매우 강력
- **특징**: Off-policy (과거 데이터 재사용 가능)

## 🎮 적용 환경

다양한 Action Space와 State Representation을 다루기 위해 5개의 대표 환경을 선정했습니다.

| 환경 | Action | State | 목적 |
|-----|--------|-------|------|
| **CartPole** | Discrete | Vector (Low-dim) | 강화학습 입문, 알고리즘 구조 이해 |
| **LunarLander** | Discrete | Vector | 복잡한 동역학, DQN의 한계 관찰 |
| **BipedalWalker** | Continuous | Vector | 연속 제어 입문, PPO vs SAC 비교 |
| **Ant** (MuJoCo) | Continuous (고차원) | Vector (High-dim) | 고차원 연속 제어, 안정성/샘플 효율 |
| **Breakout** (Atari) | Discrete | Image (pixel) | Representation Learning, Image-based RL |

## 🗂️ 프로젝트 구조

```
ez-drl/
├── train.py             # 🔥 통합 학습 진입점
├── test_all.py          # 전체 테스트 스크립트
├── config.py            # 공통 설정
├── utils.py             # 유틸리티 함수
│
├── algorithms/          # 알고리즘별 구현
│   ├── dqn/            # DQN 학습 스크립트
│   ├── a2c/            # A2C 학습 스크립트
│   ├── ppo/            # PPO 학습 스크립트
│   ├── sac/            # SAC 학습 스크립트
│   └── reinforce/      # REINFORCE 참고 구현
│
├── docs/               # 알고리즘 설명 문서
│   ├── dqn.md
│   ├── a2c.md
│   ├── ppo.md
│   ├── sac.md
│   └── comparison.md
│
├── examples/           # 간단한 예제
├── environments/       # 환경 설정
├── results/           # 학습 결과 (TensorBoard 로그)
│
├── requirements.txt   # pip 의존성
├── environment.yml    # conda 환경
├── QUICK_START.md     # 빠른 시작 가이드
└── README.md
```

## 🚀 빠른 시작

### 1. 환경 설정

**Conda 사용 (권장)**
```bash
# Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate ez-drl
```

**또는 pip 사용**
```bash
pip install -r requirements.txt
```

### 2. 예제 실행

**통합 학습 스크립트 (train.py)**:
```bash
# 대화형 모드 - 초보자에게 권장
python train.py

# 명령줄 모드 - 빠른 실행
python train.py --env CartPole-v1 --algo ppo
python train.py --env LunarLander-v2 --algo dqn --timesteps 300000
python train.py --env BipedalWalker-v3 --algo sac

# 병렬 환경 사용 (A2C, PPO)
python train.py --env CartPole-v1 --algo ppo --n-envs 4

# 사용 가능한 환경 목록
python train.py --list
```

**알고리즘별 스크립트 (run_algorithm.py)**:
```bash
# JSON 설정 자동 로드
python algorithms/run_algorithm.py --algo ppo --env CartPole-v1
python algorithms/run_algorithm.py --algo sac --env BipedalWalker-v3

# 학습 후 테스트 실행
python algorithms/run_algorithm.py --algo dqn --env LunarLander-v2 --test
```

**REINFORCE 참고 구현** (개념 학습용):
```bash
python examples/train_reinforce_cartpole.py
```

**전체 테스트**:
```bash
# 모든 알고리즘 x 환경 조합 테스트 (각 3분)
python test_all.py

# 빠른 테스트 (각 1분)
python test_all.py --quick

# Atari 제외 (시간 절약)
python test_all.py --exclude-atari
```

## 📊 알고리즘 × 환경 매핑

각 알고리즘이 어떤 환경에서 잘 작동하는지 확인할 수 있습니다.

|  | CartPole | LunarLander | BipedalWalker | Ant | Breakout |
|---|:---:|:---:|:---:|:---:|:---:|
| **DQN** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **A2C** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **PPO** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **SAC** | ❌ | ❌ | ✅ | ✅ | ❌ |

- ✅: 적합 / 권장
- ❌: 부적합 (알고리즘 제약)

**왜 이런 차이가 있을까요?**
- DQN: Discrete action만 지원 → Continuous 환경 불가
- A2C, PPO: 모든 환경 지원, 범용성 높음
- SAC: Continuous action 전용 → Discrete 환경 불가

## 📖 학습 가이드

### 추천 학습 순서

1. **CartPole (DQN)** → Value-based 기초 이해
2. **CartPole (A2C)** → Actor-Critic의 기초 체감
3. **LunarLander (PPO)** → Trust region과 안정성 이해
4. **BipedalWalker (PPO vs SAC)** → Continuous control에서 두 방법 비교
5. **Ant (SAC)** → 고차원 제어와 샘플 효율성
6. **Breakout (PPO)** → Image-based RL의 특수성

## 🛠️ 요구사항

- Python 3.10
- PyTorch
- Stable-Baselines3
- Gymnasium >= 0.29.0
- NumPy < 2.0.0
- (Optional) MuJoCo (Ant 환경용)

## 📝 각 알고리즘 문서

자세한 내용은 `docs/` 폴더를 참고하세요:

- [DQN 가이드](docs/dqn.md)
- [A2C 가이드](docs/a2c.md)
- [PPO 가이드](docs/ppo.md)
- [SAC 가이드](docs/sac.md)
- [알고리즘 비교](docs/comparison.md)

## ❌ 본 저장소에서 다루지 않는 것

- SOTA 성능 튜닝
- 대규모 분산 학습
- Offline RL / Model-based RL
- 논문 재현 목적의 복잡한 세팅

## 🚀 주요 기능

### 1. 통합 학습 스크립트
```bash
python train.py --env CartPole-v1 --algo ppo
```
- 모든 알고리즘과 환경을 하나의 인터페이스로 학습
- 대화형 모드 지원
- 자동 TensorBoard 로깅
- 체크포인트 및 평가 콜백

### 2. TensorBoard 통합
```bash
tensorboard --logdir results/
```
- 환경/알고리즘별로 체계적으로 정리된 로그
- 같은 환경의 다른 알고리즘 비교 용이
- 실시간 학습 모니터링

### 3. 유연한 설정
```bash
# 병렬 환경
python train.py --env CartPole-v1 --algo ppo --n-envs 4

# 네트워크 구조 커스터마이징
python train.py --env CartPole-v1 --algo ppo --net-arch 256 256

# 시드 고정 (재현성)
python train.py --env CartPole-v1 --algo ppo --seed 42
```

### 4. 전체 테스트
```bash
python test_all.py
```
- 모든 알고리즘 × 환경 조합 자동 테스트
- 각 조합을 3분간 학습하여 코드 검증
- CI/CD 파이프라인에 통합 가능

## 💡 빠른 시작 예시

```bash
# 1. 환경 설정
conda env create -f environment.yml
conda activate ez-drl

# 2. 첫 실험 (DQN으로 CartPole 학습)
python train.py --env CartPole-v1 --algo dqn

# 3. TensorBoard로 결과 확인
tensorboard --logdir results/

# 4. 다른 알고리즘 비교
python train.py --env CartPole-v1 --algo a2c
python train.py --env CartPole-v1 --algo ppo

# 5. 연속 제어 환경 시도
python train.py --env BipedalWalker-v3 --algo sac
```

## 🎓 핵심 메시지

> **DQN → A2C → PPO → SAC**로 이어지는 흐름은 "성능 경쟁"이 아니라
> **"강화학습이 어떤 문제를 해결해 왔는가"**를 보여줍니다.

**이 저장소의 특징**:
- ✨ **단순함**: 하나의 명령어로 모든 실험 실행
- 📝 **체계적**: JSON으로 하이퍼파라미터 중앙 관리
- 🔄 **재현 가능**: 설정 파일 공유로 동일한 결과 재현
- 📊 **비교 용이**: TensorBoard로 알고리즘 성능 비교

**학습 목표**:
- CartPole → Breakout → Ant 환경 변화 속에서
- 각 알고리즘의 한계와 진화를 직접 체험

## 📄 License

MIT License

## 🤝 Contributing

이슈와 Pull Request를 환영합니다!
