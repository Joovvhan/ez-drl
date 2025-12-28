# A2C (Advantage Actor-Critic)

## 개요

A2C는 **동기식(synchronous) Actor-Critic** 알고리즘으로, REINFORCE의 높은 분산 문제를 해결하면서도 구현이 간단한 방법입니다.

## 왜 A2C가 등장했는가?

### REINFORCE의 문제점
- ✗ 매우 높은 분산 (Variance)
- ✗ 학습 불안정
- ✗ Monte Carlo 방법 (에피소드 끝까지 기다려야 함)

### A2C의 해결책
- ✓ **Baseline (Value function)** 사용 → 분산 감소
- ✓ **TD learning** → 매 스텝마다 학습 가능
- ✓ **Advantage function** → 더 정확한 gradient 추정
- ✓ **동기식 업데이트** → 안정성과 재현성

## 핵심 아이디어

### 1. Actor-Critic 구조

**Actor (π)**: 정책 네트워크
- 행동을 선택합니다
- Policy gradient로 업데이트

**Critic (V)**: 가치 네트워크
- 상태의 가치를 평가합니다
- TD error로 업데이트

### 2. Advantage Function

REINFORCE는 전체 return R을 사용:
```python
loss = -log π(a|s) * R
```

A2C는 Advantage를 사용:
```python
A(s,a) = Q(s,a) - V(s)
       ≈ r + γV(s') - V(s)  # TD estimate

loss = -log π(a|s) * A(s,a)
```

**의미**: "이 행동이 평균보다 얼마나 좋은가?"

### 3. Temporal Difference (TD) Learning

**REINFORCE (Monte Carlo)**:
- 에피소드가 끝나야 return을 계산
- R = r₁ + γr₂ + γ²r₃ + ...
- 높은 분산, 낮은 bias

**A2C (TD)**:
- 매 스텝마다 업데이트 가능
- TD target = r + γV(s')
- 낮은 분산, 약간의 bias

### 4. Synchronous Updates

**A3C (Asynchronous)**:
- 여러 worker가 비동기적으로 업데이트
- 빠르지만 불안정할 수 있음

**A2C (Synchronous)**:
- 모든 worker의 gradient를 모아서 한 번에 업데이트
- 안정적이고 재현 가능
- GPU 활용 효율적

## 알고리즘 흐름

```
for each step:
    1. 현재 정책으로 n_steps만큼 경험 수집
    2. Advantage 계산:
       A(s,a) = r + γV(s') - V(s)
    3. Actor 업데이트:
       θ ← θ + α ∇_θ log π(a|s) * A(s,a)
    4. Critic 업데이트:
       φ ← φ - β ∇_φ (V(s) - target)²
```

## 장점

### 1. 낮은 분산 ✅
- Baseline (V)로 분산 감소
- REINFORCE보다 훨씬 안정적

### 2. 온라인 학습 ✅
- 에피소드가 끝나지 않아도 학습
- 무한 에피소드 환경에도 적용 가능

### 3. Discrete + Continuous 지원 ✅
- 모든 action space 지원
- 범용성 높음

### 4. 구현 단순 ✅
- PPO보다 간단
- 이해하기 쉬움

## 한계

### 1. PPO보다 불안정할 수 있음 ⚠️
- Trust region이 없음
- 큰 policy update로 인한 성능 저하 가능

### 2. On-policy ⚠️
- 현재 정책의 데이터만 사용
- SAC보다 샘플 효율성 낮음

### 3. 작은 batch size ⚠️
- n_steps가 작아서 (보통 5) 학습이 noisy할 수 있음
- PPO는 더 큰 batch 사용 (2048)

## A2C vs PPO

| 특징 | A2C | PPO |
|-----|-----|-----|
| **안정성** | 중간 | 높음 |
| **구현 난이도** | 쉬움 | 쉬움 |
| **학습 속도** | 빠름 | 중간 |
| **Batch size** | 작음 (5) | 큼 (2048) |
| **Trust region** | 없음 | Clipping |
| **샘플 효율성** | 낮음 | 중간 |

**언제 A2C를 쓸까?**
- 빠른 프로토타이핑
- 간단한 환경
- 학습 속도 > 안정성

**언제 PPO를 쓸까?**
- 안정성 중요
- 복잡한 환경
- Production 환경

## 적용 환경

| 환경 | 적합도 | 이유 |
|-----|-------|------|
| CartPole | ✅ 적합 | 간단한 환경, 빠른 학습 |
| LunarLander | ✅ 적합 | 중간 복잡도 |
| BipedalWalker | ⚠️ 중간 | PPO가 더 안정적 |
| Ant | ⚠️ 중간 | PPO나 SAC 권장 |
| Breakout | ✅ 적합 | Discrete, 빠른 학습 |

## 하이퍼파라미터 설명

```python
model = A2C(
    policy="MlpPolicy",
    learning_rate=7e-4,       # 학습률 (PPO보다 높음)
    n_steps=5,                # 한 번에 수집할 스텝 수
    gamma=0.99,               # Discount factor
    gae_lambda=1.0,           # GAE λ (1.0 = TD(1))
    ent_coef=0.0,             # Entropy bonus
    vf_coef=0.5,              # Value loss 계수
    max_grad_norm=0.5,        # Gradient clipping
    use_rms_prop=True,        # RMSprop 사용 (기본값)
)
```

### 주요 파라미터

#### n_steps
- **역할**: 업데이트 전 수집할 스텝 수
- **기본값**: 5
- **조절**:
  - 크면: 안정적, 느림
  - 작으면: 빠름, noisy

#### learning_rate
- **기본값**: 7e-4 (PPO의 3e-4보다 높음)
- **이유**: 작은 batch size 때문

#### gae_lambda
- **기본값**: 1.0 (TD(1), Monte Carlo와 유사)
- **조절**:
  - 낮추면: TD(0)에 가까워짐, 분산 감소

## 실전 팁

### 1. 병렬 환경 활용
```python
# 병렬 환경으로 샘플 수집 속도 향상
python train.py --env CartPole-v1 --algo a2c --n-envs 4
```

### 2. 간단한 환경에서 시작
A2C는 간단한 환경에서 빠르게 학습:
```python
# CartPole: 매우 빠름
python train.py --env CartPole-v1 --algo a2c --timesteps 50000
```

### 3. 복잡한 환경에는 PPO 고려
BipedalWalker, Ant 같은 환경:
- A2C로 빠른 프로토타입
- PPO로 최종 성능 향상

## A2C의 역사

```
REINFORCE (1992)
    ↓
+ Baseline (Value function)
    ↓
Actor-Critic (1999)
    ↓
A3C (Asynchronous, 2016)
    ↓
A2C (Synchronous, 2016)
    ↓
현재: PPO에 밀려 덜 사용되지만
      여전히 빠른 프로토타이핑에 유용
```

## 학습 곡선 특징

### 정상적인 학습
```
Reward
  │    ╱──
  │   ╱
  │  ╱
  │ ╱
  └────── Steps
```
- REINFORCE보다 안정적
- PPO보다 약간 noisy
- 빠른 초기 학습

### 문제 신호
- **진동**: learning_rate 감소
- **정체**: n_steps 증가
- **발산**: learning_rate 감소 또는 PPO 사용

## 코드 예제

### 기본 사용
```python
from config import TrainingConfig
from utils import train_model

config = TrainingConfig(
    env_name="CartPole-v1",
    algorithm="a2c",
    n_envs=4,  # 병렬 환경
)

model, _, _ = train_model(config)
```

### 병렬 환경
```bash
# 4개 병렬 환경으로 빠른 학습
python train.py --env CartPole-v1 --algo a2c --n-envs 4
```

## 비교 실험

### A2C vs PPO (CartPole)

**실험 설정**:
```bash
# A2C
python train.py --env CartPole-v1 --algo a2c --n-envs 4

# PPO
python train.py --env CartPole-v1 --algo ppo
```

**예상 결과**:
- A2C: 더 빠름 (wall-clock time)
- PPO: 더 안정적 (낮은 분산)
- 최종 성능: 유사

### TensorBoard 비교
```bash
tensorboard --logdir results/
```
- A2C: 더 noisy한 학습 곡선
- PPO: 더 smooth한 학습 곡선

## 언제 A2C를 사용할까?

### ✅ A2C 선택
- 빠른 실험 / 프로토타이핑
- 간단한 환경 (CartPole, LunarLander)
- 학습 속도가 중요
- 병렬 환경 활용 가능

### ❌ PPO 선택
- 안정성 중요
- 복잡한 환경 (BipedalWalker, Ant)
- Production 환경
- 재현성 중요

## 핵심 메시지

> **A2C는 REINFORCE와 PPO 사이의 다리입니다.**
>
> - REINFORCE의 높은 분산을 해결 (Baseline)
> - PPO의 안정성은 없지만 더 빠름
> - Actor-Critic의 기초를 이해하기 좋음

**학습 순서 권장**:
1. REINFORCE → 정책 gradient의 직관
2. **A2C** → Actor-Critic의 기초
3. PPO → Trust region과 안정성

## 참고 자료

- A3C 논문: [Asynchronous Methods for Deep RL (2016)](https://arxiv.org/abs/1602.01783)
- OpenAI Baselines: [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c)
- Stable-Baselines3: [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
