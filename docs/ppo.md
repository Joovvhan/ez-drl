# PPO (Proximal Policy Optimization)

## 개요

PPO는 현재 **가장 널리 사용되는** 강화학습 알고리즘입니다. **Actor-Critic** 방법으로, 안정성과 성능의 균형이 뛰어납니다.

## 왜 PPO가 등장했는가?

### REINFORCE의 문제점
- ✗ 높은 분산 (Variance)
- ✗ 학습 불안정
- ✗ 샘플 비효율성

### PPO의 해결책
- ✓ **Baseline (Value function)** 사용 → 분산 감소
- ✓ **Clipped objective** 사용 → 안정성 향상
- ✓ **GAE (Generalized Advantage Estimation)** → Bias-variance 균형

## 핵심 아이디어

### 1. Actor-Critic 구조
- **Actor (π)**: 정책 네트워크, 행동 선택
- **Critic (V)**: 가치 네트워크, 상태 평가

```
Advantage = Q(s,a) - V(s)
          ≈ r + γV(s') - V(s)  (TD 방식)
```

### 2. Trust Region / Clipping
**문제**: 정책을 너무 크게 바꾸면 학습이 불안정해집니다.

**해결책**: 정책 변화를 제한합니다.

#### Clipped Objective
```python
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
L = min(ratio * A, clipped_ratio * A)
```

- **ratio**: 새 정책과 옛 정책의 확률 비율
- **clip**: 비율을 [1-ε, 1+ε] 범위로 제한 (보통 ε=0.2)
- **효과**: 정책이 급격히 바뀌는 것을 방지

### 3. Generalized Advantage Estimation (GAE)
```
A^GAE = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

- **λ**: Bias-variance tradeoff 조절
  - λ=0: Low variance, high bias (TD)
  - λ=1: High variance, low bias (Monte Carlo)
  - 보통 λ=0.95 사용

## 알고리즘 흐름

1. **데이터 수집**: 현재 정책으로 n_steps만큼 경험 수집
2. **Advantage 계산**: GAE로 advantage 추정
3. **미니배치 학습**:
   - 수집한 데이터를 여러 epoch 동안 재사용
   - Clipped objective로 actor 업데이트
   - MSE loss로 critic 업데이트
4. 반복

## 장점

### 1. 안정적인 학습 ✅
- Clipping으로 정책 변화 제한
- Trust region 보장
- **재현성 좋음**

### 2. Discrete + Continuous 모두 지원 ✅
- Discrete: Categorical distribution
- Continuous: Gaussian distribution
- **범용성 높음**

### 3. 구현과 튜닝이 비교적 쉬움 ✅
- 하이퍼파라미터가 환경에 robust
- 기본 설정으로도 좋은 성능
- **업계 표준 (default choice)**

### 4. 적당한 샘플 효율성 ✅
- On-policy지만 미니배치 재사용
- REINFORCE보다 훨씬 효율적

## 한계

### 1. On-policy ⚠️
- 현재 정책으로 수집한 데이터만 사용
- SAC 같은 off-policy보다 샘플 효율성 낮음
- **Replay buffer 사용 불가**

### 2. 고차원 연속 제어에서 SAC보다 느릴 수 있음 ⚠️
- Ant 같은 환경에서 SAC가 더 빠를 수 있음
- 하지만 안정성은 PPO가 우수

## 적용 환경

| 환경 | 적합도 | 이유 |
|-----|-------|------|
| CartPole | ✅ 매우 적합 | Discrete, 안정적 학습 |
| LunarLander | ✅ 매우 적합 | Discrete, 복잡한 동역학 |
| BipedalWalker | ✅ 적합 | Continuous, 안정성 중요 |
| Ant | ✅ 적합 | Continuous, 고차원 |
| Breakout | ✅ 적합 | Discrete, 이미지 입력 |

**PPO는 모든 환경에서 사용 가능한 baseline입니다!**

## 하이퍼파라미터 설명

```python
model = PPO(
    policy="MlpPolicy",       # MLP 또는 CNN
    learning_rate=3e-4,       # 학습률
    n_steps=2048,             # 한 번에 수집할 스텝 수
    batch_size=64,            # 미니배치 크기
    n_epochs=10,              # 데이터 재사용 횟수
    gamma=0.99,               # Discount factor
    gae_lambda=0.95,          # GAE λ 파라미터
    clip_range=0.2,           # Clipping 범위 ε
    ent_coef=0.0,             # Entropy bonus 계수
    vf_coef=0.5,              # Value loss 계수
    max_grad_norm=0.5,        # Gradient clipping
)
```

### 주요 파라미터 조절 가이드

#### n_steps
- **역할**: 한 번에 수집할 경험의 양
- **기본값**: 2048
- **조절**:
  - 큰 값: 더 안정적, 느림
  - 작은 값: 빠름, 불안정할 수 있음

#### n_epochs
- **역할**: 데이터를 몇 번 재사용할지
- **기본값**: 10
- **조절**:
  - 너무 크면: Overfitting, 정책 왜곡
  - 너무 작으면: 데이터 낭비

#### clip_range (ε)
- **역할**: 정책 변화 제한
- **기본값**: 0.2
- **조절**:
  - 크면: 빠른 학습, 불안정
  - 작으면: 안정적, 느림

#### ent_coef
- **역할**: 탐색 장려 (Entropy bonus)
- **기본값**: 0.0
- **조절**:
  - 탐색이 부족하면 증가 (0.01~0.1)

## PPO vs DQN vs SAC

| 특징 | DQN | PPO | SAC |
|-----|-----|-----|-----|
| **Action Space** | Discrete만 | 둘 다 | Continuous만 |
| **On/Off-policy** | Off-policy | On-policy | Off-policy |
| **안정성** | 중간 | 높음 | 중간 |
| **샘플 효율성** | 높음 | 중간 | 매우 높음 |
| **구현 난이도** | 쉬움 | 쉬움 | 중간 |
| **적용 범위** | 제한적 | 매우 넓음 | Continuous만 |

## 학습 곡선 해석

### 정상적인 학습
```
Episode Reward
  │     ┌─────
  │    ╱
  │   ╱
  │  ╱
  │ ╱
  └─────────── Steps
```
- 안정적인 상승 곡선
- 변동성 낮음
- 수렴 후 안정화

### 문제 신호
- **진동**: clip_range 조절 또는 learning_rate 감소
- **정체**: ent_coef 증가 (탐색 부족)
- **급락**: n_epochs 감소 (overfitting)

## PPO의 변형

### 1. PPO-Clip (일반적)
- Clipped objective 사용
- 구현 간단
- **Stable-Baselines3에서 사용**

### 2. PPO-Penalty (원논문)
- KL divergence penalty 사용
- Adaptive penalty 조절
- 구현 복잡

## 실전 팁

### 1. 처음에는 기본 설정 사용
```python
model = PPO("MlpPolicy", env)
```
대부분의 환경에서 기본 설정이 잘 작동합니다.

### 2. TensorBoard로 모니터링
```bash
tensorboard --logdir results/ppo/
```
- `rollout/ep_rew_mean`: 평균 에피소드 보상
- `train/clip_fraction`: Clipping 비율 (0.1~0.3이 적절)
- `train/policy_loss`: Actor loss
- `train/value_loss`: Critic loss

### 3. 환경별 조절
- **간단한 환경 (CartPole)**: timesteps 줄이기
- **복잡한 환경 (Ant)**: timesteps, n_steps 늘리기
- **탐색 필요 (Sparse reward)**: ent_coef 증가

## 다음 단계

PPO를 이해했다면:
1. **SAC**와 샘플 효율성 비교 (BipedalWalker, Ant)
2. **DQN**과 Discrete 환경 성능 비교
3. 다양한 환경에서 PPO baseline 구축

## 참고 자료

- 원논문: [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)
- OpenAI 블로그: [Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- Stable-Baselines3 문서: [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
