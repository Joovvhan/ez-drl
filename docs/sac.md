# SAC (Soft Actor-Critic)

## 개요

SAC는 **Continuous control**에 특화된 최신 강화학습 알고리즘입니다. **Off-policy Actor-Critic** 방법으로, 샘플 효율성과 탐색의 균형이 뛰어납니다.

## 왜 SAC가 등장했는가?

### PPO의 한계
- ✗ On-policy → 샘플 효율성 낮음
- ✗ 고차원 연속 제어에서 느릴 수 있음
- ✗ 탐색 전략이 단순 (Gaussian noise)

### SAC의 해결책
- ✓ **Off-policy** + Replay buffer → 샘플 효율성 극대화
- ✓ **Entropy maximization** → 적극적 탐색
- ✓ **Automatic temperature tuning** → 하이퍼파라미터 자동 조절

## 핵심 아이디어

### 1. Maximum Entropy RL

**일반적인 RL 목표**:
```
최대화: E[Σ γ^t r_t]
```

**SAC의 목표**:
```
최대화: E[Σ γ^t (r_t + α H(π(·|s_t)))]
```

- **H(π)**: 정책의 엔트로피 (불확실성)
- **α**: Temperature (탐색 정도 조절)

**의미**: "높은 보상"을 받으면서 동시에 "다양한 행동"을 시도합니다.

#### 왜 엔트로피가 중요한가?

**높은 엔트로피 (다양한 행동)**:
- 탐색 강화
- Local optimum 회피
- Robust한 정책 학습

**예시**: BipedalWalker
- 낮은 엔트로피: 한 가지 걸음걸이만 학습 → 장애물에 약함
- 높은 엔트로피: 다양한 걸음걸이 시도 → 환경 변화에 강함

### 2. Off-Policy Learning

- **Replay buffer** 사용
- 과거 데이터 재사용
- **샘플 효율성 극대화**

PPO vs SAC:
- PPO: 2M steps 필요
- SAC: 300K steps로 유사한 성능 (약 6배 효율적!)

### 3. Automatic Temperature Tuning

**문제**: α를 수동으로 조절하기 어려움

**해결**: α를 자동으로 학습
```python
ent_coef="auto"  # α를 자동 조절
```

- 초반: α 높음 → 탐색 중심
- 후반: α 낮음 → 착취 중심

## 알고리즘 구조

### 네트워크
1. **Actor (π)**: 정책 네트워크
   - Gaussian policy 사용
   - Mean과 std 출력

2. **Critic (Q)**: 2개의 Q-network
   - Q1, Q2 (Twin Q-networks)
   - Overestimation 방지

3. **Target networks**: Critic의 타겟 네트워크
   - Soft update로 안정성 향상

### 학습 흐름

1. **데이터 수집**: 현재 정책으로 환경과 상호작용
2. **Replay buffer 저장**: (s, a, r, s') 저장
3. **샘플링**: Replay buffer에서 미니배치 추출
4. **Q-function 업데이트**: Bellman equation
5. **Policy 업데이트**: Policy gradient
6. **Temperature 업데이트**: Entropy constraint
7. **Target networks 업데이트**: Soft update
8. 반복

## 장점

### 1. 뛰어난 샘플 효율성 ✅
- Off-policy + replay buffer
- **PPO 대비 5~10배 빠를 수 있음**
- 고가의 시뮬레이션 환경에 유리

### 2. 안정적인 학습 ✅
- Twin Q-networks → Overestimation 방지
- Soft target update → 안정성
- Automatic tuning → 튜닝 부담 감소

### 3. 강력한 탐색 ✅
- Entropy maximization
- Local optimum 회피
- Robust한 정책

### 4. Continuous control 최적화 ✅
- Gaussian policy
- Squashed Gaussian (tanh) → Bounded action
- **로보틱스, 제어 문제에 적합**

## 한계

### 1. Continuous action만 지원 ❌
- Discrete action 불가
- CartPole, Breakout 등 사용 불가
- **해결책**: Discrete SAC 변형 (복잡함)

### 2. 구현 복잡도 ⚠️
- 3개 네트워크 + 타겟 네트워크
- Temperature tuning 추가
- PPO보다 복잡

### 3. 메모리 사용량 ⚠️
- Replay buffer 필요
- 대규모 buffer는 RAM 소모
- **하지만 샘플 효율성으로 상쇄**

## 적용 환경

| 환경 | 적합도 | 이유 |
|-----|-------|------|
| CartPole | ❌ 불가 | Discrete action |
| LunarLander | ❌ 불가 | Discrete action |
| BipedalWalker | ✅ 매우 적합 | Continuous, 복잡한 동역학 |
| Ant | ✅ 매우 적합 | 고차원 continuous control |
| Breakout | ❌ 불가 | Discrete action |

**SAC는 Continuous control의 최강자입니다!**

## 하이퍼파라미터 설명

```python
model = SAC(
    policy="MlpPolicy",
    learning_rate=3e-4,       # 학습률
    buffer_size=1000000,      # Replay buffer 크기
    learning_starts=10000,    # 학습 시작 전 데이터 수집
    batch_size=256,           # 미니배치 크기
    tau=0.005,                # Target network soft update
    gamma=0.99,               # Discount factor
    train_freq=1,             # 매 스텝마다 학습
    gradient_steps=1,         # 업데이트 횟수
    ent_coef="auto",          # Temperature 자동 조절
)
```

### 주요 파라미터 조절 가이드

#### buffer_size
- **역할**: 과거 데이터 저장량
- **기본값**: 1M
- **조절**:
  - 크면: 더 다양한 데이터, 메모리 소모
  - 작으면: 최신 데이터 중심, 편향 가능

#### learning_starts
- **역할**: 학습 시작 전 랜덤 탐색
- **기본값**: 10K
- **조절**:
  - 복잡한 환경: 늘리기
  - 간단한 환경: 줄이기

#### tau
- **역할**: Target network soft update 속도
- **기본값**: 0.005
- **조절**:
  - 크면: 빠른 update, 불안정
  - 작으면: 안정적, 느림

#### ent_coef
- **역할**: Entropy weight (탐색 정도)
- **기본값**: "auto" (권장!)
- **수동 조절**:
  - 탐색 부족: 증가 (0.1~1.0)
  - 너무 랜덤: 감소 (0.01~0.05)

## SAC vs PPO 비교

### BipedalWalker 예시

| 지표 | PPO | SAC |
|-----|-----|-----|
| **학습 timesteps** | 2M | 300K |
| **벽시계 시간** | 중간 | 빠름 |
| **최종 성능** | 280~300 | 300+ |
| **안정성** | 매우 안정 | 안정 |
| **탐색** | 제한적 | 적극적 |

### Ant 예시

| 지표 | PPO | SAC |
|-----|-----|-----|
| **학습 timesteps** | 2M | 1M |
| **샘플 효율성** | 중간 | 높음 |
| **최종 성능** | 유사 | 유사 |
| **안정성** | 높음 | 중간 |

**결론**: 고차원 연속 제어에서 SAC가 더 빠르게 학습!

## 학습 곡선 해석

### 정상적인 학습
```
Episode Reward
  │        ┌────
  │      ╱╱
  │    ╱╱
  │  ╱╱
  │╱╱
  └──────────── Steps
```
- 초반: 랜덤 탐색 (낮은 보상)
- 중반: 급격한 상승
- 후반: 높은 성능 유지

### TensorBoard 모니터링
- `rollout/ep_rew_mean`: 평균 보상
- `train/ent_coef`: Temperature α (자동 조절 확인)
- `train/actor_loss`: Policy loss
- `train/critic_loss`: Q-function loss

## Twin Q-Networks

### 왜 2개의 Q-network?

**DQN의 문제**: Q 값 과대평가 (Overestimation)
```
Q_target = r + γ max Q(s', a')
```

**SAC의 해결책**: 2개 Q 중 작은 값 사용
```
Q_target = r + γ min(Q1(s', a'), Q2(s', a'))
```

**효과**:
- Overestimation 방지
- 더 보수적인 Q 추정
- 안정적인 학습

## 실전 팁

### 1. 기본 설정으로 시작
```python
model = SAC("MlpPolicy", env, ent_coef="auto")
```

### 2. 충분한 탐색 시간 부여
```python
learning_starts=10000  # 최소 10K
```
초반 랜덤 탐색이 중요합니다.

### 3. Replay buffer 크기 조절
- **메모리 충분**: 1M
- **메모리 제한**: 100K~300K

### 4. 환경별 학습 시간
- **BipedalWalker**: 300K steps
- **Ant**: 1M steps
- **복잡한 로봇**: 2M+ steps

## SAC의 변형

### 1. SAC (이 저장소)
- Continuous action
- Gaussian policy
- **Stable-Baselines3 구현**

### 2. Discrete SAC
- Discrete action 지원
- Gumbel-Softmax trick 사용
- 구현 복잡, 성능은 DQN과 유사

### 3. TQC (Truncated Quantile Critics)
- SAC 개선 버전
- 더 안정적
- SB3-Contrib에 구현

## 언제 SAC를 사용할까?

### SAC 선택
- ✅ Continuous action space
- ✅ 샘플 효율성 중요 (시뮬레이션 비용 높음)
- ✅ 고차원 제어 (로보틱스, 제어)
- ✅ 적극적 탐색 필요

### PPO 선택
- ✅ Discrete action space
- ✅ 안정성 최우선
- ✅ 구현 단순성 중요
- ✅ 범용 baseline 필요

## 다음 단계

SAC를 이해했다면:
1. **PPO와 직접 비교** (BipedalWalker, Ant)
2. **TensorBoard**로 entropy 변화 관찰
3. **Temperature tuning** 실험

## 참고 자료

- 원논문: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL (2018)](https://arxiv.org/abs/1801.01290)
- 개선 논문: [Soft Actor-Critic Algorithms and Applications (2019)](https://arxiv.org/abs/1812.05905)
- OpenAI Spinning Up: [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- Stable-Baselines3: [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
