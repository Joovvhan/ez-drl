# DQN (Deep Q-Network)

## 개요

DQN은 **Value-based** 강화학습 알고리즘으로, Q-learning에 딥러닝을 결합한 방법입니다.

## 핵심 아이디어

### Q-Learning
- **Q-function**: Q(s, a)는 상태 s에서 행동 a를 했을 때의 기대 수익을 나타냅니다
- **최적 정책**: Q 값이 가장 큰 행동을 선택합니다

### Deep Q-Network
- Q-function을 신경망으로 근사합니다
- **Bellman Equation**을 이용해 학습합니다:
  ```
  Q(s, a) = r + γ * max Q(s', a')
  ```

## 주요 기법

### 1. Experience Replay
- 과거 경험 (s, a, r, s')을 replay buffer에 저장
- 랜덤하게 샘플링하여 학습
- **장점**:
  - 데이터 간 상관관계 제거
  - 샘플 효율성 향상

### 2. Target Network
- 학습 안정성을 위해 별도의 타겟 네트워크 사용
- 일정 주기마다 타겟 네트워크를 업데이트
- **장점**: 학습 목표가 급격히 변하는 것을 방지

### 3. Epsilon-Greedy Exploration
- ε 확률로 랜덤 행동, (1-ε) 확률로 최적 행동
- 학습이 진행됨에 따라 ε를 감소시킴

## 장점

1. **Discrete action에서 강력함**
   - Action space가 작을 때 효과적
   - CartPole, LunarLander 같은 환경에서 좋은 성능

2. **Off-policy 학습**
   - Replay buffer로 샘플 효율성 향상
   - 과거 데이터 재사용 가능

3. **구현이 상대적으로 단순함**

## 한계

### 1. Continuous Action 불가 ❌
- Q(s, a)는 모든 action에 대해 계산해야 함
- Action space가 연속일 경우 불가능
- **해결책**: 정책 기반 방법 (PPO, SAC)

### 2. Overestimation Bias
- max 연산으로 인해 Q 값을 과대평가하는 경향
- **개선**: Double DQN, Dueling DQN 등

### 3. 고차원 환경에서 불안정
- State space가 크거나 복잡할 때 학습 어려움
- **개선**: Rainbow DQN 등

## 적용 환경

| 환경 | 적합도 | 이유 |
|-----|-------|------|
| CartPole | ✅ 매우 적합 | Discrete, 저차원 |
| LunarLander | ✅ 적합 | Discrete, 중간 복잡도 |
| Breakout (Atari) | ✅ 적합 | Discrete, 이미지 입력 (CNN) |
| BipedalWalker | ❌ 불가 | Continuous action |
| Ant | ❌ 불가 | Continuous action |

## 하이퍼파라미터 설명

```python
model = DQN(
    learning_rate=1e-4,          # 학습률
    buffer_size=100000,          # Replay buffer 크기
    learning_starts=10000,       # 학습 시작 전 수집할 샘플 수
    batch_size=32,               # 미니배치 크기
    tau=1.0,                     # Target network soft update 계수
    gamma=0.99,                  # Discount factor
    target_update_interval=500,  # Target network 업데이트 주기
    exploration_fraction=0.1,    # Exploration 기간 비율
    exploration_final_eps=0.05,  # 최종 epsilon 값
)
```

## 학습 곡선 해석

### 정상적인 학습
- 초반: 랜덤한 탐색으로 낮은 보상
- 중반: 급격한 성능 향상
- 후반: 수렴하며 안정화

### 문제 신호
- **발산**: Learning rate가 너무 높음
- **수렴 실패**: Exploration이 부족하거나 네트워크 용량 부족
- **진동**: Target update interval 조정 필요

## 다음 단계

DQN의 한계를 이해했다면:
1. **REINFORCE**를 통해 정책 기반 방법의 직관 이해
2. **PPO**로 Actor-Critic의 안정성 확인
3. **SAC**로 continuous control의 발전 이해

## 참고 자료

- 원논문: [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602)
- 개선 논문: [Human-level control through deep reinforcement learning (Nature, 2015)](https://www.nature.com/articles/nature14236)
