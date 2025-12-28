# 빠른 시작 가이드

ez-drl을 처음 시작하는 분들을 위한 가이드입니다.

## 1️⃣ 환경 설정 (5분)

### Conda 사용 (권장)

```bash
# 1. Conda 환경 생성
conda env create -f environment.yml

# 2. 환경 활성화
conda activate ez-drl

# 3. 설치 확인
python -c "import gymnasium; import stable_baselines3; print('설치 완료!')"
```

### pip 사용

```bash
# 1. Python 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt
```

## 2️⃣ 첫 실험 (5분)

### 실험 1: DQN으로 CartPole 학습

**목적**: Value-based 강화학습의 기초 이해

```bash
python train.py --env CartPole-v1 --algo dqn
```

**예상 결과**:
- 5만 timesteps, 약 5분 소요
- 안정적인 학습 곡선
- 모델이 `models/dqn/CartPole-v1/`에 저장됨

### 실험 2: A2C로 CartPole 학습

**목적**: Actor-Critic 방법의 이해

```bash
python train.py --env CartPole-v1 --algo a2c
```

**예상 결과**:
- 5만 timesteps, 약 3분 소요 (병렬 환경 덕분)
- DQN과 유사한 안정성

### 실험 3: PPO로 CartPole 학습

**목적**: 안정적인 정책 학습 이해

```bash
python train.py --env CartPole-v1 --algo ppo
```

**예상 결과**:
- 5만 timesteps, 약 5분 소요
- 매우 안정적인 학습

## 3️⃣ 학습 결과 확인

### TensorBoard로 시각화

```bash
# 전체 결과 보기
tensorboard --logdir logs/

# 브라우저에서 http://localhost:6006 열기
```

**주요 지표**:
- `rollout/ep_rew_mean`: 평균 에피소드 보상 (높을수록 좋음)
- `train/loss`: 학습 손실

### 학습된 모델 테스트

학습할 때 `--test` 옵션 추가:

```bash
python train.py --env CartPole-v1 --algo dqn --test
```

또는 학습된 모델로 별도 테스트:

```python
from utils import test_model
from config import TrainingConfig

config = TrainingConfig(env_name="CartPole-v1", algorithm="dqn")
test_model(config, n_episodes=5)
```

렌더링된 화면에서 학습된 에이전트를 확인할 수 있습니다!

## 4️⃣ 다음 단계

### 실험 4: SAC vs PPO (연속 제어)

**목적**: 샘플 효율성 비교

```bash
# PPO (느림, 안정적)
python train.py --env BipedalWalker-v3 --algo ppo

# SAC (빠름, 효율적)
python train.py --env BipedalWalker-v3 --algo sac
```

**비교 포인트**:
- 같은 성능에 도달하는 데 필요한 timesteps
- 학습 곡선의 안정성
- TensorBoard에서 두 결과 비교

### 실험 5: 이미지 기반 RL (Atari)

**목적**: Representation Learning 이해

```bash
python train.py --env ALE/Breakout-v5 --algo ppo
```

**참고**:
- 학습 시간이 오래 걸립니다 (1~2시간)
- CnnPolicy가 자동으로 사용됩니다
- 시간이 없다면 `--timesteps 100000` 옵션으로 줄여보세요

## 5️⃣ 권장 학습 경로

```
1주차: 기초 이해
├─ Day 1: DQN (Value-based 기초)
├─ Day 2: A2C (Actor-Critic 기초)
├─ Day 3: PPO (안정적 정책 학습)
└─ Day 4: 문서 읽기 (docs/)

2주차: 심화 실험
├─ Day 1: LunarLander (복잡한 환경)
├─ Day 2: BipedalWalker (PPO vs SAC 비교)
├─ Day 3: Ant (고차원 제어)
└─ Day 4: 하이퍼파라미터 튜닝

3주차: 프로젝트
└─ 자신만의 환경에 적용
```

## 6️⃣ 자주 묻는 질문 (FAQ)

### Q1. GPU가 필요한가요?
**A**: 아니요. 모든 예제는 CPU로 실행 가능합니다.
- CartPole, LunarLander: CPU로 충분
- BipedalWalker, Ant: GPU 권장 (하지만 CPU도 가능)
- Breakout: GPU 강력 권장

### Q2. 학습이 너무 느려요
**A**: Timesteps를 줄여보세요 (테스트용):
```python
model.learn(total_timesteps=10000)  # 원래 50000
```

### Q3. 어떤 알고리즘을 선택해야 하나요?
**A**: [비교 가이드](docs/comparison.md) 참고
- **시작**: PPO
- **Discrete**: DQN 또는 PPO
- **Continuous**: PPO 또는 SAC

### Q4. 하이퍼파라미터를 어떻게 조절하나요?
**A**: 각 알고리즘 문서 참고:
- [DQN 가이드](docs/dqn.md)
- [PPO 가이드](docs/ppo.md)
- [SAC 가이드](docs/sac.md)

### Q5. MuJoCo 설치 오류
**A**: MuJoCo는 선택사항입니다.
```bash
# MuJoCo 없이 다른 환경 사용
pip install gymnasium[box2d]  # BipedalWalker
pip install gymnasium[atari]  # Breakout
```

### Q6. 성능이 기대보다 낮아요
**A**: 체크리스트:
1. 충분한 timesteps? (너무 짧으면 성능 낮음)
2. TensorBoard로 학습 곡선 확인
3. 하이퍼파라미터 조절 시도
4. 환경이 알고리즘과 맞나요? (DQN은 continuous 불가 등)

## 7️⃣ 도움말

### 문서
- [README](README.md): 프로젝트 개요
- [알고리즘 비교](docs/comparison.md): DQN vs PPO vs SAC
- [JSON 설정 가이드](configs/README.md): 하이퍼파라미터 관리

### 코드
- `train.py`: 통합 학습 스크립트
- `algorithms/run_algorithm.py`: JSON 기반 학습
- `configs/`: JSON 설정 파일
- `docs/`: 알고리즘 상세 설명

### 외부 자료
- [Stable-Baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

## 8️⃣ 다음 목표

이 저장소를 마스터했다면:

1. **자신만의 환경 적용**
   - Custom Gymnasium 환경 만들기
   - 실제 문제에 RL 적용

2. **고급 알고리즘 탐구**
   - Rainbow DQN
   - TD3 (SAC의 변형)
   - Offline RL

3. **분산 학습**
   - Ray RLlib
   - OpenAI Baselines

4. **실전 프로젝트**
   - 로보틱스 제어
   - 게임 AI
   - 금융 트레이딩

## 🎉 시작하세요!

```bash
# 지금 바로 시작
conda activate ez-drl
python train.py --env CartPole-v1 --algo dqn
```

좋은 학습 되세요! 🚀
