# 하이퍼파라미터 설정 파일

이 디렉토리에는 각 알고리즘의 환경별 하이퍼파라미터 설정이 JSON 형식으로 저장되어 있습니다.

## 파일 구조

```
configs/
├── dqn.json          # DQN 하이퍼파라미터
├── a2c.json          # A2C 하이퍼파라미터
├── ppo.json          # PPO 하이퍼파라미터
└── sac.json          # SAC 하이퍼파라미터
```

## JSON 파일 형식

각 JSON 파일은 다음과 같은 구조를 가집니다:

```json
{
  "algorithm": "알고리즘_이름",
  "common": {
    // 모든 환경에 공통으로 적용되는 파라미터
  },
  "environments": {
    "환경_이름": {
      // 환경별 특정 파라미터
      "total_timesteps": 50000,
      "learning_rate": 0.001,
      ...
    }
  }
}
```

## 파라미터 우선순위

1. 명령줄 인자 (최우선)
2. 환경별 설정 (`environments`)
3. 공통 설정 (`common`)
4. 코드 기본값

## 사용 예시

### 기본 사용 (JSON 설정 자동 로드)
```bash
python train.py --env CartPole-v1 --algo ppo
# configs/ppo.json의 CartPole-v1 설정 사용
```

### 특정 파라미터 오버라이드
```bash
python train.py --env CartPole-v1 --algo ppo --learning-rate 0.001
# JSON 설정을 로드하되, learning_rate만 0.001로 변경
```

### 커스텀 설정 파일 사용
```bash
python train.py --env CartPole-v1 --algo ppo --config my_config.json
```

## 새로운 환경 추가

환경을 추가하려면 해당 알고리즘의 JSON 파일을 편집합니다:

```json
{
  "algorithm": "ppo",
  "common": { ... },
  "environments": {
    "기존_환경": { ... },
    "새로운_환경": {
      "total_timesteps": 100000,
      "learning_rate": 0.0003,
      ...
    }
  }
}
```

## 설정 템플릿

### DQN (Discrete Action)
```json
{
  "total_timesteps": 50000,
  "learning_rate": 0.001,
  "buffer_size": 50000,
  "learning_starts": 1000,
  "batch_size": 32,
  "tau": 1.0,
  "target_update_interval": 500,
  "exploration_fraction": 0.1,
  "exploration_initial_eps": 1.0,
  "exploration_final_eps": 0.05
}
```

### A2C (모든 환경)
```json
{
  "total_timesteps": 50000,
  "learning_rate": 0.0007,
  "n_steps": 5
}
```

### PPO (모든 환경)
```json
{
  "total_timesteps": 50000,
  "learning_rate": 0.0003,
  "n_steps": 2048,
  "batch_size": 64,
  "n_epochs": 10
}
```

### SAC (Continuous Action)
```json
{
  "total_timesteps": 300000,
  "learning_rate": 0.0003,
  "buffer_size": 300000,
  "learning_starts": 10000,
  "batch_size": 256
}
```

## 주의사항

1. **JSON 형식**: 유효한 JSON 형식을 유지해야 합니다 (마지막 항목 뒤 쉼표 없음)
2. **문자열 vs 숫자**:
   - 문자열: `"auto"`, `"MlpPolicy"`
   - 숫자: `0.001`, `50000`
   - Boolean: `true`, `false` (소문자)
3. **환경 이름**: Gymnasium 환경 이름과 정확히 일치해야 합니다

## 검증

설정 파일이 올바른지 확인:
```bash
python -m json.tool configs/ppo.json
```

## 기본값으로 되돌리기

설정을 기본값으로 되돌리려면:
```bash
git checkout configs/
```
