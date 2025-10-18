# 🔐 환경변수 설정 가이드

## 필수 환경변수

### OPENAI_API_KEY (선택)
GPT-4 Vision을 사용하려면 OpenAI API 키가 필요합니다.

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**설정 방법:**

1. **로컬 개발 (macOS/Linux):**
   ```bash
   # ~/.zshrc 또는 ~/.bashrc에 추가
   echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Docker 실행 시:**
   ```bash
   docker run -e OPENAI_API_KEY="sk-your-key" ...
   ```

3. **없어도 작동:**
   - GPT-4 없이도 규칙 기반 분석으로 작동
   - 50개 이상 피드백 후 자체 모델 사용 가능

## 프론트엔드 환경변수

### VITE_API_URL
백엔드 API URL (기본값: http://localhost:8000)

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
```

## 데이터베이스

자동으로 `backend/climbmate.db` 생성됩니다.

## 모델 파일

학습된 모델은 `backend/models/` 디렉토리에 저장됩니다:
- `difficulty_model.pkl`
- `type_model.pkl`
- `difficulty_encoder.pkl`
- `type_encoder.pkl`

