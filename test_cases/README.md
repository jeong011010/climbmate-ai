# 색상 분류 테스트 케이스

## 📋 설명

이 디렉토리는 색상 분류 함수의 테스트 케이스를 JSON 형식으로 관리합니다.
새로운 피드백이 들어올 때마다 `color_classification_test_cases.json` 파일에 테스트 케이스를 추가하면
자동으로 모든 케이스를 테스트할 수 있습니다.

## 🚀 사용법

### 테스트 실행
```bash
python3 test_color_classification.py
```

### 새 테스트 케이스 추가

`color_classification_test_cases.json` 파일의 `test_cases` 배열에 다음 형식으로 추가하세요:

```json
{
  "id": "hold_XX",
  "name": "홀드 #XX - 색상명",
  "hsv": [H, S, V],
  "expected": "예상되는색상명",
  "description": "설명",
  "date_added": "2024-01-XX",
  "fix_applied": "적용된 수정 내용"
}
```

### 예시

```json
{
  "id": "hold_99",
  "name": "홀드 #99 - Red",
  "hsv": [10, 200, 180],
  "expected": "red",
  "description": "높은 채도 red 테스트",
  "date_added": "2024-01-15",
  "fix_applied": "Red 범위 조정"
}
```

## 📊 필드 설명

- **id**: 테스트 케이스 고유 ID (중복 불가)
- **name**: 테스트 케이스 이름 (표시용)
- **hsv**: [H, S, V] 형식의 HSV 값 배열
  - H: Hue (0-179)
  - S: Saturation (0-255)
  - V: Value/Brightness (0-255)
- **expected**: 예상되는 색상명 (소문자)
  - 가능한 값: `red`, `orange`, `yellow`, `lime`, `green`, `mint`, `blue`, `purple`, `pink`, `white`, `black`, `brown`, `unknown`
- **description**: 테스트 케이스 설명
- **date_added**: 추가 날짜 (선택사항)
- **fix_applied**: 적용된 수정 내용 (선택사항)

## ✅ 테스트 결과

테스트 실행 후 다음 정보가 표시됩니다:
- 각 테스트 케이스의 통과/실패 여부
- 예상 색상 vs 실제 색상
- 신 дроб수도
- 전체 통계 (통과율, 색상별 통계)

## 🔄 유지보수

새로운 피드백이 들어오면:
1. `color_classification_test_cases.json`에 테스트 케이스 추가
2. `python3 test_color_classification.py` 실행
3. 모든 테스트가 통과하는지 확인
4. 테스트가 실패하면 `holdcheck/clustering.py`의 색상 분류 로직 수정

