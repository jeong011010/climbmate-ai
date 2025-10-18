"""
🚀 OpenAI Vision API를 사용한 클라이밍 벽 분석
- 로컬 AI 모델 대신 OpenAI 서버 사용
- 무제한 동시 처리 가능
- 메모리 사용량 최소화
"""
import openai
import base64
import json
import os
from typing import Dict, List, Any

# OpenAI 클라이언트 초기화
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def analyze_climbing_wall_with_openai(image_base64: str) -> Dict[str, Any]:
    """
    OpenAI Vision API로 클라이밍 벽 분석
    
    Args:
        image_base64: Base64 인코딩된 이미지
        
    Returns:
        분석 결과 딕셔너리
    """
    try:
        # OpenAI Vision API 호출
        response = await client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """이 클라이밍 벽 이미지를 분석해주세요:

1. 홀드(손잡이)들을 감지하고 개수를 세어주세요
2. 홀드들을 색상별로 그룹핑해주세요
3. 각 색상 그룹의 홀드 개수를 알려주세요
4. 클라이밍 문제(루트)를 제안해주세요

결과를 다음 JSON 형식으로 반환해주세요:
{
    "total_holds": 숫자,
    "color_groups": [
        {"color": "색상명", "count": 숫자, "holds": [{"x": x좌표, "y": y좌표}]}
    ],
    "suggested_problems": [
        {"name": "문제명", "difficulty": "난이도", "holds": ["색상1", "색상2"]}
    ]
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        # 응답 파싱
        content = response.choices[0].message.content
        
        # JSON 추출 (```json ... ``` 형태일 수 있음)
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_str = content.strip()
        
        # JSON 파싱
        result = json.loads(json_str)
        
        # 결과 포맷팅
        formatted_result = {
            "problems": result.get("suggested_problems", []),
            "statistics": {
                "total_holds": result.get("total_holds", 0),
                "total_problems": len(result.get("suggested_problems", [])),
                "color_groups": result.get("color_groups", [])
            },
            "annotated_image_base64": None,  # OpenAI는 이미지 주석을 제공하지 않음
            "analysis_method": "openai_vision"
        }
        
        return formatted_result
        
    except Exception as e:
        print(f"❌ OpenAI Vision 분석 실패: {e}")
        return {
            "problems": [],
            "statistics": {"total_holds": 0, "total_problems": 0},
            "error": str(e),
            "analysis_method": "openai_vision"
        }

def get_openai_status() -> Dict[str, Any]:
    """OpenAI API 상태 확인"""
    try:
        # 간단한 API 호출로 상태 확인
        response = client.models.list()
        return {
            "available": True,
            "message": "OpenAI API 정상 작동",
            "model_count": len(response.data)
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"OpenAI API 오류: {str(e)}"
        }
