import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "climbmate.db")

def init_db():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 클라이밍 문제 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS climbing_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            image_base64 TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- 홀드 정보 (JSON)
            holds_data TEXT,
            num_holds INTEGER,
            color_distribution TEXT,
            
            -- GPT-4 예측
            gpt4_difficulty TEXT,
            gpt4_type TEXT,
            gpt4_confidence REAL,
            gpt4_response TEXT,
            
            -- 사용자 피드백
            user_difficulty TEXT,
            user_type TEXT,
            user_feedback TEXT,
            is_verified INTEGER DEFAULT 0,
            feedback_at TIMESTAMP,
            feedback_source TEXT,
            feedback_timestamp TIMESTAMP,
            
            -- 메타데이터
            wall_angle TEXT,
            image_width INTEGER,
            image_height INTEGER,
            
            -- 통계
            avg_hold_size REAL,
            max_hold_distance REAL,
            avg_hold_distance REAL,
            height_range REAL,
            horizontal_range REAL
        )
    """)
    
    # 모델 성능 추적 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            total_samples INTEGER,
            verified_samples INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 🎨 홀드 색상 피드백 테이블 (ML 학습용)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hold_color_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id INTEGER,
            hold_id INTEGER,
            
            -- 홀드 위치
            center_x REAL,
            center_y REAL,
            
            -- 색상 특징 (다양한 색상 공간)
            rgb_r INTEGER,
            rgb_g INTEGER,
            rgb_b INTEGER,
            hsv_h REAL,
            hsv_s REAL,
            hsv_v REAL,
            lab_l REAL,
            lab_a REAL,
            lab_b REAL,
            
            -- 통계 특징 (JSON으로 저장)
            color_stats TEXT,
            
            -- AI 예측 vs 사용자 피드백
            predicted_color TEXT,
            user_correct_color TEXT,
            
            -- 메타데이터
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confirmed INTEGER DEFAULT 0,
            
            FOREIGN KEY (problem_id) REFERENCES climbing_problems(id)
        )
    """)
    
    # 인덱스 생성
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_verified 
        ON climbing_problems(is_verified)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at 
        ON climbing_problems(created_at)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hold_feedback_color
        ON hold_color_feedback(user_correct_color)
    """)
    
    # ✅ 기존 테이블에 새 컬럼 추가 (마이그레이션)
    try:
        cursor.execute("ALTER TABLE hold_color_feedback ADD COLUMN confirmed INTEGER DEFAULT 0")
        print("✅ confirmed 컬럼 추가 완료")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("✅ confirmed 컬럼 이미 존재")
        else:
            print(f"⚠️ confirmed 컬럼 추가 실패: {e}")
    
    conn.commit()
    conn.close()
    print("✅ 데이터베이스 초기화 완료")

def save_problem(
    image_base64: str,
    holds_data: List[Dict],
    gpt4_result: Dict,
    wall_angle: Optional[str] = None,
    image_width: int = 0,
    image_height: int = 0,
    statistics: Optional[Dict] = None
) -> int:
    """클라이밍 문제 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 색상 분포 계산
    color_dist = {}
    for hold in holds_data:
        color = hold.get('color_name', 'unknown')
        color_dist[color] = color_dist.get(color, 0) + 1
    
    # 통계 계산
    if statistics is None:
        statistics = calculate_statistics(holds_data)
    
    cursor.execute("""
        INSERT INTO climbing_problems (
            image_base64, holds_data, num_holds, color_distribution,
            gpt4_difficulty, gpt4_type, gpt4_confidence, gpt4_response,
            wall_angle, image_width, image_height,
            avg_hold_size, max_hold_distance, avg_hold_distance,
            height_range, horizontal_range
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_base64,
        json.dumps(holds_data),
        len(holds_data),
        json.dumps(color_dist),
        gpt4_result.get('difficulty'),
        gpt4_result.get('type'),
        gpt4_result.get('confidence', 0.5),
        json.dumps(gpt4_result),
        wall_angle,
        image_width,
        image_height,
        statistics.get('avg_hold_size', 0),
        statistics.get('max_hold_distance', 0),
        statistics.get('avg_hold_distance', 0),
        statistics.get('height_range', 0),
        statistics.get('horizontal_range', 0)
    ))
    
    problem_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"✅ 문제 ID {problem_id} 저장 완료")
    return problem_id

def save_user_feedback(
    problem_id: int,
    user_difficulty: str,
    user_type: str,
    user_feedback: Optional[str] = None
):
    """사용자 피드백 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE climbing_problems
        SET user_difficulty = ?,
            user_type = ?,
            user_feedback = ?,
            is_verified = 1,
            feedback_at = ?
        WHERE id = ?
    """, (
        user_difficulty,
        user_type,
        user_feedback,
        datetime.now(),
        problem_id
    ))
    
    conn.commit()
    conn.close()
    print(f"✅ 문제 ID {problem_id} 피드백 저장 완료")

def get_training_data(min_samples: int = 10) -> List[Dict]:
    """검증된 훈련 데이터 가져오기"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, holds_data, user_difficulty, user_type,
               avg_hold_size, max_hold_distance, avg_hold_distance,
               height_range, horizontal_range, wall_angle
        FROM climbing_problems
        WHERE is_verified = 1
        ORDER BY created_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    training_data = []
    for row in rows:
        training_data.append({
            'id': row[0],
            'holds_data': json.loads(row[1]),
            'difficulty': row[2],
            'type': row[3],
            'avg_hold_size': row[4],
            'max_hold_distance': row[5],
            'avg_hold_distance': row[6],
            'height_range': row[7],
            'horizontal_range': row[8],
            'wall_angle': row[9]
        })
    
    print(f"✅ 훈련 데이터 {len(training_data)}건 로드")
    return training_data

def convert_gpt4_to_training_data():
    """GPT-4 분석 결과를 훈련 데이터로 자동 변환"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # GPT-4 분석이 있지만 사용자 피드백이 없는 데이터 찾기
    cursor.execute("""
        SELECT id, gpt4_difficulty, gpt4_type, gpt4_confidence, gpt4_reasoning
        FROM climbing_problems
        WHERE gpt4_difficulty IS NOT NULL 
        AND gpt4_type IS NOT NULL
        AND is_verified = 0
        AND user_difficulty IS NULL
    """)
    
    gpt4_data = cursor.fetchall()
    converted_count = 0
    
    for row in gpt4_data:
        problem_id, gpt4_difficulty, gpt4_type, gpt4_confidence, gpt4_reasoning = row
        
        # GPT-4 결과를 사용자 피드백으로 임시 저장 (신뢰도가 높은 경우만)
        if gpt4_confidence and gpt4_confidence >= 0.7:
            cursor.execute("""
                UPDATE climbing_problems 
                SET user_difficulty = ?, user_type = ?, is_verified = 1,
                    feedback_source = 'gpt4_auto', feedback_timestamp = datetime('now')
                WHERE id = ?
            """, (gpt4_difficulty, gpt4_type, problem_id))
            converted_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ GPT-4 결과 {converted_count}건을 훈련 데이터로 변환")
    return converted_count

def get_model_stats() -> Dict:
    """모델 성능 통계"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 전체 데이터 수
    cursor.execute("SELECT COUNT(*) FROM climbing_problems")
    total = cursor.fetchone()[0]
    
    # 검증된 데이터 수
    cursor.execute("SELECT COUNT(*) FROM climbing_problems WHERE is_verified = 1")
    verified = cursor.fetchone()[0]
    
    # GPT-4 정확도 (검증된 데이터 기준)
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN gpt4_difficulty = user_difficulty THEN 1 ELSE 0 END) as correct_difficulty,
            SUM(CASE WHEN gpt4_type = user_type THEN 1 ELSE 0 END) as correct_type,
            COUNT(*) as total_verified
        FROM climbing_problems
        WHERE is_verified = 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row[2] > 0:
        difficulty_accuracy = row[0] / row[2]
        type_accuracy = row[1] / row[2]
    else:
        difficulty_accuracy = 0
        type_accuracy = 0
    
    return {
        'total_problems': total,
        'verified_problems': verified,
        'unverified_problems': total - verified,
        'gpt4_difficulty_accuracy': round(difficulty_accuracy * 100, 1),
        'gpt4_type_accuracy': round(type_accuracy * 100, 1),
        'ready_for_training': verified >= 50
    }

def save_hold_color_feedback(
    problem_id: int,
    hold_id: int,
    hold_center: List[float],
    hold_features: Dict,
    predicted_color: str,
    user_correct_color: str
):
    """🎨 홀드 색상 피드백 저장 (ML 학습용)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # RGB, HSV, LAB 추출
    rgb = hold_features.get('dominant_rgb', [128, 128, 128])
    hsv = hold_features.get('dominant_hsv', [0, 0, 128])
    lab = hold_features.get('dominant_lab', [0, 0, 0])
    
    cursor.execute("""
        INSERT INTO hold_color_feedback (
            problem_id, hold_id, center_x, center_y,
            rgb_r, rgb_g, rgb_b,
            hsv_h, hsv_s, hsv_v,
            lab_l, lab_a, lab_b,
            color_stats,
            predicted_color, user_correct_color
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        problem_id, hold_id,
        hold_center[0], hold_center[1],
        rgb[0], rgb[1], rgb[2],
        hsv[0], hsv[1], hsv[2],
        lab[0], lab[1], lab[2],
        json.dumps(hold_features),  # 전체 특징 저장
        predicted_color,
        user_correct_color
    ))
    
    feedback_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"✅ 홀드 색상 피드백 ID {feedback_id} 저장 완료")
    return feedback_id

def get_color_training_data(min_samples: int = 10) -> List[Dict]:
    """🎨 색상 학습 데이터 가져오기 (확인된 피드백만)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            rgb_r, rgb_g, rgb_b,
            hsv_h, hsv_s, hsv_v,
            lab_l, lab_a, lab_b,
            color_stats,
            user_correct_color
        FROM hold_color_feedback
        WHERE confirmed = 1
        ORDER BY created_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    training_data = []
    gray_to_white_count = 0
    for row in rows:
        color_stats = json.loads(row[9]) if row[9] else {}
        correct_color = row[10]
        
        # 🔄 회색을 흰색으로 변환 (gray 개념 제거)
        if correct_color == 'gray':
            correct_color = 'white'
            gray_to_white_count += 1
        
        training_data.append({
            'rgb': [row[0], row[1], row[2]],
            'hsv': [row[3], row[4], row[5]],
            'lab': [row[6], row[7], row[8]],
            'color_stats': color_stats,
            'correct_color': correct_color,
            'area': 0,  # 기본값
            'circularity': 0  # 기본값
        })
    
    if gray_to_white_count > 0:
        print(f"   🔄 학습 데이터: gray {gray_to_white_count}개 → white로 변환")
    print(f"✅ 색상 학습 데이터 {len(training_data)}건 로드")
    return training_data

def get_all_color_feedbacks() -> List[Dict]:
    """🎨 모든 홀드 색상 피드백 가져오기 (관리용)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, problem_id, hold_id,
            center_x, center_y,
            rgb_r, rgb_g, rgb_b,
            hsv_h, hsv_s, hsv_v,
            predicted_color, user_correct_color,
            created_at, confirmed
        FROM hold_color_feedback
        ORDER BY created_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    feedbacks = []
    for row in rows:
        feedbacks.append({
            'id': row[0],
            'problem_id': row[1],
            'hold_id': row[2],
            'center': [row[3], row[4]],
            'rgb': [row[5], row[6], row[7]],
            'hsv': [row[8], row[9], row[10]],
            'predicted_color': row[11],
            'user_correct_color': row[12],
            'created_at': row[13],
            'confirmed': row[14] == 1  # Boolean으로 변환
        })
    
    return feedbacks

def confirm_color_feedback(feedback_id: int):
    """🎨 색상 피드백 확인 (ML 학습 데이터로 확정)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE hold_color_feedback 
        SET confirmed = 1 
        WHERE id = ?
    """, (feedback_id,))
    
    conn.commit()
    conn.close()
    
    print(f"✅ 피드백 ID {feedback_id} 확인 완료 (ML 학습용)")
    return True

def confirm_all_unconfirmed_feedbacks():
    """🎨 모든 미확인 피드백을 한 번에 확인 처리"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 미확인 피드백 개수 확인
    cursor.execute("SELECT COUNT(*) FROM hold_color_feedback WHERE confirmed = 0")
    count = cursor.fetchone()[0]
    
    if count == 0:
        conn.close()
        return 0
    
    # 모두 확인 처리
    cursor.execute("""
        UPDATE hold_color_feedback 
        SET confirmed = 1 
        WHERE confirmed = 0
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✅ {count}개의 피드백 일괄 확인 완료 (ML 학습용)")
    return count

def delete_color_feedback(feedback_id: int):
    """🎨 색상 피드백 삭제"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM hold_color_feedback WHERE id = ?", (feedback_id,))
    
    conn.commit()
    conn.close()
    
    print(f"✅ 피드백 ID {feedback_id} 삭제 완료")
    return True

def calculate_statistics(holds_data: List[Dict]) -> Dict:
    """홀드 데이터로부터 통계 계산"""
    if not holds_data:
        return {}
    
    import numpy as np
    
    areas = [h.get('area', 0) for h in holds_data]
    centers = [h.get('center', [0, 0]) for h in holds_data]
    
    # 거리 계산
    distances = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            distances.append(dist)
    
    # 높이/수평 범위
    heights = [c[1] for c in centers]
    horizontals = [c[0] for c in centers]
    
    return {
        'avg_hold_size': float(np.mean(areas)) if areas else 0,
        'max_hold_distance': float(max(distances)) if distances else 0,
        'avg_hold_distance': float(np.mean(distances)) if distances else 0,
        'height_range': float(max(heights) - min(heights)) if len(heights) > 1 else 0,
        'horizontal_range': float(max(horizontals) - min(horizontals)) if len(horizontals) > 1 else 0
    }

# 앱 시작 시 DB 초기화
init_db()

