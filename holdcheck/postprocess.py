import cv2
import numpy as np

# -------------------------------
# 📌 결과 시각화 (테두리만 강조)
# -------------------------------
def visualize_groups(original_image, hold_data, masks, clicked_id=None, rec_ids=None):
    highlight = original_image.copy()
    rec_ids = rec_ids or []

    for hold in hold_data:
        mask = masks[hold["id"]].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = (0, 255, 0)  # 기본: 초록 테두리
        if clicked_id is not None and hold["id"] == clicked_id:
            color = (0, 0, 255)  # 클릭한 홀드 빨강
        elif hold["id"] in rec_ids:
            color = (255, 0, 0)  # 추천 홀드 파랑

        cv2.drawContours(highlight, contours, -1, color, 3)

    return highlight