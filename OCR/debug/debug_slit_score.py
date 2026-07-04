import cv2
import numpy as np
import template_parser
import re

path = "OCR/total_assault_99 - frame at 0m0s.png"
parser = template_parser.TemplateParser(data_dir=".")

img = cv2.imread(path)
H, W, _ = img.shape

crop_x_start = int(W * 0.35)
crop_x_end = int(W * 0.55)
crop_y_start = int(H * 0.35)
crop_y_end = int(H * 0.90)
cropped_img = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

# 1行目の順位を検出
gray_rank_roi = gray[0:300, 0:240]
raw_ranks = parser.match_column(gray_rank_roi, parser.rank_templates, 0, 240, is_rank=True)
raw_ranks.sort(key=lambda r: r['y'])
top_rank_data = raw_ranks[0]
top_y = top_rank_data['y']

print(f"Top Rank: '{top_rank_data['text']}' at Y={top_y}")

# スコア検出のテスト
y_start = int(top_y + 30)
y_end = int(top_y + 90)
gray_score_roi = gray[y_start:y_end, 255:cropped_img.shape[1]]

print(f"Slit size: {gray_score_roi.shape[1]}x{gray_score_roi.shape[0]} at Y={y_start} to {y_end}")

raw_scores = parser.match_column(gray_score_roi, parser.score_templates, 0, cropped_img.shape[1] - 255, is_rank=False)
print(f"Detected score chunks: {raw_scores}")
