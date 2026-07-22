import os
import re
import cv2
import numpy as np
import pandas as pd
import argparse

class TemplateParser:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.rank_templates = {}
        self.score_templates = {}
        self.rank_templates_scaled = {}
        self.score_templates_scaled = {}
        self.load_templates()

    def load_templates(self):
        # Load rank templates
        for i in range(10):
            self.rank_templates[str(i)] = cv2.imread(f"OCR/templates/rank/{i}.png", 0)
        self.rank_templates["位"] = cv2.imread("OCR/templates/rank/wei.png", 0)
        
        # Load score templates
        for i in range(10):
            self.score_templates[str(i)] = cv2.imread(f"OCR/templates/score/{i}.png", 0)
        self.score_templates[","] = cv2.imread("OCR/templates/score/comma.png", 0)
        
        # Pre-scale and pre-cast Rank templates (0.5x, float32)
        for char, temp in self.rank_templates.items():
            if temp is not None:
                th, tw = temp.shape
                scaled = cv2.resize(temp, (max(1, int(tw * 0.5)), max(1, int(th * 0.5))), interpolation=cv2.INTER_AREA)
                self.rank_templates_scaled[char] = scaled.astype(np.float32)
                self.rank_templates[char] = temp.astype(np.float32)
                
        # Pre-scale and pre-cast Score templates (0.5x, float32)
        for char, temp in self.score_templates.items():
            if temp is not None:
                th, tw = temp.shape
                scaled = cv2.resize(temp, (max(1, int(tw * 0.5)), max(1, int(th * 0.5))), interpolation=cv2.INTER_AREA)
                self.score_templates_scaled[char] = scaled.astype(np.float32)
                self.score_templates[char] = temp.astype(np.float32)

    def is_box_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        # Calculate horizontal and vertical intersection
        horizontal_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
        vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
        
        if horizontal_overlap > 0 and vertical_overlap > 0:
            overlap_area = horizontal_overlap * vertical_overlap
            min_area = min(w1 * h1, w2 * h2)
            # If overlap area covers >40% of the smaller bounding box, it is a duplicate
            if (overlap_area / min_area) > 0.40:
                return True
        return False

    def match_column(self, gray_img, templates, x_min, x_max, is_rank=False, threshold=0.78):
        # Ensure image is float32 for faster matchTemplate when template is float32
        if gray_img.dtype != np.float32:
            gray_img = gray_img.astype(np.float32)
            
        all_matches = []
        
        for char, temp in templates.items():
            if temp is None:
                continue
            h, w = temp.shape
            res = cv2.matchTemplate(gray_img, temp, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            
            # Dynamically restrict search width for rank digits to prevent capturing score numbers
            # If Rank mode and character is a digit, limit search boundary (150px for equal scale, 75px for 0.5x scale)
            current_x_max = x_max
            if is_rank and char.isdigit():
                current_x_max = min(x_max, 150 if h > 30 else 75)
                
            y_indices, x_indices = loc
            mask = (x_indices >= x_min) & (x_indices <= current_x_max)
            filtered_y = y_indices[mask]
            filtered_x = x_indices[mask]
            
            for y_val, x_val in zip(filtered_y, filtered_x):
                all_matches.append((x_val, y_val, w, h, char, res[y_val, x_val]))
                    
        # Sort by confidence descending
        all_matches.sort(key=lambda x: x[5], reverse=True)
        
        # Non-Maximum Suppression (NMS) using Bounding Box Intersection and X-distance
        keep = []
        for m in all_matches:
            x, y, w, h, char, conf = m
            overlap = False
            for km in keep:
                kx, ky, kw, kh, kchar, kconf = km
                
                # 1. Bounding Box Overlap check
                if self.is_box_overlap(x, y, w, h, kx, ky, kw, kh):
                    overlap = True
                    break
                    
                # 2. X-distance check for extremely close digits on the same line
                if abs(y - ky) < 15: # Same line
                    min_dist = 12 if is_rank else 8
                    if abs(x - kx) < min_dist:
                        overlap = True
                        break
            if not overlap:
                keep.append(m)
                
        # Group by Y coordinate (lines)
        keep.sort(key=lambda x: x[1])
        lines = []
        for m in keep:
            x, y, w, h, char, conf = m
            found_line = False
            for line in lines:
                avg_y = sum(item[1] for item in line) / len(line)
                if abs(y - avg_y) < 25: # Increased tolerance from 10 to 25 to group commas
                    line.append(m)
                    found_line = True
                    break
            if not found_line:
                lines.append([m])
                
        # Reconstruct string for each line by simple left-to-right sorting
        results = []
        for line in lines:
            line.sort(key=lambda x: x[0])
            text = "".join(item[4] for item in line)
            avg_y = sum(item[1] + item[3]/2 for item in line) / len(line)
            avg_x = sum(item[0] + item[2]/2 for item in line) / len(line)
            results.append({'text': text, 'x': avg_x, 'y': avg_y})
            
        return results

    def parse_image(self, image_path, threshold_rank=0.84):
        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = cv2.imread(image_path)
            
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return []
            
        # Check if the image is already cropped and grayscale to bypass
        if len(img.shape) == 2 or img.shape[2] == 1:
            # It's already cropped grayscale image from video parser
            gray = img
            h, w = img.shape
            crop_width = w
        else:
            height, width, _ = img.shape
            # New extreme crop: shift left start by 76px and subtract 10px from right
            crop_x_start = int(width * 0.35) + 76
            crop_x_end = int(width * 0.55) - 10
            crop_y_start = int(height * 0.35)
            crop_y_end = int(height * 0.90)
            cropped_img = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            crop_width = crop_x_end - crop_x_start
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Detect the Top Rank (早番) using 2-step Stateless ROI Cut
        # まず全体等倍固定スキャンで「位」の位置を大まかに特定する
        rank_scan_limit = 190
        gray_rank_roi = gray[0:300, 0:rank_scan_limit]
        first_pass_ranks = self.match_column(gray_rank_roi, self.rank_templates, 0, rank_scan_limit, is_rank=True, threshold=threshold_rank)
        
        if first_pass_ranks:
            # 「位」が含まれる検出 data を探す
            wei_matches = [r for r in first_pass_ranks if "位" in r['text']]
            if wei_matches:
                wei_y = wei_matches[0]['y']
                # 「位」の文字の中心X座標を取得し、その右隣 of バッジノイズをカットする基準を設定
                wei_x = wei_matches[0]['x']
                wei_left = wei_x - 10 # 余裕を持たせた右端カット境界
                
                # アサーションを避けるため、Yスリットの縦幅を64px固定で安全に切り出す
                y_start = int(wei_y - 32)
                y_end = int(wei_y + 32)
                
                # 境界保護と縦幅(64px)の維持
                if y_start < 0:
                    y_end += abs(y_start)
                    y_start = 0
                if y_end > gray_rank_roi.shape[0]:
                    y_start -= (y_end - gray_rank_roi.shape[0])
                    y_end = gray_rank_roi.shape[0]
                    
                y_start = max(0, y_start)
                y_end = min(gray_rank_roi.shape[0], y_end)
                gray_digits_roi = gray_rank_roi[y_start:y_end, 0:rank_scan_limit]
                
                # 高さ64pxの極小スリットに対して【数字と「位」の両方】をスキャンする
                second_pass_ranks = self.match_column(gray_digits_roi, self.rank_templates, 0, rank_scan_limit, is_rank=True, threshold=threshold_rank)
                
                # 検出座標のY復元と、Xフィルタ（「位」の右側のバッジ等のゴミを完全に除外）
                valid_ranks = []
                for r in second_pass_ranks:
                    r['y'] += y_start
                    # 文字列のX座標（中心）が、検出された「位」の左端境界（wei_left）より左、
                    # もしくは自分自身が「位」を含んでいるもののみを採用する
                    if r['x'] < wei_left or "位" in r['text']:
                        valid_ranks.append(r)
                raw_ranks = valid_ranks
            else:
                raw_ranks = first_pass_ranks
        else:
            raw_ranks = []
            
        if not raw_ranks:
            return []
            
        # Select the top-most valid rank (smallest Y)
        raw_ranks.sort(key=lambda r: r['y'])
        top_rank_data = raw_ranks[0]
        
        digits = re.sub(r'[^0-9]', '', top_rank_data['text'])
        if not digits:
            return []
            
        top_rank_val = int(digits)
        top_y = top_rank_data['y']
        
        # Step 2: Predict offsets and coordinates dynamically based on the base rank's position (top_y)
        # Scale parameters to match the input image resolution (from 1080p to 4K etc.)
        height_cropped = gray.shape[0]
        height_orig = int(height_cropped / 0.55)
        scale_ratio = height_orig / 1080.0
        row_pitch = int(201 * scale_ratio)
        
        limit_1 = int(107 * scale_ratio)
        limit_2 = int(166 * scale_ratio)
        
        # Determine the scan range dynamically (Mutually Exclusive)
        if top_y < limit_1:
            offsets = (0, 1, 2)
        elif top_y < limit_2:
            offsets = (0, 1)
        else:
            offsets = (-1, 0, 1)
            
        # Step 3: Match Scores using 1.0x Equal Scale inside narrow vertical slits (50px height)
        score_rows = []
        x_start_score = int(crop_width * 0.29866) # Scale 89px relative to 298px crop_width
        
        for k in offsets:
            rank_val = top_rank_val + k
            base_y = top_y + row_pitch * k
            
            # 中心予測位置(base_y + 60px)から縦幅50px固定でスリットを切り出す
            score_center_y = base_y + int(60 * scale_ratio)
            y_slit_height = int(25 * scale_ratio)
            y_start = int(score_center_y - y_slit_height)
            y_end = int(score_center_y + y_slit_height)
            
            # 境界保護と縦幅の維持
            if y_start < 0:
                y_end += abs(y_start)
                y_start = 0
            if y_end > gray.shape[0]:
                y_start -= (y_end - gray.shape[0])
                y_end = gray.shape[0]
                
            y_start = max(0, y_start)
            y_end = min(gray.shape[0], y_end)
            
            # Slit ROI
            gray_score_roi = gray[y_start:y_end, x_start_score:crop_width]
            
            # Match using 1.0x equal scale score templates (threshold=0.78)
            raw_scores = self.match_column(gray_score_roi, self.score_templates, 0, crop_width - x_start_score, is_rank=False, threshold=0.78)
            
            if raw_scores:
                # Sort from left to right (ascending X)
                raw_scores.sort(key=lambda s: s['x'])
                score_text = "".join(s['text'] for s in raw_scores)
                score_digits = re.sub(r'[^0-9]', '', score_text)
                if len(score_digits) >= 8:
                    score_val = int(score_digits[:8])
                    score_rows.append({'rank': rank_val, 'score': score_val})
        return score_rows

    def validate_scores(self, rows):
        """
        Check if scores are monotonically decreasing with respect to ranks.
        """
        valid_rows = []
        errors = []
        for i in range(len(rows)):
            curr = rows[i]
            if i > 0:
                prev = rows[i-1]
                if curr['score'] > prev['score']:
                    errors.append(f"Validation Error: Rank {curr['rank']} ({curr['score']}) > Rank {prev['rank']} ({prev['score']})")
            valid_rows.append(curr)
            
        if errors:
            print("[WARNING] Score anomalies detected:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("[INFO] All scores passed validation (monotonically decreasing).")
            
        return valid_rows

    def process_and_save(self, image_path, event_id):
        rows = self.parse_image(image_path)
        if not rows:
            print("[WARNING] No rank/score data extracted from the image.")
            return None
            
        validated_rows = self.validate_scores(rows)
        df = pd.DataFrame(validated_rows)
        df_save = pd.DataFrame({'score': df['score'].astype('int32')})
        save_path = os.path.join(self.data_dir, f"ocr_rank_data_{event_id}.parquet")
        df_save.to_parquet(save_path, compression='zstd')
        print(f"[SUCCESS] Template matching parser complete. Saved to {save_path} (N={len(df_save)})")
        return df_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ranking data using template matching.")
    parser.add_argument("--image", required=True, help="Path to the screenshot image")
    parser.add_argument("--event", required=True, help="Event ID (e.g., total_assault_99)")
    parser.add_argument("--outdir", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    tp = TemplateParser(data_dir=args.outdir)
    tp.process_and_save(args.image, args.event)
