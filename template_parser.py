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
        self.load_templates()

    def load_templates(self):
        # Load rank templates
        for i in range(10):
            self.rank_templates[str(i)] = cv2.imread(f"templates/rank/{i}.png", 0)
        self.rank_templates["位"] = cv2.imread("templates/rank/wei.png", 0)
        
        # Load score templates
        for i in range(10):
            self.score_templates[str(i)] = cv2.imread(f"templates/score/{i}.png", 0)
        self.score_templates[","] = cv2.imread("templates/score/comma.png", 0)

    def match_column(self, gray_img, templates, x_min, x_max, is_rank=False):
        all_matches = []
        threshold = 0.84  # Lowered from 0.88 to prevent last digit dropouts
        
        for char, temp in templates.items():
            if temp is None:
                continue
            h, w = temp.shape
            res = cv2.matchTemplate(gray_img, temp, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                if x_min <= pt[0] <= x_max:
                    all_matches.append((pt[0], pt[1], w, h, char, res[pt[1], pt[0]]))
                    
        # Sort by confidence descending
        all_matches.sort(key=lambda x: x[5], reverse=True)
        
        # Non-Maximum Suppression (NMS)
        keep = []
        for m in all_matches:
            x, y, w, h, char, conf = m
            overlap = False
            for km in keep:
                kx, ky, kw, kh, kchar, kconf = km
                # Check overlap in 2D box
                # If they overlap horizontally and vertically significantly
                if abs(x - kx) < w * 0.75 and abs(y - ky) < h * 0.75:
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
                
        # Reconstruct string for each line
        results = []
        for line in lines:
            line.sort(key=lambda x: x[0])
            text = "".join(item[4] for item in line)
            avg_y = sum(item[1] + item[3]/2 for item in line) / len(line)
            avg_x = sum(item[0] + item[2]/2 for item in line) / len(line)
            results.append({'text': text, 'x': avg_x, 'y': avg_y})
            
        return results

    def parse_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return []
            
        height, width, _ = img.shape
        crop_x_start = int(width * 0.35)
        crop_x_end = int(width * 0.68)
        cropped_img = img[:, crop_x_start:crop_x_end]
        crop_width = crop_x_end - crop_x_start
        
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # Match Ranks on the left side (X < 150px)
        raw_ranks = self.match_column(gray, self.rank_templates, 0, 150, is_rank=True)
        # Match Scores on the right side (X >= 150px)
        raw_scores = self.match_column(gray, self.score_templates, 150, crop_width, is_rank=False)
        
        # Parse Rank values
        ranks = []
        for r in raw_ranks:
            digits = re.sub(r'[^0-9]', '', r['text'])
            if 1 <= len(digits) <= 4:
                val = int(digits)
                ranks.append({'value': val, 'x': r['x'], 'y': r['y']})
                
        # Parse Score values
        scores = []
        for s in raw_scores:
            digits = re.sub(r'[^0-9]', '', s['text'])
            if 7 <= len(digits) <= 9:
                val = int(digits)
                scores.append({'value': val, 'x': s['x'], 'y': s['y']})
                
        # Match Rank and Score by Y-coordinate spacing
        rows = []
        for r in ranks:
            matched_score = None
            min_y_diff = float('inf')
            
            for s in scores:
                y_diff = s['y'] - r['y']
                x_diff = s['x'] - r['x']
                
                # Rank to Score spacing constraints:
                # Score is 30px to 90px below Rank
                # Score is -50px to 220px right of Rank
                if 30 <= y_diff <= 90 and -50 <= x_diff <= 220:
                    dist = y_diff * y_diff + x_diff * x_diff
                    if dist < min_y_diff:
                        min_y_diff = dist
                        matched_score = s['value']
                        
            if matched_score is not None:
                rows.append({'rank': r['value'], 'score': matched_score})
                
        rows.sort(key=lambda x: x['rank'])
        return rows

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
