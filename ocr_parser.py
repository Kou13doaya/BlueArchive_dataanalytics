import os
import re
import argparse
import cv2
import pandas as pd
import easyocr

class OCRParser:
    def __init__(self, data_dir=".", use_gpu=False):
        self.data_dir = data_dir
        # Initialize EasyOCR reader for Japanese and English
        self.reader = easyocr.Reader(['en', 'ja'], gpu=use_gpu)
        print("[INFO] EasyOCR Reader initialized.")

    def parse_image(self, image_path):
        """
        Extract rank and score from a single image using EasyOCR.
        """
        print(f"[INFO] Processing image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return []
            
        height, width, _ = img.shape
        
        # Crop the ranking list region: X between 35% and 65% of width.
        # Shifted slightly left (from 38%) to prevent cutting off the first digits of 3/4-digit ranks.
        crop_x_start = int(width * 0.35)
        crop_x_end = int(width * 0.65)
        cropped_img = img[:, crop_x_start:crop_x_end]
        
        # Save temporary cropped image for debugging/EasyOCR
        temp_cropped_path = "temp_cropped.png"
        cv2.imwrite(temp_cropped_path, cropped_img)
        
        results = self.reader.readtext(temp_cropped_path)
        
        # Clean up temp file
        if os.path.exists(temp_cropped_path):
            os.remove(temp_cropped_path)
            
        crop_width = crop_x_end - crop_x_start
        
        ranks = []
        scores = []
        
        for bbox, text, prob in results:
            # bbox: [top-left, top-right, bottom-right, bottom-left]
            x_center = (bbox[0][0] + bbox[1][0]) / 2
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            bbox_height = bbox[2][1] - bbox[0][1]
            cleaned_text = text.strip()
            
            # Rank filtering:
            # 1. Left side of cropped image (X < 40% of cropped width)
            # 2. Smaller font height (typically < 32 pixels in 1080p)
            # 3. Contains "位" or is a pure number matching rank pattern
            is_left_side = x_center < (crop_width * 0.4)
            rank_match = re.search(r'([0-9,]+)\s*位', cleaned_text)
            
            if is_left_side:
                if rank_match:
                    r_val = int(rank_match.group(1).replace(',', ''))
                    ranks.append({'value': r_val, 'x': x_center, 'y': y_center, 'height': bbox_height, 'text': cleaned_text})
                else:
                    # Fallback for when "位" is missed but it's a number in the rank column with smaller height
                    num_match = re.match(r'^([0-9,]+)$', cleaned_text)
                    if num_match and bbox_height < 32:
                        r_val = int(num_match.group(1).replace(',', ''))
                        if r_val < 10000: # Ranks are under 10000
                            ranks.append({'value': r_val, 'x': x_center, 'y': y_center, 'height': bbox_height, 'text': cleaned_text})
            
            # Score filtering:
            # 1. Right side of cropped image (X >= 40% of cropped width)
            # 2. Larger font height (typically >= 26 pixels)
            # 3. Matches score pattern (comma-separated numbers)
            is_right_side = x_center >= (crop_width * 0.4)
            score_match = re.search(r'([0-9]{2,3}(,[0-9]{3})+)', cleaned_text)
            
            if is_right_side and bbox_height >= 26:
                if score_match:
                    s_val = int(score_match.group(1).replace(',', ''))
                    scores.append({'value': s_val, 'x': x_center, 'y': y_center, 'height': bbox_height, 'text': cleaned_text})
                else:
                    # Fallback if comma is missed but it's a large number in the score column
                    num_match = re.match(r'^([0-9]{7,9})$', cleaned_text)
                    if num_match:
                        s_val = int(num_match.group(1))
                        scores.append({'value': s_val, 'x': x_center, 'y': y_center, 'height': bbox_height, 'text': cleaned_text})
                
        # Match ranks to scores based on relative positions
        # Score should be at a similar height (Y) to the rank, but slightly right.
        # Since we crop tightly, we expect Y difference to be small (e.g. -20 to 20 px).
        # We search for the score closest in Y-axis alignment.
        rows = []
        for r in ranks:
            matched_score = None
            min_y_diff = float('inf')
            
            for s in scores:
                y_diff = s['y'] - r['y']
                x_diff = s['x'] - r['x']
                
                # Restore original layout-specific thresholds:
                # Score center Y is 30 to 90px below Rank center Y
                # Score center X is -50 to 220px right of Rank center X
                if 30 <= y_diff <= 90 and -50 <= x_diff <= 220:
                    dist = y_diff * y_diff + x_diff * x_diff
                    if dist < min_y_diff:
                        min_y_diff = dist
                        matched_score = s['value']
            
            if matched_score is not None:
                rows.append({'rank': r['value'], 'score': matched_score})
            else:
                print(f"[WARNING] Could not find matching score for Rank {r['value']} (Y={r['y']})")
                
        # Sort rows by rank
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
        print(f"[SUCCESS] OCR processing complete. Saved to {save_path} (N={len(df_save)})")
        
        return df_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ranking data from Blue Archive screenshots.")
    parser.add_argument("--image", required=True, help="Path to the screenshot image")
    parser.add_argument("--event", required=True, help="Event ID (e.g., R43)")
    parser.add_argument("--outdir", default=".", help="Output directory for the parquet file")
    
    args = parser.parse_args()
    
    ocr = OCRParser(data_dir=args.outdir)
    ocr.process_and_save(args.image, args.event)
