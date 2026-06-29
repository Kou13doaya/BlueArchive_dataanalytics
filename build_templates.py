import os
import re
import cv2
import numpy as np
import easyocr

def merge_close_boxes(boxes, max_dist=4):
    if not boxes:
        return []
    # boxes is list of (x, y, w, h) sorted by x
    merged = []
    curr = boxes[0]
    for next_box in boxes[1:]:
        curr_right = curr[0] + curr[2]
        gap = next_box[0] - curr_right
        
        if gap < max_dist:
            new_x = min(curr[0], next_box[0])
            new_y = min(curr[1], next_box[1])
            new_w = max(curr[0] + curr[2], next_box[0] + next_box[2]) - new_x
            new_h = max(curr[1] + curr[3], next_box[1] + next_box[3]) - new_y
            curr = (new_x, new_y, new_w, new_h)
        else:
            merged.append(curr)
            curr = next_box
    merged.append(curr)
    return merged

def build_templates():
    print("[INFO] Initializing EasyOCR Reader...")
    reader = easyocr.Reader(['en', 'ja'], gpu=False)
    
    images = [
        "OCR/total_assault_99 - frame at 0m0s.png",
        "OCR/total_assault_99 - frame at 0m1s.jpg",
        "OCR/total_assault_99 - frame at 0m3s.jpg",
        "OCR/total_assault_99 - frame at 0m4s.jpg"
    ]
    
    os.makedirs("templates/rank", exist_ok=True)
    os.makedirs("templates/score", exist_ok=True)
    
    needed_rank = set("0123456789位")
    needed_score = set("0123456789,")
    
    saved_rank = set()
    saved_score = set()
    
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
            
        print(f"\n[INFO] Processing {img_path} for templates...")
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        crop_x_start = int(width * 0.35)
        crop_x_end = int(width * 0.68)
        cropped_img = img[:, crop_x_start:crop_x_end]
        crop_width = crop_x_end - crop_x_start
        
        gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(cropped_img)
        
        for bbox, text, prob in results:
            cleaned_text = text.replace(" ", "").strip()
            if not cleaned_text:
                continue
                
            x_min = int(min(pt[0] for pt in bbox))
            x_max = int(max(pt[0] for pt in bbox))
            y_min = int(min(pt[1] for pt in bbox))
            y_max = int(max(pt[1] for pt in bbox))
            
            pad = 2
            x_min = max(0, x_min - pad)
            x_max = min(crop_width, x_max + pad)
            y_min = max(0, y_min - pad)
            y_max = min(height, y_max + pad)
            
            w_w = x_max - x_min
            w_h = y_max - y_min
            if w_w <= 5 or w_h <= 5:
                continue
                
            x_center = (x_min + x_max) / 2
            is_left_side = x_center < (crop_width * 0.4)
            is_right_side = x_center >= (crop_width * 0.4)
            
            is_rank = False
            is_score = False
            
            rank_match = re.match(r'^([0-9,]+位?|位)$', cleaned_text)
            if is_left_side and rank_match:
                is_rank = True
                
            score_match = re.match(r'^([0-9]{2,3}(,[0-9]{3})+|[0-9]{7,9})$', cleaned_text)
            if is_right_side and score_match:
                is_score = True
                
            if not (is_rank or is_score):
                continue
                
            word_gray = gray_cropped[y_min:y_max, x_min:x_max]
            _, word_thresh = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if np.mean(word_thresh) > 127:
                word_thresh = cv2.bitwise_not(word_thresh)
                
            contours, _ = cv2.findContours(word_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            char_boxes = []
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if cw >= 2 and ch >= 5:
                    char_boxes.append((cx, cy, cw, ch))
                    
            char_boxes.sort(key=lambda b: b[0])
            
            # Grouping parameters
            if is_rank:
                merged_boxes = merge_close_boxes(char_boxes, max_dist=4)
            else:
                merged_boxes = char_boxes # Do not merge digits/commas for scores
            
            if len(merged_boxes) == len(cleaned_text):
                category = "rank" if is_rank else "score"
                target_set = saved_rank if is_rank else saved_score
                needed_set = needed_rank if is_rank else needed_score
                
                for idx, char in enumerate(cleaned_text):
                    if char in needed_set and char not in target_set:
                        cx, cy, cw, ch = merged_boxes[idx]
                        char_img = word_gray[cy:cy+ch, cx:cx+cw]
                        
                        # Map non-ASCII file names to ASCII to prevent cv2.imwrite bugs on Windows
                        filename = char
                        if char == ',':
                            filename = "comma"
                        elif char == '位':
                            filename = "wei"
                            
                        save_name = f"templates/{category}/{filename}.png"
                        cv2.imwrite(save_name, char_img)
                        target_set.add(char)
                        print(f"  [SAVED] {category.upper()} template for '{char}' -> {save_name} (Size: {cw}x{ch})")
                        
    print("\n--- Summary ---")
    print(f"Rank templates saved: {sorted(list(saved_rank))} / {list(needed_rank)}")
    print(f"Score templates saved: {sorted(list(saved_score))} / {list(needed_score)}")

if __name__ == "__main__":
    build_templates()
