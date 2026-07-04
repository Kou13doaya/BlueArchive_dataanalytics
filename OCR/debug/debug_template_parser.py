import cv2
import re
import numpy as np

img = cv2.imread("OCR/total_assault_99 - frame at 0m0s.png")
height, width, _ = img.shape
crop_x_start = int(width * 0.35)
crop_x_end = int(width * 0.65)
cropped_img = img[:, crop_x_start:crop_x_end]
crop_width = crop_x_end - crop_x_start
gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

# Load templates
rank_templates = {}
for i in range(10):
    rank_templates[str(i)] = cv2.imread(f"OCR/templates/rank/{i}.png", 0)
rank_templates["位"] = cv2.imread("OCR/templates/rank/wei.png", 0)

score_templates = {}
for i in range(10):
    score_templates[str(i)] = cv2.imread(f"OCR/templates/score/{i}.png", 0)
score_templates[","] = cv2.imread("OCR/templates/score/comma.png", 0)

def match_column(gray_img, templates, x_min, x_max, is_rank=False):
    all_matches = []
    threshold = 0.85 if is_rank else 0.88
    
    for char, temp in templates.items():
        if temp is None:
            print(f"  [DEBUG] Template for '{char}' is None!")
            continue
        h, w = temp.shape
        res = cv2.matchTemplate(gray_img, temp, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            if x_min <= pt[0] <= x_max:
                all_matches.append((pt[0], pt[1], w, h, char, res[pt[1], pt[0]]))
                
    print(f"  [DEBUG] Total matches before NMS: {len(all_matches)}")
    all_matches.sort(key=lambda x: x[5], reverse=True)
    
    keep = []
    for m in all_matches:
        x, y, w, h, char, conf = m
        overlap = False
        for km in keep:
            kx, ky, kw, kh, kchar, kconf = km
            if abs(x - kx) < w * 0.75 and abs(y - ky) < h * 0.75:
                overlap = True
                break
        if not overlap:
            keep.append(m)
            
    print(f"  [DEBUG] Total matches after NMS: {len(keep)}")
    for km in keep:
        print(f"    Match: '{km[4]}' at X={km[0]}, Y={km[1]}, Conf={km[5]:.3f}")
        
    keep.sort(key=lambda x: x[1])
    lines = []
    for m in keep:
        x, y, w, h, char, conf = m
        found_line = False
        for line in lines:
            avg_y = sum(item[1] for item in line) / len(line)
            if abs(y - avg_y) < 10:
                line.append(m)
                found_line = True
                break
        if not found_line:
            lines.append([m])
            
    results = []
    for line in lines:
        line.sort(key=lambda x: x[0])
        text = "".join(item[4] for item in line)
        avg_y = sum(item[1] + item[3]/2 for item in line) / len(line)
        avg_x = sum(item[0] + item[2]/2 for item in line) / len(line)
        results.append({'text': text, 'x': avg_x, 'y': avg_y})
        
    return results

print("\n--- Matching Ranks ---")
ranks = match_column(gray, rank_templates, 0, int(crop_width * 0.4), is_rank=True)
print("Reconstructed Ranks:")
for r in ranks:
    print(r)

print("\n--- Matching Scores ---")
scores = match_column(gray, score_templates, int(crop_width * 0.4), crop_width, is_rank=False)
print("Reconstructed Scores:")
for s in scores:
    print(s)
