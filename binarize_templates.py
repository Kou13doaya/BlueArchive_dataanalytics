import os
import cv2
import numpy as np

def binarize_templates():
    categories = ["rank", "score"]
    for cat in categories:
        folder = f"templates/{cat}"
        if not os.path.exists(folder):
            continue
            
        for f in os.listdir(folder):
            if not f.endswith(".png"):
                continue
            path = os.path.join(folder, f)
            img = cv2.imread(path, 0)
            if img is None:
                continue
                
            # Binarize template
            # In BA UI, text is bright and background is dark.
            # We use Otsu's thresholding to get a clean binary mask.
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # If the background ended up white (due to Otsu choosing wrong mode on very light background template), invert it
            if np.mean(thresh) > 127:
                thresh = cv2.bitwise_not(thresh)
                
            # Overwrite original template
            cv2.imwrite(path, thresh)
            print(f"[BINARIZED] {path} (Size: {thresh.shape[1]}x{thresh.shape[0]})")

if __name__ == "__main__":
    binarize_templates()
