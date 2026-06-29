import os
import re
import argparse
import cv2
import pandas as pd
import numpy as np
import easyocr
from template_parser import TemplateParser

class VideoOCRParser:
    def __init__(self, data_dir=".", use_gpu=False):
        self.data_dir = data_dir
        self.parser = TemplateParser(data_dir=data_dir)
        print("[INFO] Video OCR Parser initialized (Template Matching Mode).")

    def parse_video(self, video_path, sample_interval_sec=1.0, diff_threshold=5.0):
        """
        Process a video file, sample frames, detect changes, run OCR, and aggregate results.
        """
        print(f"[INFO] Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        print(f"[INFO] Video Details: FPS={fps:.2f}, Total Frames={total_frames}, Duration={duration_sec:.2f}s")

        frame_step = int(fps * sample_interval_sec)
        
        aggregated_data = {} # rank -> list of scores (to handle duplicates/voting)
        rank_detected_frames = {} # rank -> list of frame_idx
        last_cropped_frame = None
        
        frame_idx = 0
        processed_count = 0
        skipped_static_count = 0
        
        max_frame = int(fps * 3.0) # Process only the first 3 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx > max_frame:
                print(f"[INFO] Reached 3s limit (frame {frame_idx}). Stopping.")
                break
                
            if frame_idx % frame_step == 0:
                h, w, _ = frame.shape
                # Crop ranking list region
                crop_x_start = int(w * 0.35)
                crop_x_end = int(w * 0.68)
                cropped_img = frame[:, crop_x_start:crop_x_end]
                
                # Check for frame change (skip if static)
                if last_cropped_frame is not None:
                    diff = cv2.absdiff(cropped_img, last_cropped_frame)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff < diff_threshold:
                        skipped_static_count += 1
                        frame_idx += 1
                        continue
                
                last_cropped_frame = cropped_img.copy()
                processed_count += 1
                
                # Save temp frame
                temp_frame_path = f"temp_frame_{frame_idx}.png"
                cv2.imwrite(temp_frame_path, frame)
                
                print(f"[INFO] Processing Frame {frame_idx}/{total_frames} (Time: {frame_idx/fps:.1f}s)...")
                rows = self.parser.parse_image(temp_frame_path)
                
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                    
                # Aggregate results
                for r in rows:
                    rank = r['rank']
                    score = r['score']
                    if rank not in aggregated_data:
                        aggregated_data[rank] = []
                        rank_detected_frames[rank] = []
                    aggregated_data[rank].append(score)
                    rank_detected_frames[rank].append(frame_idx)

            
            frame_idx += 1
            
        cap.release()
        print(f"[INFO] Video processing finished. Processed frames: {processed_count}, Skipped static frames: {skipped_static_count}")
        
        # Check for missing ranks and backtrack
        detected_ranks = sorted(list(aggregated_data.keys()))
        if detected_ranks:
            min_r = min(detected_ranks)
            max_r = max(detected_ranks)
            missing_ranks = []
            for r in range(min_r, max_r + 1):
                if r not in aggregated_data:
                    missing_ranks.append(r)
            
            if missing_ranks:
                print(f"[INFO] Initial missing ranks: {missing_ranks}")
                cap_bt = cv2.VideoCapture(video_path)
                for G in missing_ranks:
                    # Find previous detected rank
                    prev_r = G - 1
                    while prev_r >= min_r and prev_r not in aggregated_data:
                        prev_r -= 1
                    # Find next detected rank
                    next_r = G + 1
                    while next_r <= max_r and next_r not in aggregated_data:
                        next_r += 1
                        
                    if prev_r in aggregated_data and next_r in aggregated_data:
                        start_f = max(rank_detected_frames[prev_r])
                        end_f = min(rank_detected_frames[next_r])
                        
                        # Restrict backtrack window size to prevent infinite loop or huge scans
                        if start_f < end_f and (end_f - start_f) <= 30: # 30 frames is ~1 second, very safe
                            print(f"[INFO] Backtracking: Missing Rank {G} between Rank {prev_r} (frame {start_f}) and Rank {next_r} (frame {end_f})")
                            cap_bt.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                            for f_idx in range(start_f, end_f + 1):
                                ret, frame = cap_bt.read()
                                if not ret:
                                    break
                                # Process the frame
                                h, w, _ = frame.shape
                                crop_x_start = int(w * 0.35)
                                crop_x_end = int(w * 0.68)
                                cropped_img = frame[:, crop_x_start:crop_x_end]
                                
                                temp_frame_path = f"temp_frame_backtrack_{f_idx}.png"
                                cv2.imwrite(temp_frame_path, frame)
                                rows = self.parser.parse_image(temp_frame_path)
                                if os.path.exists(temp_frame_path):
                                    os.remove(temp_frame_path)
                                    
                                for r in rows:
                                    if r['rank'] == G:
                                        if G not in aggregated_data:
                                            aggregated_data[G] = []
                                        aggregated_data[G].append(r['score'])
                                        print(f"  [FOUND] Rank {G} (Score: {r['score']}) in backtracking at frame {f_idx}!")
                cap_bt.release()
        
        # Resolve final score for each rank
        final_results = []
        for rank, scores in sorted(aggregated_data.items()):
            most_common_score = max(set(scores), key=scores.count)
            final_results.append({'rank': rank, 'score': most_common_score})
            
        return final_results

    def clean_anomalies(self, rows):
        """
        Iteratively remove ranking entries that violate the monotonic score decrease constraint.
        Unlike a simple cascade deletion, this intelligently identifies whether the dip (i)
        or the spike (i+1) is the anomaly.
        """
        cleaned_rows = list(rows)
        while True:
            violation_idx = -1
            for i in range(len(cleaned_rows) - 1):
                if cleaned_rows[i]['score'] < cleaned_rows[i+1]['score']:
                    violation_idx = i
                    break
            if violation_idx == -1:
                break
                
            # If the next element is the last element, it's definitely the anomaly.
            # If the current element's score is >= the score of the element after next (i+2),
            # then the next element (i+1) is a temporary spike, so we remove i+1.
            # Otherwise, the current element (i) is a temporary dip, so we remove i.
            remove_idx = violation_idx
            if violation_idx + 1 == len(cleaned_rows) - 1:
                remove_idx = violation_idx + 1
            elif cleaned_rows[violation_idx]['score'] >= cleaned_rows[violation_idx + 2]['score']:
                remove_idx = violation_idx + 1
                
            print(f"[INFO] Removing OCR anomaly: Rank {cleaned_rows[remove_idx]['rank']} (Score: {cleaned_rows[remove_idx]['score']})")
            cleaned_rows.pop(remove_idx)
            
        return cleaned_rows

    def process_and_save(self, video_path, event_id, sample_interval_sec=1.0):
        rows = self.parse_video(video_path, sample_interval_sec=sample_interval_sec)
        if not rows:
            print("[WARNING] No ranking data extracted from the video.")
            return None
            
        # Clean up any remaining OCR anomalies (e.g. rank digit dropouts)
        cleaned_rows = self.clean_anomalies(rows)
        
        validated_rows = self.parser.validate_scores(cleaned_rows)
        
        # Check for missing ranks to help user verify completeness
        ranks_found = [r['rank'] for r in validated_rows]
        if ranks_found:
            min_rank = min(ranks_found)
            max_rank = max(ranks_found)
            expected_ranks = set(range(min_rank, max_rank + 1))
            missing_ranks = expected_ranks - set(ranks_found)
            if missing_ranks:
                print(f"[WARNING] Missing ranks in the sequence: {sorted(list(missing_ranks))}")
        
        df = pd.DataFrame(validated_rows)
        df_save = pd.DataFrame({'score': df['score'].astype('int32')})
        
        save_path = os.path.join(self.data_dir, f"ocr_rank_data_{event_id}.parquet")
        df_save.to_parquet(save_path, compression='zstd')
        print(f"[SUCCESS] Video OCR processing complete. Saved to {save_path} (N={len(df_save)})")
        
        return df_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ranking data from Blue Archive scroll videos.")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--event", required=True, help="Event ID (e.g., R43)")
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds")
    parser.add_argument("--outdir", default=".", help="Output directory for the parquet file")
    
    args = parser.parse_args()
    
    video_ocr = VideoOCRParser(data_dir=args.outdir)
    video_ocr.process_and_save(args.video, args.event, sample_interval_sec=args.interval)
