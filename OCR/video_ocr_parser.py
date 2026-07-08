import os
import re
import sys
import argparse

# Add project root and current directory to sys.path to allow running from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in [project_root, current_dir]:
    if p not in sys.path:
        sys.path.append(p)

import cv2
import pandas as pd
import numpy as np
import easyocr
from ocr_engine import TemplateParser
from concurrent.futures import ThreadPoolExecutor


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
        rank_detected_frames = {} # rank -> list of frame_idx (for logging/compat)
        rank_detected_msec = {} # rank -> list of timestamp in msec (for VFR-safe backtrack seek)
        last_cropped_frame = None
        
        frame_idx = 0
        skipped_static_count = 0
        
        max_frame = total_frames # Process the entire video duration
        
        import queue
        import threading
        
        MAX_WORKERS = 8
        q = queue.Queue(maxsize=64)
        lock = threading.Lock()
        
        def process_single_frame(task):
            f_idx, f_data = task
            rows = self.parser.parse_image(f_data)
            return f_idx, rows
            
        def process_single_frame_backtrack(task):
            f_idx, f_data = task
            rows = self.parser.parse_image(f_data)
            return f_idx, rows
            
        def worker():
            while True:
                task = q.get()
                if task is None:
                    q.task_done()
                    break
                f_idx, f_msec, f_data = task
                _, rows = process_single_frame((f_idx, f_data))
                
                with lock:
                    for r in rows:
                        rank = r['rank']
                        score = r['score']
                        
                        # 桁落ち防止フィルタ（スレッドセーフ）
                        # すでに確定している最大順位より桁数が少ない場合は桁落ち誤検知と判定
                        if aggregated_data:
                            current_max_digits = len(str(max(aggregated_data.keys())))
                            if len(str(rank)) < current_max_digits:
                                continue
                            
                        if rank not in aggregated_data:
                            aggregated_data[rank] = []
                            rank_detected_frames[rank] = []
                            rank_detected_msec[rank] = []
                        aggregated_data[rank].append(score)
                        rank_detected_frames[rank].append(f_idx)
                        rank_detected_msec[rank].append(f_msec)
                q.task_done()
                
        # Start worker threads
        threads = []
        for _ in range(MAX_WORKERS):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
            
        # Producer loop
        while True:
            # タイムスタンプはcap.read()の前に取得するとVFR動画でも正確
            frame_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx > max_frame:
                break
                
            if frame_idx % frame_step == 0:
                h, w, _ = frame.shape
                crop_x_start = int(w * 0.35) + 76
                crop_x_end = int(w * 0.55) - 10
                crop_y_start = int(h * 0.35)
                crop_y_end = int(h * 0.90)
                cropped_img = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                
                # Convert to grayscale early for diff calculation and template matching
                cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                
                # Check for frame change (skip if static)
                if last_cropped_frame is not None:
                    diff = cv2.absdiff(cropped_gray, last_cropped_frame)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff < diff_threshold:
                        skipped_static_count += 1
                        frame_idx += 1
                        continue
                
                last_cropped_frame = cropped_gray.copy()
                
                # Put in queue with msec timestamp (blocks if queue is full)
                q.put((frame_idx, frame_msec, cropped_gray))
            
            frame_idx += 1
            
        # Signal workers to exit
        for _ in range(MAX_WORKERS):
            q.put(None)
            
        for t in threads:
            t.join()
            
        cap.release()
        print(f"[INFO] Video frame processing finished. Skipped static frames: {skipped_static_count}")
        
        # IQRによる順位の外れ値排除（9718などの桁挿入誤読をシングルスレッドで安全に排除）
        if len(aggregated_data) >= 4:
            detected_ranks = sorted(aggregated_data.keys())
            q1 = detected_ranks[len(detected_ranks) // 4]
            q3 = detected_ranks[3 * len(detected_ranks) // 4]
            iqr = q3 - q1
            upper_fence = q3 + iqr * 3
            
            for r in [r for r in detected_ranks if r > upper_fence]:
                print(f"[INFO] Removing rank outlier (IQR filter): {r} (upper fence: {upper_fence})")
                del aggregated_data[r]
                del rank_detected_frames[r]
                del rank_detected_msec[r]
        
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
                
                # Plan backtracking windows (start_msec, end_msec, target_ranks)
                # Use msec timestamps for VFR-safe seeking
                bt_windows = {}  # (start_msec, end_msec) -> set of target ranks
                
                for G in missing_ranks:
                    prev_r = G - 1
                    while prev_r >= min_r and prev_r not in aggregated_data:
                        prev_r -= 1
                    next_r = G + 1
                    while next_r <= max_r and next_r not in aggregated_data:
                        next_r += 1
                        
                    if prev_r in aggregated_data and next_r in aggregated_data:
                        start_msec = max(rank_detected_msec[prev_r])
                        end_msec = min(rank_detected_msec[next_r])
                        start_f = max(rank_detected_frames[prev_r])
                        end_f = min(rank_detected_frames[next_r])
                        
                        # VFR対応: フレーム数ではなくタイムスタンプ差（ms）で判定
                        # 30フレーム相当の時間を上限とする（約1500ms）
                        if start_msec < end_msec and (end_msec - start_msec) <= 1500:
                            print(f"[INFO] Backtracking plan: Rank {G} between Rank {prev_r} (frame {start_f}, {start_msec:.0f}ms) and Rank {next_r} (frame {end_f}, {end_msec:.0f}ms)")
                            key = (start_msec, end_msec)
                            if key not in bt_windows:
                                bt_windows[key] = set()
                            bt_windows[key].add(G)
                
                if bt_windows:
                    # 各ウィンドウについて、1秒前にシークして逐次読み込みで全フレームを取得
                    # → cap.set(POS_MSEC)のキーフレーム精度問題を回避
                    frames_to_backtrack = []  # (actual_msec, gray, frozenset of targets)
                    cap_bt = cv2.VideoCapture(video_path)
                    
                    for (start_msec, end_msec), targets in sorted(bt_windows.items()):
                        # 1秒前にシークしてから逐次読み込み（確実にキーフレームを跨ぐ）
                        cap_bt.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_msec - 1000.0))
                        
                        while True:
                            pre_read_msec = cap_bt.get(cv2.CAP_PROP_POS_MSEC)
                            ret, frame = cap_bt.read()
                            if not ret:
                                break
                            # このフレームの実際のタイムスタンプ（read前の位置がそのフレームのmsec）
                            actual_msec = pre_read_msec
                            if actual_msec > end_msec + 100:
                                break
                            if actual_msec >= start_msec - 50:
                                h, w, _ = frame.shape
                                crop_x_start = int(w * 0.35) + 76
                                crop_x_end = int(w * 0.55) - 10
                                crop_y_start = int(h * 0.35)
                                crop_y_end = int(h * 0.90)
                                cropped_img = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                                cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                                frames_to_backtrack.append((actual_msec, cropped_gray, frozenset(targets)))
                    cap_bt.release()
                    
                    print(f"[INFO] Starting parallel backtracking scan on {len(frames_to_backtrack)} frames...")
                    
                    def process_bt_frame(task):
                        t_msec, gray, tgt = task
                        rows = self.parser.parse_image(gray)
                        return t_msec, rows, tgt
                    
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        bt_results = executor.map(process_bt_frame, frames_to_backtrack)
                        
                    for t_msec, rows, targets in bt_results:
                        for r in rows:
                            if r['rank'] in targets:
                                G = r['rank']
                                if G not in aggregated_data:
                                    aggregated_data[G] = []
                                aggregated_data[G].append(r['score'])
                                print(f"  [FOUND] Rank {G} (Score: {r['score']}) in backtracking at {t_msec:.0f}ms!")

        
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
        # Map index of df_save to the 'rank' values from df properly
        df_save = df.set_index('rank')[['score']]
        if not df_save.empty:
            min_r = df_save.index.min()
            max_r = df_save.index.max()
            full_index = pd.RangeIndex(start=min_r, stop=max_r + 1, name='rank')
            df_save = df_save.reindex(full_index)
            df_save['score'] = df_save['score'].astype(pd.Int32Dtype())
        
        os.makedirs(self.data_dir, exist_ok=True)
        save_path = os.path.join(self.data_dir, f"rank_data_{event_id}.parquet")
        df_save.to_parquet(save_path, compression='zstd')
        print(f"[SUCCESS] Video OCR processing complete. Saved to {save_path} (N={len(df_save)})")
        
        return df_save

def parse_video_filename(video_path):
    """
    Parse video filename based on pattern: [TorG][Season]_[Date]_[TimeOrLast]
    Returns standard event_id if matched, otherwise None.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    pattern = r"^(?P<type>[TG])(?P<season>\d+)_(?:(?P<date>\d{8})_(?P<time>\d{4})|(?P<last>last))$"
    match = re.match(pattern, basename)
    if not match:
        return None
        
    gd = match.groupdict()
    type_str = "total_assault" if gd["type"] == "T" else "grand_assault"
    season = gd["season"]
    
    if gd["last"]:
        event_id = f"{type_str}_{season}_last"
    else:
        event_id = f"{type_str}_{season}_{gd['date']}_{gd['time']}"
        
    return event_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ranking data from Blue Archive scroll videos.")
    parser.add_argument("--video", required=False, help="Path to the video file (filename only or full path inside video folder)")
    parser.add_argument("--event", required=False, help="Event ID (e.g., R43). If omitted, auto-detected from video filename.")
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds")
    parser.add_argument("--outdir", default="rank_data", help="Output directory for the parquet file")
    
    args = parser.parse_args()
    
    # OCR/video ディレクトリの設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(current_dir, "video")
    
    # video ディレクトリ内の動画ファイルをスキャン
    all_videos = []
    if os.path.exists(video_dir):
        all_videos = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
        
    # 正しい命名ルールにマッチする動画を抽出
    valid_videos = []
    for v in all_videos:
        full_path = os.path.join(video_dir, v)
        ev_id = parse_video_filename(full_path)
        if ev_id:
            valid_videos.append((v, full_path, ev_id))
            
    # 正しい動画が1つもない場合はエラー終了
    if not valid_videos:
        print("エラー: 動画ファイル名が条件を満たさないタイトルになっているため、処理を中断します。正しい形式（例: T/Gシーズン[総力戦/大決戦]_年月日_日時 または last[最終結果]）に修正してください。", file=sys.stderr)
        sys.exit(1)
        
    # ビデオパスの決定
    target_video_path = None
    target_event_id = args.event
    
    if args.video:
        # 直接指定された場合
        # 指定された動画が video_dir 内のファイル名か、あるいはそこへのパスか検証
        video_name = os.path.basename(args.video)
        # 拡張子なしで指定された場合の補完
        if not video_name.lower().endswith(".mp4"):
            video_name += ".mp4"
            
        matched_video = None
        for name, full_path, ev_id in valid_videos:
            if name.lower() == video_name.lower():
                matched_video = (name, full_path, ev_id)
                break
                
        if not matched_video:
            # 命名ルール違反、または video フォルダ内に存在しない
            print("エラー: 動画ファイル名が条件を満たさないタイトルになっているため、処理を中断します。正しい形式（例: T/Gシーズン[総力戦/大決戦]_年月日_日時 または last[最終結果]）に修正してください。", file=sys.stderr)
            sys.exit(1)
            
        target_video_path = matched_video[1]
        if not target_event_id:
            target_event_id = matched_video[2]
    else:
        # 指定がない場合、インタラクティブ選択
        print("[INFO] OCR/video フォルダ内に以下の正しい形式の動画が見つかりました:")
        for idx, (name, _, _) in enumerate(valid_videos, 1):
            print(f"{idx}: {name}")
        print("")
        
        while True:
            try:
                choice = input(f"OCRにかける動画の番号を入力してください (1-{len(valid_videos)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(valid_videos):
                    selected = valid_videos[choice_idx]
                    break
                else:
                    print(f"1 から {len(valid_videos)} の範囲で入力してください。")
            except ValueError:
                print("有効な数値を入力してください。")
                
        target_video_path = selected[1]
        if not target_event_id:
            target_event_id = selected[2]
            
    print(f"[INFO] Selected Video: {target_video_path}")
    print(f"[INFO] Target Event ID: {target_event_id}")
    
    video_ocr = VideoOCRParser(data_dir=args.outdir)
    video_ocr.process_and_save(target_video_path, target_event_id, sample_interval_sec=args.interval)
