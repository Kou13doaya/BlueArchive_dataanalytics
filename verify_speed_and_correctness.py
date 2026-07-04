import time
import subprocess
import os
import pandas as pd

def main():
    # キャッシュデータの削除（純粋な測定のため）
    target_parquet = "OCR/ocr_rank_data_total_assault_99.parquet"
    if os.path.exists(target_parquet):
        try:
            os.remove(target_parquet)
            print("[INFO] Cleaned up existing target parquet.")
        except Exception as e:
            print(f"[WARNING] Could not delete target parquet: {e}")

    # 計測開始
    print("[INFO] Starting video OCR parsing on total_assault_99.mp4 (Entire video)...")
    start_time = time.time()
    
    # 0.1秒間隔で処理を実行
    res = subprocess.run([
        "python", "video_ocr_parser.py",
        "--video", "OCR/total_assault_99.mp4",
        "--event", "total_assault_99",
        "--interval", "0.1",
        "--outdir", "OCR"
    ])
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n[METRIC] Execution Time: {elapsed:.2f} seconds")

    if res.returncode != 0:
        print("[ERROR] video_ocr_parser.py execution failed.")
        return

    if not os.path.exists(target_parquet):
        print(f"[ERROR] Expected output file was not found: {target_parquet}")
        return

    # 正解データと比較
    ref_parquet = "rank_data/rank_data_total_assault_89.parquet"
    if not os.path.exists(ref_parquet):
        print(f"[ERROR] Reference parquet file not found: {ref_parquet}")
        return

    print(f"\n[INFO] Loading reference dataset: {ref_parquet}")
    df_ref = pd.read_parquet(ref_parquet)
    print(f"[INFO] Reference data shape: {df_ref.shape}")
    print(df_ref.head(5))

    print(f"\n[INFO] Loading target dataset: {target_parquet}")
    df_tgt = pd.read_parquet(target_parquet)
    print(f"[INFO] Target data shape: {df_tgt.shape}")
    print(df_tgt.head(5))

    # 同一性の比較
    # total_assault_89.parquetには明示的な'rank'カラムがなく、インデックスが0から始まっていると仮定します。
    # そのため、インデックス + 1 を順位キーとしてマッピングします。
    ref_dict = {i + 1: score for i, score in enumerate(df_ref['score'])}
    tgt_dict = df_tgt['score'].to_dict()

    # 一致するキー(順位)の確認
    common_ranks = set(ref_dict.keys()) & set(tgt_dict.keys())
    print(f"[INFO] Common ranks found: {len(common_ranks)}")

    mismatches = []
    for rank in sorted(list(common_ranks)):
        if ref_dict[rank] != tgt_dict[rank]:
            mismatches.append((rank, ref_dict[rank], tgt_dict[rank]))

    missing_in_tgt = set(ref_dict.keys()) - set(tgt_dict.keys())
    extra_in_tgt = set(tgt_dict.keys()) - set(ref_dict.keys())

    print("\n=== IDENTICALITY VERIFICATION RESULTS ===")
    if not mismatches and not missing_in_tgt and not extra_in_tgt:
        print("[SUCCESS] All detected ranks and scores perfectly match the reference data!")
    else:
        if mismatches:
            print(f"[WARNING] Detected {len(mismatches)} score mismatches:")
            for rank, ref_score, tgt_score in mismatches[:10]:
                print(f"  - Rank {rank}: Ref={ref_score:,}, Tgt={tgt_score:,}")
            if len(mismatches) > 10:
                print("  - ... and more")
        if missing_in_tgt:
            print(f"[WARNING] Missing ranks in target (present in reference): {sorted(list(missing_in_tgt))[:10]}")
        if extra_in_tgt:
            print(f"[WARNING] Extra ranks in target (not present in reference): {sorted(list(extra_in_tgt))[:10]}")

if __name__ == "__main__":
    main()
