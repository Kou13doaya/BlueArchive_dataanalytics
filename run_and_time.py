import time
import subprocess
import os
import pandas as pd

# キャッシュのクリーンアップ
parquet_path = "OCR/ocr_rank_data_total_assault_99.parquet"
if os.path.exists(parquet_path):
    try:
        os.remove(parquet_path)
        print("[INFO] Cleaned up existing parquet.")
    except Exception as e:
        print(f"[WARNING] Could not delete parquet: {e}")

start = time.time()
res = subprocess.run([
    "python", "video_ocr_parser.py", 
    "--video", "OCR/total_assault_99.mp4", 
    "--event", "total_assault_99", 
    "--interval", "0.1", 
    "--outdir", "OCR"
])
end = time.time()

print(f"\nTOTAL_RUN_TIME (Final Pure 1.0x Equal-Scale Model): {end - start:.2f} seconds")

# パークエファイルをロードして結果を綺麗に出力
if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(f"\n=== DETECTED RANKS AND SCORES (Total: {len(df)}) ===")
    # pandasのインデックスが'rank'なので、df.iterrows()の第1引数(キー)がrankになります
    for rank, row in df.iterrows():
        print(f"Rank {rank}: Score {row['score']:,}")
else:
    print("[ERROR] Parquet output file was not created!")
