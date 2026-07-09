# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import re
import pandas as pd

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# OCRフォルダもインポート用にパス追加
ocr_dir = os.path.join(project_root, "OCR")
if ocr_dir not in sys.path:
    sys.path.append(ocr_dir)

from OCR.video_ocr_parser import (
    interactive_patch_missing_data,
    check_and_register_season,
    merge_dataframes,
    parse_video_filename,
)


def run_ocr_pipeline():
    """
    OCR/video_ocr_parser.py を実行します。
    """
    print("\n" + "="*50)
    print(" 1. OCRの実行とデータ作成/補完")
    print("="*50)
    
    # OCRスクリプトを実行
    parser_path = os.path.join(project_root, "OCR", "video_ocr_parser.py")
    try:
        # スレッドや対話型プロンプトが正しく動作するように現在のプロセスと同じ標準入出力で起動
        subprocess.run([sys.executable, parser_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] OCR実行中にエラーが発生しました: {e}")
        return
    except KeyboardInterrupt:
        print("\n[INFO] OCRの実行が中断されました。")
        return

    # OCRと補完が完了した後、GitHubへアップロードするか確認
    choice = input("\n[PROMPT] このまま続けてGitHubへデータをアップデートしますか？ (y/n): ").strip().lower()
    if choice == 'y':
        push_to_github()



def merge_ocr_results():
    """
    既存の Parquet をベースとして、関連する動画を追加 OCR し統合します。
    """
    print("\n" + "="*50)
    print(" 2. 複数OCR結果の統合")
    print("="*50)

    data_dir = os.path.join(project_root, "rank_data")
    video_dir = os.path.join(project_root, "OCR", "video")

    if not os.path.exists(data_dir):
        print(f"[ERROR] データディレクトリが見つかりません: {data_dir}")
        return

    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".bak")])
    if not parquet_files:
        print("[INFO] 統合対象の Parquet ファイルが見つかりません。")
        return

    # ベース Parquet の選択
    print("\n統合先（ベース）の Parquet ファイルを選択してください:")
    for idx, f in enumerate(parquet_files, 1):
        print(f"{idx}: {f}")

    while True:
        choice = input(f"番号を選択してください (1-{len(parquet_files)}, 終了は Enter): ").strip()
        if not choice:
            return
        try:
            base_file = parquet_files[int(choice) - 1]
            break
        except (ValueError, IndexError):
            print(f"1 から {len(parquet_files)} の範囲で入力してください。")

    base_path = os.path.join(data_dir, base_file)
    print(f"\n[INFO] ベースファイル: {base_file}")

    # Parquet 名から event_id を逆算してビデオ候補を絞り込む
    # 例: rank_data_total_assault_90_last.parquet -> total_assault_90_last
    event_id_from_parquet = re.sub(r'^rank_data_', '', re.sub(r'\.parquet$', '', base_file))
    # event_id -> ビデオファイル名プレフィックスに変換
    # total_assault -> T, grand_assault -> G
    vid_prefix = None
    for prefix_key, vid_char in [("total_assault", "T"), ("grand_assault", "G")]:
        if event_id_from_parquet.startswith(prefix_key + "_"):
            rest = event_id_from_parquet[len(prefix_key) + 1:]  # 例: 90_last
            vid_prefix = f"{vid_char}{rest}"  # 例: T90_last
            break

    # 動画フォルダが存在する場合のみ候補を表示
    candidate_videos = []
    if vid_prefix and os.path.exists(video_dir):
        all_videos = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
        for v in all_videos:
            ev_id = parse_video_filename(os.path.join(video_dir, v))
            if ev_id == event_id_from_parquet:
                candidate_videos.append((v, os.path.join(video_dir, v)))
        candidate_videos.sort(key=lambda x: x[0])

    if not candidate_videos:
        print(f"[INFO] OCR/video/ 内に '{vid_prefix}*' に一致する動画ファイルが見つかりませんでした。")
        print("       動画ファイルを OCR/video/ フォルダに配置してから再試行してください。")
        return

    print(f"\n以下の動画が '{base_file}' に統合できる候補として見つかりました:")
    for idx, (name, _) in enumerate(candidate_videos, 1):
        print(f"  {idx}: {name}")

    print("追加でOCRして統合したい動画を選択してください（スペース区切りで複数可）。")
    sel_input = input(f"番号を入力 (1-{len(candidate_videos)}, 終了は Enter): ").strip()
    if not sel_input:
        print("[INFO] キャンセルしました。")
        return

    sel_indices = []
    for tok in sel_input.split():
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(candidate_videos):
                sel_indices.append(idx)
            else:
                print(f"[WARNING] 範囲外の番号をスキップ: {tok}")
        except ValueError:
            print(f"[WARNING] 無効な入力をスキップ: {tok}")

    if not sel_indices:
        print("[INFO] 有効な選択がありませんでした。")
        return

    selected_videos = [candidate_videos[i] for i in sel_indices]

    # ベース Parquet を読み込む
    print(f"\n[INFO] ベースデータを読み込んでいます: {base_file}")
    df_base = pd.read_parquet(base_path)
    if 'score' in df_base.columns:
        df_base['score'] = df_base['score'].astype(pd.Int32Dtype())

    # 各動画を OCR してマージ
    parser_path = os.path.join(project_root, "OCR", "video_ocr_parser.py")
    for name, vid_path in selected_videos:
        print(f"\n{'='*50}")
        print(f"[INFO] OCR 実行中: {name}")
        print(f"{'='*50}")

        # video_ocr_parser.py を subprocess 実行（統合フラグを渡す）
        # ただし process_and_save の既存ファイル上書き挙動を避けるため
        # ここでは VideoOCRParser を直接インポートして利用する
        try:
            # 動的インポート（subprocess 回避）
            import importlib.util
            spec = importlib.util.spec_from_file_location("video_ocr_parser", parser_path)
            vop_module = importlib.util.load_module_from_spec(spec) if hasattr(importlib.util, 'load_module_from_spec') else None

            # 標準 import に切り替え
            from OCR.video_ocr_parser import VideoOCRParser
            parser = VideoOCRParser(data_dir=data_dir)

            rows = parser.parse_video(vid_path, sample_interval_sec=0.1)
            if not rows:
                print(f"[WARNING] {name} からデータが取得できませんでした。スキップします。")
                continue

            cleaned = parser.clean_anomalies(rows)
            validated = parser.parser.validate_scores(cleaned)

            import pandas as _pd
            df_new = _pd.DataFrame(validated).set_index('rank')[['score']]
            if not df_new.empty:
                min_r = df_new.index.min()
                max_r = df_new.index.max()
                df_new = df_new.reindex(_pd.RangeIndex(start=min_r, stop=max_r + 1, name='rank'))
                df_new['score'] = df_new['score'].astype(_pd.Int32Dtype())

            print(f"[INFO] {name} の OCR 完了 ({len(df_new)} 件)。ベースデータと統合します...")
            df_base = merge_dataframes(df_base, df_new)

        except Exception as e:
            print(f"[ERROR] {name} の OCR 中にエラーが発生しました: {e}")
            import traceback; traceback.print_exc()
            continue

    # 欠損補完
    df_base = interactive_patch_missing_data(df_base)

    # 上書き保存
    df_base.to_parquet(base_path, compression='zstd')
    print(f"\n[SUCCESS] 統合完了。{base_file} に上書き保存しました (N={len(df_base)})")

    # GitHub push の確認
    choice = input("\n[PROMPT] このまま続けてGitHubへデータをアップデートしますか？ (y/n): ").strip().lower()
    if choice == 'y':
        push_to_github()



def patch_existing_data():
    """
    既存のParquetファイルを指定して欠損順位のスコアを手動で補完します。
    """
    print("\n" + "="*50)
    print(" 2. 既存データの欠損補完")
    print("="*50)
    
    data_dir = os.path.join(project_root, "rank_data")
    if not os.path.exists(data_dir):
        print(f"[ERROR] データディレクトリが見つかりません: {data_dir}")
        return
        
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    if not files:
        print("[INFO] 補完対象のParquetファイルが見つかりません。")
        return
        
    print("以下のファイルが見つかりました。補完する番号を選択してください:")
    for idx, f in enumerate(files, 1):
        print(f"{idx}: {f}")
        
    while True:
        choice = input(f"選択してください (1-{len(files)}, 終了は Enter): ").strip()
        if not choice:
            return
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(files):
                selected_file = files[choice_idx]
                break
            else:
                print(f"1 から {len(files)} の範囲で入力してください。")
        except ValueError:
            print("有効な数値を入力してください。")
            
    file_path = os.path.join(data_dir, selected_file)
    print(f"\n[INFO] {selected_file} を読み込んでいます...")
    
    try:
        # バックアップ作成
        backup_path = file_path + ".bak"
        import shutil
        shutil.copyfile(file_path, backup_path)
        print(f"[INFO] バックアップを作成しました: {selected_file}.bak")
        
        df = pd.read_parquet(file_path)
        # score列が欠損を含めるよう Float/Int32型にする
        if 'score' in df.columns:
            df['score'] = df['score'].astype(pd.Int32Dtype())
            
        # 対話型で補完
        df_patched = interactive_patch_missing_data(df)
        
        # 保存
        df_patched.to_parquet(file_path, compression='zstd')
        print(f"[SUCCESS] データを更新して保存しました: {selected_file}")
        
        # 補完が完了した後、GitHubへアップロードするか確認
        choice = input("\n[PROMPT] このまま続けてGitHubへデータをアップデートしますか？ (y/n): ").strip().lower()
        if choice == 'y':
            push_to_github()
    except Exception as e:
        print(f"[ERROR] データの処理中にエラーが発生しました: {e}")



def push_to_github():
    """
    gitコマンドを使って数値データをGitHubへ自動アップデートします。
    """
    print("\n" + "="*50)
    print(" 3. GitHubへの数値データの自動アップデート")
    print("="*50)
    
    # gitがインストールされているか確認
    try:
        subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("[ERROR] gitコマンドが使用できません。Gitが正しくインストールされているか確認してください。")
        return
        
    # 現在のステータスを表示
    print("[INFO] git status を確認しています...")
    subprocess.run(["git", "status", "rank_data/"])
    
    choice = input("\n変更されたParquetデータをGitHubへプッシュしますか？ (y/n): ").strip().lower()
    if choice != 'y':
        print("[INFO] アップデートをキャンセルしました。")
        return
        
    try:
        # add
        print("[INFO] ファイルを追加しています (git add)...")
        # rank_data内の変更されたすべてのParquetファイルを追加
        subprocess.run(["git", "add", "rank_data/*.parquet"], check=True)

        # git diff --cached でコミット対象のステージされた変更があるか確認
        check_diff = subprocess.run(["git", "diff", "--cached", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 戻り値が 0 の場合は変更（差分）がないことを意味します
        if check_diff.returncode == 0:
            print("\n[INFO] コミット対象の新しい数値データ（Parquet）の変更はありませんでした。")
            return
            
        # commit
        commit_msg = input("コミットメッセージを入力してください (デフォルト: 'update rank data'): ").strip()
        if not commit_msg:
            commit_msg = "update rank data"
        print("[INFO] コミットしています (git commit)...")
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # push
        print("[INFO] GitHubへプッシュしています (git push)...")
        subprocess.run(["git", "push"], check=True)
        print("[SUCCESS] GitHubへのデータ反映が正常に完了しました！")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Gitコマンド実行中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"[ERROR] 処理中に予期しないエラーが発生しました: {e}")


def generate_visualization():
    """
    run_analysis.py を呼び出してグラフを生成します。
    """
    print("\n" + "="*50)
    print(" 4. 可視化グラフの生成")
    print("="*50)
    
    event_id = input("イベントIDを入力してください (例: R90, E34, total_assault_90): ").strip()
    if not event_id:
        print("[INFO] キャンセルしました。")
        return
        
    analysis_script = os.path.join(project_root, "run_analysis.py")
    try:
        print(f"[INFO] グラフを生成中: {event_id} ...")
        subprocess.run([sys.executable, analysis_script, "-e", event_id], check=True)
        print("[SUCCESS] グラフの生成が完了しました！")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] グラフ生成中にエラーが発生しました: {e}")


def main_menu():
    while True:
        print("\n" + "="*50)
        print("  ブルアカ データ解析・OCR管理 Hub")
        print("="*50)
        print("1: OCRの実行とデータ作成/補完")
        print("2: 複数OCR結果の統合")
        print("3: 既存データの欠損補完")
        print("4: GitHubへの数値データの自動アップデート")
        print("5: 可視化グラフの生成")
        print("6: 終了")
        print("="*50)
        
        choice = input("メニュー番号を選択してください (1-6): ").strip()
        if choice == '1':
            run_ocr_pipeline()
        elif choice == '2':
            merge_ocr_results()
        elif choice == '3':
            patch_existing_data()
        elif choice == '4':
            push_to_github()
        elif choice == '5':
            generate_visualization()
        elif choice == '6':
            print("\nHubツールを終了します。お疲れ様でした！")
            break
        else:
            print("[WARNING] 1から6の数値を入力してください。")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n[INFO] プログラムが中断されました。終了します。")
