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

from OCR.video_ocr_parser import interactive_patch_missing_data, check_and_register_season


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
        print("2: 既存データの欠損補完")
        print("3: GitHubへの数値データの自動アップデート")
        print("4: 可視化グラフの生成")
        print("5: 終了")
        print("="*50)
        
        choice = input("メニュー番号を選択してください (1-5): ").strip()
        if choice == '1':
            run_ocr_pipeline()
        elif choice == '2':
            patch_existing_data()
        elif choice == '3':
            push_to_github()
        elif choice == '4':
            generate_visualization()
        elif choice == '5':
            print("\nHubツールを終了します。お疲れ様でした！")
            break
        else:
            print("[WARNING] 1から5の数値を入力してください。")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n[INFO] プログラムが中断されました。終了します。")
