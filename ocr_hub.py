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
from common.event_metadata import EVENT_META, normalize_event_id


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


def input_boundary_ranks_flow():
    """
    ユーザーが各難易度の一位（順位・スコア）およびNormalの最下位（総参加者数）、
    主要ボーダー（20,000位、120,000位、240,000位）を入力し、
    間の区間を 'missing_interval' として埋めてParquetファイルを作成・更新する。
    """
    print("\n" + "="*50)
    print(" 3. 難易度境界順位からのデータ生成・補完")
    print("="*50)

    data_dir = os.path.join(project_root, "rank_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # ① 対象 Parquet ファイルの選択または新規作成
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".bak")])
    print("編集または新規作成するデータを選択してください:")
    print("0: [新規作成] 新しいシーズン・時間で作成")
    for idx, f in enumerate(parquet_files, 1):
        print(f"{idx}: {f}")

    while True:
        choice = input(f"番号を選択してください (0-{len(parquet_files)}, 終了は Enter): ").strip()
        if not choice:
            return
        try:
            choice_idx = int(choice)
            if 0 <= choice_idx <= len(parquet_files):
                break
            else:
                print(f"0 から {len(parquet_files)} の範囲で入力してください。")
        except ValueError:
            print("有効な数値を入力してください。")

    if choice_idx == 0:
        # 新規作成の場合、イベントID（および時間など）を入力
        print("\n[新規作成] イベントIDを入力してください (例: total_assault_90_last, grand_assault_34_20260707_0120)")
        while True:
            new_event_id = input("イベントID: ").strip()
            if new_event_id:
                break
            print("イベントIDは必須です。")
        check_and_register_season(new_event_id)
        file_path = os.path.join(data_dir, f"rank_data_{new_event_id}.parquet")
        df = pd.DataFrame(columns=['score', 'status'])
        df.index.name = 'rank'
    else:
        selected_file = parquet_files[choice_idx - 1]
        file_path = os.path.join(data_dir, selected_file)
        df = pd.read_parquet(file_path)
        if 'status' not in df.columns:
            df['status'] = pd.Series(dtype='string')
        # status列の型を文字列にする
        df['status'] = df['status'].astype(pd.StringDtype())
        if 'score' in df.columns:
            df['score'] = df['score'].astype(pd.Int32Dtype())

    # ② 難易度・ボーダーの登録画面ループ
    # 難易度リスト
    diffs = ["Lunatic", "Torment", "Insane", "Extreme", "Hardcore", "VeryHard", "Hard", "Normal"]
    
    # ユーザーが入力した情報を一時保存する辞書
    # キー: rank, 値: (score, status_label)
    temp_entries = {}
    
    # 既存データの `boundary` と `ocr` 状態のものを引き継ぐ
    # status列がない、あるいは欠損している場合は、scoreが存在すれば 'ocr' 扱いとする
    for r, row in df.iterrows():
        sc = row.get('score', None)
        if not pd.isna(sc):
            st = row.get('status', None)
            st_str = str(st) if (not pd.isna(st) and st is not None) else 'ocr'
            if st_str in ['ocr', 'boundary']:
                temp_entries[int(r)] = (int(sc), st_str)

    # 既存データ（OCR等）から難易度の一位およびボーダーのスコアを自動検出する処理
    # イベントIDの抽出
    event_id = re.sub(r'^rank_data_', '', re.sub(r'\.parquet$', '', selected_file)) if choice_idx != 0 else ""
    event_id = normalize_event_id(event_id)
    meta = EVENT_META.get(event_id)
    boss_name = meta["boss"] if meta else "ビナー"
    
    limit_sec = 240
    if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
        limit_sec = 180
    elif boss_name in ["イェソド", "ドラム缶ガニ"]:
        limit_sec = 270
    limit_type = limit_sec / 60.0
    
    if limit_type == 3.0:
        thresholds = {
            "Lunatic": 43235000, "Torment": 31076000, "Insane": 19249600, "Extreme": 9392000,
            "Hardcore": 3832000, "VeryHard": 1916000, "Hard": 958000, "Normal": 479000
        }
    elif limit_type == 4.0:
        thresholds = {
            "Lunatic": 44025000, "Torment": 31708000, "Insane": 21016000, "Extreme": 10160000,
            "Hardcore": 4216000, "VeryHard": 2108000, "Hard": 1054000, "Normal": 527000
        }
    else:
        thresholds = {
            "Lunatic": 44664000, "Torment": 32502000, "Insane": 21741016, "Extreme": 10578880,
            "Hardcore": 4437600, "VeryHard": 2218800, "Hard": 1109400, "Normal": 554700
        }

    while True:
        print("\n--- 現在の各難易度のTop順位登録状態 ---")
        
        def get_menu_line(choice_num, diff_name=None, border_rank=None, is_normal_bottom=False):
            # 1. ボーダー (20k, 120k, 240k)
            if border_rank:
                border_names = {20000: "チナトロボーダー", 120000: "ゴルドロボーダー", 240000: "シルトロボーダー"}
                b_name = border_names.get(border_rank, f"{border_rank}位ボーダー")
                if border_rank in temp_entries:
                    val, st = temp_entries[border_rank]
                    return f"{choice_num}: {b_name} {border_rank:,}位 - {val:,}点 (登録済み: [{st}])"
                return f"{choice_num}: {b_name} {border_rank:,}位 (未登録)"
                
            # 2. 総参加者数
            if is_normal_bottom:
                candidates = [r for r in temp_entries.keys() if r not in [20000, 120000, 240000]]
                if candidates:
                    max_r = max(candidates)
                    val, st = temp_entries[max_r]
                    return f"{choice_num}: 総参加者数 {max_r:,}位 - {val:,}点 (登録済み: [{st}])"
                return f"{choice_num}: 総参加者数 (未登録)"

            # 3. 難易度一位
            if diff_name:
                thresh = thresholds.get(diff_name, 0)
                valid_ranks = []
                for r, (sc, st) in temp_entries.items():
                    if sc >= thresh:
                        diff_idx = diffs.index(diff_name)
                        if diff_idx == 0:
                            valid_ranks.append((r, sc, st))
                        else:
                            prev_diff = diffs[diff_idx - 1]
                            prev_thresh = thresholds.get(prev_diff, 999999999)
                            if sc < prev_thresh:
                                valid_ranks.append((r, sc, st))

                if valid_ranks:
                    valid_ranks.sort(key=lambda x: x[0])
                    best_rank, best_score, st = valid_ranks[0]
                    status_label = "検出済み" if st == 'ocr' else "登録済み"
                    return f"{choice_num}: {diff_name} {best_rank:,}位 - {best_score:,}点 ({status_label}: [{st}])"
                return f"{choice_num}: {diff_name} 一位 (未登録)"
            return ""

        # 選択メニューと登録状態の表示
        print(get_menu_line(1, diff_name="Lunatic"))
        print(get_menu_line(2, diff_name="Torment"))
        print(get_menu_line(3, diff_name="Insane"))
        print(get_menu_line(4, diff_name="Extreme"))
        print(get_menu_line(5, diff_name="Hardcore"))
        print(get_menu_line(6, diff_name="VeryHard"))
        print(get_menu_line(7, diff_name="Hard"))
        print(get_menu_line(8, diff_name="Normal"))
        print(get_menu_line(9, is_normal_bottom=True))
        print(get_menu_line(10, border_rank=20000))
        print(get_menu_line(11, border_rank=120000))
        print(get_menu_line(12, border_rank=240000))
        print("13: 確定してデータ生成・保存へ進む")
        print("14: 中断してメニューに戻る")

        menu_choice = input("項目を選択してください (1-14): ").strip()
        if menu_choice == '14':
            print("[INFO] 中断しました。")
            return
        elif menu_choice == '13':
            break

        # 入力処理
        target_rank = None
        target_name = ""
        is_fixed_rank = False

        if menu_choice in [str(i) for i in range(1, 9)]:
            d_idx = int(menu_choice) - 1
            target_name = f"{diffs[d_idx]} 一位"
        elif menu_choice == '9':
            target_name = "総参加者数"
        elif menu_choice == '10':
            target_rank = 20000
            target_name = "チナトロボーダー 20,000位"
            is_fixed_rank = True
        elif menu_choice == '11':
            target_rank = 120000
            target_name = "ゴルドロボーダー 120,000位"
            is_fixed_rank = True
        elif menu_choice == '12':
            target_rank = 240000
            target_name = "シルトロボーダー 240,000位"
            is_fixed_rank = True
        else:
            print("[WARNING] 無効な選択です。")
            continue

        print(f"\n--- {target_name} の登録 ---")
        if not is_fixed_rank:
            rank_input = input("順位を入力してください: ").strip()
            if not rank_input:
                print("入力をキャンセルしました。")
                continue
            try:
                target_rank = int(rank_input)
                if target_rank <= 0:
                    print("[ERROR] 順位は1以上である必要があります。")
                    continue
            except ValueError:
                print("[ERROR] 数値を入力してください。")
                continue

        score_input = input("スコアを入力してください: ").strip()
        if not score_input:
            print("入力をキャンセルしました。")
            continue
        try:
            target_score = int(score_input)
            if target_score < 0:
                print("[ERROR] スコアは0以上である必要があります。")
                continue
        except ValueError:
            print("[ERROR] 数値を入力してください。")
            continue

        # 登録 (既存の 'ocr' を上書きするか確認)
        if target_rank in temp_entries and temp_entries[target_rank][1] == 'ocr':
            confirm = input(f"[WARNING] 順位 {target_rank} は既にOCR実測値が存在します。上書きしますか？ (y/n): ").strip().lower()
            if confirm != 'y':
                continue

        temp_entries[target_rank] = (target_score, 'boundary')
        print(f"-> {target_name} (順位: {target_rank}, スコア: {target_score}) を登録しました。")

    # ③ データ生成（補完）処理
    if not temp_entries:
        print("[WARNING] 登録データがないため、保存をスキップしました。")
        return

    # Normal 最下位（総参加者数）が登録されているか確認
    max_rank = max(temp_entries.keys())
    print(f"\n[INFO] データ生成中 (最大順位: {max_rank})...")

    # 1 から max_rank までの RangeIndex
    new_index = pd.RangeIndex(start=1, stop=max_rank + 1, name='rank')
    
    # 既存の DataFrame の ocr データを優先的にマッピング
    final_scores = {}
    final_status = {}

    # 元の DataFrame から ocr データを引き継ぐ
    for r, row in df.iterrows():
        if row.get('status') == 'ocr' and not pd.isna(row.get('score')):
            final_scores[int(r)] = int(row['score'])
            final_status[int(r)] = 'ocr'

    # 手動入力した境界データをマッピング (ocr 優先のため、ocr がない場所のみ上書き)
    for r, (sc, st) in temp_entries.items():
        if final_status.get(r) != 'ocr':
            final_scores[r] = sc
            final_status[r] = st

    # 間の区間を missing_interval で埋める
    df_new = pd.DataFrame(index=new_index)
    df_new['score'] = pd.Series(final_scores).reindex(new_index).astype(pd.Int32Dtype())
    
    # status 列の設定
    status_series = pd.Series(final_status).reindex(new_index)
    # 値がない（NaNの）行は全て 'missing_interval' にする
    status_series = status_series.fillna('missing_interval')
    df_new['status'] = status_series.astype(pd.StringDtype())

    # 保存
    df_new.to_parquet(file_path, compression='zstd')
    print(f"[SUCCESS] 難易度境界データから生成し、保存しました: {os.path.basename(file_path)} (N={len(df_new)})")

    # GitHub反映
    choice = input("\n[PROMPT] このまま続けてGitHubへデータをアップデートしますか？ (y/n): ").strip().lower()
    if choice == 'y':
        push_to_github()


def main_menu():
    while True:
        print("\n" + "="*50)
        print("  ブルアカ データ解析・OCR管理 Hub")
        print("="*50)
        print("1: OCRの実行とデータ作成/補完")
        print("2: 複数OCR結果の統合")
        print("3: 難易度境界順位からのデータ生成・補完")
        print("4: 既存データの欠損補完")
        print("5: GitHubへの数値データの自動アップデート")
        print("6: 可視化グラフの生成")
        print("7: 終了")
        print("="*50)
        
        choice = input("メニュー番号を選択してください (1-7): ").strip()
        if choice == '1':
            run_ocr_pipeline()
        elif choice == '2':
            merge_ocr_results()
        elif choice == '3':
            input_boundary_ranks_flow()
        elif choice == '4':
            patch_existing_data()
        elif choice == '5':
            push_to_github()
        elif choice == '6':
            generate_visualization()
        elif choice == '7':
            print("\nHubツールを終了します。お疲れ様でした！")
            break
        else:
            print("[WARNING] 1から7の数値を入力してください。")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n[INFO] プログラムが中断されました。終了します。")
