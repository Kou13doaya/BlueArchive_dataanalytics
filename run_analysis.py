# -*- coding: utf-8 -*-
import argparse
import os
import sys

from data_loader import DataLoader
import total_assault
import grand_assault
from utils import normalize_event_id

def main():
    parser = argparse.ArgumentParser(description="ブルーアーカイブ 総力戦・大決戦のスコア分布可視化ツール")
    parser.add_argument("--event", "-e", required=True, help="イベントID (例: R89, E34)")
    parser.add_argument("--type", "-t", choices=["total", "grand"], help="イベントタイプ (total: 総力戦, grand: 大決戦)。未指定の場合はeventの頭文字で自動判定します。")
    parser.add_argument("--output", "-o", help="出力画像ファイルパス (指定しない場合は自動生成します)")
    
    # 大決戦用の設定
    parser.add_argument("--mode", "-m", choices=["High", "Mid", "Low"], default="High", help="大決戦の表示ブロック (デフォルト: High)")
    
    # 総力戦の高度な可視化調整用パラメータ
    parser.add_argument("--rank-mode", default="Platinum (Top 23k)", choices=["Custom (All)", "Platinum (Top 23k)", "Gold (Top 125k)"], help="総力戦の足切りモード")
    parser.add_argument("--zones", nargs="+", default=["Lunatic", "Torment"], help="総力戦で可視化する難易度 (例: Lunatic Torment Insane)")
    parser.add_argument("--l-compress", type=int, default=50900000, help="Lunaticの圧縮しきい値")
    parser.add_argument("--l-bin", type=int, default=150000, help="Lunaticのビンサイズ")
    parser.add_argument("--t-compress", type=int, default=39484000, help="Tormentの圧縮しきい値")
    parser.add_argument("--t-bin", type=int, default=1500, help="Tormentのビンサイズ")
    parser.add_argument("--i-compress", type=int, default=27467000, help="Insaneの圧縮しきい値")
    parser.add_argument("--i-bin", type=int, default=1500, help="Insaneのビンサイズ")

    args = parser.parse_args()

    event_id = normalize_event_id(args.event)
    
    # イベントタイプの自動判定
    event_type = args.type
    if not event_type:
        if event_id.startswith("total_assault_"):
            event_type = "total"
        elif event_id.startswith("grand_assault_"):
            event_type = "grand"
        else:
            print(f"[ERROR] イベントタイプを判定できませんでした: {event_id}. --type を指定してください。")
            sys.exit(1)

    print(f"[INFO] イベント: {event_id} (タイプ: {event_type}) のデータを処理しています...")

    # 1. データの読み込み
    loader = DataLoader()
    df = loader.load_data(event_id)
    
    if df is None or df.empty:
        print(f"[ERROR] データの取得に失敗しました: {event_id}")
        sys.exit(1)

    # 2. 出力パスの設定
    output_path = args.output
    if not output_path:
        if event_type == "total":
            output_path = f"output_total_{event_id}.png"
        else:
            output_path = f"output_grand_{event_id}_{args.mode}.png"

    # 3. 各可視化モジュールの実行
    if event_type == "total":
        print("[INFO] 総力戦グラフを描画しています...")
        total_assault.draw_parametric_graph(
            df=df,
            event_id=event_id,
            rank_mode=args.rank_mode,
            selected_zones=args.zones,
            l_compress=args.l_compress,
            l_bin=args.l_bin,
            t_compress=args.t_compress,
            t_bin=args.t_bin,
            i_compress=args.i_compress,
            i_bin=args.i_bin,
            save_path=output_path,
            show=False
        )
    elif event_type == "grand":
        print(f"[INFO] 大決戦グラフを描画しています (ブロック: {args.mode})...")
        grand_assault.draw_grand_assault_graph(
            df=df,
            view_mode=args.mode,
            save_path=output_path,
            show=False
        )

    print(f"[SUCCESS] 完了しました。画像は {output_path} に保存されました。")

if __name__ == "__main__":
    main()
