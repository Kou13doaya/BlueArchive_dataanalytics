# -*- coding: utf-8 -*-
"""
data_loader.py

データ分析ツール全体のデータ読み込みを担当する共通インターフェースです。
ローカルのキャッシュ（Parquet形式）からの読み込みを優先し、
キャッシュがない場合のフォールバックとして非推奨のレガシーソースから読み込みを行います。
"""

import os
import warnings
import pandas as pd
from common.event_metadata import normalize_event_id, to_legacy_event_id

class DataLoader:
    def __init__(self, data_dir="rank_data"):
        self.data_dir = data_dir
        
        # 保存先ディレクトリの作成
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_data(self, event_id, suffix=None):
        """
        指定された event_id のランキングデータをロードします。
        1. ローカルキャッシュ (rank_data_{event_id}_{suffix}.parquet) を最優先でロード (推奨)
        2. ローカルキャッシュがない場合、エラーを出力
        """
        event_id = normalize_event_id(event_id)
        legacy_id = to_legacy_event_id(event_id)
        
        suffix_part = f"_{suffix}" if suffix else ""

        # 1. ローカルのParquetキャッシュの確認
        cache_paths = [
            os.path.join(self.data_dir, f"rank_data_{event_id}{suffix_part}.parquet"),
            f"rank_data_{event_id}{suffix_part}.parquet"
        ]

        for path in cache_paths:
            if os.path.exists(path):
                print(f"[INFO] キャッシュファイル {path} からデータを読み込みます。")
                return pd.read_parquet(path)

        # 2. キャッシュがない場合はロード不可とする（Yuzu Trendsフォールバックの廃止）
        print(f"[ERROR] イベント {event_id}{suffix_part} のローカルキャッシュが見つかりません。")
        print("※ Yuzu Trends からのフォールバック機能はセキュリティとコンプライアンスの観点から廃止されました。")
        print("※ rank_data/ ディレクトリにデータを配置してください。")
        return None
