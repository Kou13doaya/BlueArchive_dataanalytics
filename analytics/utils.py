# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from common.event_metadata import (
    EVENT_META,
    diff_translation,
    block_translation,
    normalize_event_id,
    to_legacy_event_id,
    get_display_name
)
from common.score_converter import (
    score_to_clear_time,
    format_time,
    format_time_short,
    vectorize_score_to_clear_time
)

# ----------------------------------------------------
# データ集計用のヘルパー関数
# ----------------------------------------------------
def make_total_assault_summary(df, event_id):
    """
    総力戦用の数値簡易表データフレームを作成します (クリア人数のみ)。
    """
    if df is None or df.empty or not event_id:
        return pd.DataFrame()
        
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
            "Lunatic": 43235000,
            "Torment": 31076000,
            "Insane": 19249600,
            "Extreme": 9392000,
            "Hardcore": 3832000,
            "VeryHard": 1916000,
            "Hard": 958000,
            "Normal": 479000
        }
    elif limit_type == 4.0:
        thresholds = {
            "Lunatic": 44025000,
            "Torment": 31708000,
            "Insane": 21016000,
            "Extreme": 10160000,
            "Hardcore": 4216000,
            "VeryHard": 2108000,
            "Hard": 1054000,
            "Normal": 527000
        }
    else:
        thresholds = {
            "Lunatic": 44664000,
            "Torment": 32502000,
            "Insane": 21741016,
            "Extreme": 10578880,
            "Hardcore": 4437600,
            "VeryHard": 2218800,
            "Hard": 1109400,
            "Normal": 554700
        }
        
    diffs = ["Lunatic", "Torment", "Insane", "Extreme", "Hardcore", "VeryHard", "Hard", "Normal"]
    summary_data = []
    
    for idx, diff in enumerate(diffs):
        score_thresh = thresholds[diff]
        cum_count = len(df[df['score'] >= score_thresh])
        
        if idx == 0:
            single_count = cum_count
        else:
            prev_diff = diffs[idx - 1]
            prev_thresh = thresholds[prev_diff]
            single_count = len(df[(df['score'] >= score_thresh) & (df['score'] < prev_thresh)])
            
        summary_data.append({
            "難易度": diff,
            "クリア人数 (単体)": f"{single_count:,} 人" if single_count > 0 else "0 人",
            "クリア人数 (累積)": f"{cum_count:,} 人" if cum_count > 0 else "0 人"
        })
    return pd.DataFrame(summary_data)

def make_grand_assault_summary(df):
    """
    大決戦用の数値簡易表データフレームを作成します (クリア人数のみ)。
    """
    high_count = len(df[df['score'] >= 73740000])
    mid_count = len(df[(df['score'] >= 41336000) & (df['score'] < 73740000)])
    low_count = len(df[df['score'] < 41336000])
    
    summary_data = [
        {"スコア帯ブロック": "High (TTT ~ TII)", "プレイヤー数": f"{high_count:,} 人"},
        {"スコア帯ブロック": "Mid (III ~ IEE)", "プレイヤー数": f"{mid_count:,} 人"},
        {"スコア帯ブロック": "Low (EEE ~)", "プレイヤー数": f"{low_count:,} 人"}
    ]
    return pd.DataFrame(summary_data)

def translate_diff(diff_name):
    return diff_translation.get(diff_name, diff_name)

def translate_block(block_name):
    return block_translation.get(block_name, block_name)


def normalize_event_id(event_id):
    """
    R89 や e34 などの様々な形式のIDを、小文字の total_assault_89 / grand_assault_34 に正規化します。
    """
    if not event_id:
        return ""
    eid = str(event_id).lower().strip()
    if eid.startswith("total_assault_") or eid.startswith("grand_assault_"):
        return eid
    if eid.startswith("r"):
        return f"total_assault_{eid[1:]}"
    if eid.startswith("e"):
        return f"grand_assault_{eid[1:]}"
    return eid



def get_rank_scores(df, event_id):
    """
    キリの良い主要な順位のスコア一覧を取得します。総力戦の場合はクリアタイム等も追加します。
    """
    sorted_df = df.sort_values('score', ascending=False).reset_index(drop=True)
    target_ranks = [1, 1000, 5000, 10000, 15000, 20000, 120000, 240000]
    data = []
    
    event_id = normalize_event_id(event_id)
    is_total_assault = event_id.startswith("total_assault_")
    
    for rank in target_ranks:
        if rank <= len(sorted_df):
            score = sorted_df.iloc[rank - 1]['score']
            row = {
                "順位": f"{rank:,} 位",
                "スコア": f"{int(score):,}"
            }
            if is_total_assault:
                diff, t_sec = score_to_clear_time(score, event_id)
                row["クリア難易度"] = diff
                row["クリアタイム"] = format_time(t_sec)
            data.append(row)
    return pd.DataFrame(data)

def find_nearest_player(df, target_score):
    """
    指定されたスコアに最も近いスコアを持つプレイヤーの順位と実際のスコアを返します。
    """
    if df is None or df.empty:
        return 0, 0
    idx = (df['score'] - target_score).abs().idxmin()
    actual_score = df.iloc[idx]['score']
    actual_rank = idx + 1
    return actual_rank, actual_score


