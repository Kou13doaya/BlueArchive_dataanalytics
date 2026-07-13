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
    
    # 各難易度の一位（境界）の順位を特定
    if 'status' in df.columns:
        st_col = df['status'].fillna('ocr')
    else:
        st_col = pd.Series('ocr', index=df.index)
        
    # 難易度境界（一位）の決定には 'boundary_top' または 'ocr' のみを使用
    valid_df = df[(st_col.isin(['ocr', 'boundary_top'])) & (df['score'].notna())]
    
    first_ranks = {}
    for idx, diff in enumerate(diffs):
        score_thresh = thresholds[diff]
        if idx == 0:
            matching = valid_df[valid_df['score'] >= score_thresh]
        else:
            prev_diff = diffs[idx - 1]
            prev_thresh = thresholds[prev_diff]
            matching = valid_df[(valid_df['score'] >= score_thresh) & (valid_df['score'] < prev_thresh)]
            
        if not matching.empty:
            first_ranks[diff] = matching.index.min()
            
    summary_data = []
    
    # 総参加者数は単純に df の最大インデックス（最後の順位）を使用
    total_limit = df.index.max()
    
    for idx, diff in enumerate(diffs):
        this_first = first_ranks.get(diff, None)
        
        if this_first is None:
            single_count = 0
            cum_count = 0
        else:
            # 次の難易度の一位順位を探す
            next_first = None
            for next_diff in diffs[idx + 1:]:
                if next_diff in first_ranks:
                    next_first = first_ranks[next_diff]
                    break
            
            if next_first is not None:
                cum_count = next_first - 1
            else:
                cum_count = total_limit
                
            single_count = max(0, cum_count - this_first + 1)
            
        summary_data.append({
            "難易度": diff,
            "クリア人数 (単体)": f"{single_count:,} 人" if single_count > 0 else "0 人",
            "クリア人数 (累積)": f"{cum_count:,} 人" if cum_count > 0 else "0 人"
        })
    return pd.DataFrame(summary_data)

def make_grand_assault_summary(df, event_id=None):
    """
    大決戦用の数値簡易表データフレームを作成します (難易度組み合わせ基準)。
    """
    if df is None or df.empty:
        return pd.DataFrame([{"スコア帯ブロック": "データなし", "プレイヤー数": "0 人"}])

    # ボス名から制限時間を割り出す
    boss_name = "ビナー"
    if event_id:
        event_id = normalize_event_id(event_id)
        meta = EVENT_META.get(event_id)
        if meta:
            boss_name = meta.get("boss", "ビナー")

    limit_sec = 240
    if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
        limit_sec = 180
    elif boss_name in ["イェソド", "ドラム缶ガニ"]:
        limit_sec = 270

    limit_type = limit_sec / 60.0

    # 制限時間ごとの難易度理論値
    theoretical_scores_3m = {
        'L': 53603000,
        'T': 39716000,
        'I': 26161600,
        'E': 14576000,
        'H': 7288000,
        'V': 3644000,
        'A': 1822000,
        'N': 911000
    }
    theoretical_scores_4m = {
        'L': 54393000,
        'T': 40348000,
        'I': 27928000,
        'E': 15344000,
        'H': 7672000,
        'V': 3836000,
        'A': 1918000,
        'N': 959000
    }
    theoretical_scores_4m30s = {
        'L': 55032000,
        'T': 41142000,
        'I': 28653000,
        'E': 15760000,
        'H': 7893600,
        'V': 3946800,
        'A': 1973400,
        'N': 986700
    }

    if limit_type == 3.0:
        theoretical = theoretical_scores_3m
    elif limit_type == 4.5:
        theoretical = theoretical_scores_4m30s
    else:
        theoretical = theoretical_scores_4m

    combinations = ['TTT', 'TTI', 'TII', 'III', 'IIE', 'IEE', 'EEE', 'EEH', 'EHH', 'HHH', 'HHV', 'HVV', 'VVV', 'VVA', 'VAA', 'AAA', 'AAN', 'ANN', 'NNN']

    # 各ブラケットの理論値合計（max_score）を算出
    brackets = []
    for combo in combinations:
        max_score = sum(theoretical.get(char, 0) for char in combo)
        brackets.append({
            'name': combo,
            'max_score': max_score
        })
    brackets.sort(key=lambda x: x['max_score'], reverse=True)

    # 各ブラケットの min_score（下限値）を設定
    # インデックス i の下限値は、一つ下のブラケット i+1 の最大値
    for i in range(len(brackets) - 1):
        brackets[i]['min_score'] = brackets[i+1]['max_score']
    brackets[-1]['min_score'] = 0

    def get_bracket(score):
        if pd.isna(score):
            return 'Other'
        for i in range(1, len(brackets)):
            if score >= brackets[i]['min_score']:
                return brackets[i - 1]['name']
        if score >= 0:
            return brackets[-1]['name']
        return 'Other'

    difficulty_map = {
        'L': 'Lunatic',
        'T': 'Torment',
        'I': 'Insane',
        'E': 'Extreme',
        'H': 'Hardcore',
        'V': 'VeryHard',
        'A': 'Hard',
        'N': 'Normal'
    }

    def format_bracket_name(name):
        if name == 'Other':
            return 'Other'
        return '・'.join(difficulty_map.get(char, char) for char in name)

    counts = df['score'].apply(get_bracket).value_counts()

    summary_data = []
    for b in brackets:
        b_name = b['name']
        count = counts.get(b_name, 0)
        if count > 0:
            summary_data.append({
                "スコア帯ブロック": format_bracket_name(b_name),
                "プレイヤー数": f"{count:,} 人"
            })

    other_count = counts.get('Other', 0)
    if other_count > 0:
        summary_data.append({
            "スコア帯ブロック": "Other",
            "プレイヤー数": f"{other_count:,} 人"
        })

    if not summary_data:
        return pd.DataFrame([{"スコア帯ブロック": "データなし", "プレイヤー数": "0 人"}])

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
    target_ranks = [1, 100, 500] + list(range(1000, 20001, 1000)) + [120000, 240000]
    data = []
    
    event_id = normalize_event_id(event_id)
    is_total_assault = event_id.startswith("total_assault_")
    
    for rank in target_ranks:
        # 指定順位がデータフレームのインデックスに存在するかチェック
        if rank in df.index:
            row = df.loc[rank]
            score = row['score']
            st = row.get('status') if 'status' in df.columns else 'ocr'
            st_str = str(st) if (not pd.isna(st) and st is not None) else 'ocr'
            
            if st_str in ['ocr', 'boundary_border'] and not pd.isna(score):
                row_data = {
                    "順位": f"{rank:,} 位",
                    "スコア": f"{int(score):,}"
                }
                if is_total_assault:
                    diff, t_sec = score_to_clear_time(score, event_id)
                    row_data["クリア難易度"] = diff
                    row_data["クリアタイム"] = format_time_short(t_sec)
            else:
                row_data = {
                    "順位": f"{rank:,} 位",
                    "スコア": "欠損"
                }
                if is_total_assault:
                    row_data["クリア難易度"] = "不明"
                    row_data["クリアタイム"] = "不明"
        else:
            row_data = {
                "順位": f"{rank:,} 位",
                "スコア": "欠損"
            }
            if is_total_assault:
                row_data["クリア難易度"] = "不明"
                row_data["クリアタイム"] = "不明"
        data.append(row_data)
    return pd.DataFrame(data)

def find_nearest_player(df, target_score):
    """
    指定されたスコアに最も近いスコアを持つプレイヤーの順位と実際のスコアを返します。
    """
    if df is None or df.empty:
        return 0, 0
    idx = (df['score'] - target_score).abs().idxmin()
    actual_score = df.iloc[idx]['score']
    if 'rank' in df.columns:
        actual_rank = df.iloc[idx]['rank']
    else:
        actual_rank = idx + 1
    return actual_rank, actual_score


