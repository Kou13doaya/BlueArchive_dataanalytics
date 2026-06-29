# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np

# ----------------------------------------------------
# 定数・辞書データ
# ----------------------------------------------------
EVENT_META = {
    "total_assault_43": {"season": "S43", "boss": "ケセド", "period": "2022/12/07 ～ 2022/12/14"},
    "total_assault_44": {"season": "S44", "boss": "ペロロジラ", "period": "2022/12/21 ～ 2022/12/28"},
    "total_assault_80": {"season": "S80", "boss": "ホド", "period": "2025/07/30 ～ 2025/08/06"},
    "total_assault_81": {"season": "S81", "boss": "ペロロジラ", "period": "2025/08/27 ～ 2025/09/03"},
    "total_assault_82": {"season": "S82", "boss": "ケセド", "period": "2025/10/01 ～ 2025/10/08"},
    "total_assault_83": {"season": "S83", "boss": "イェソド", "period": "2025/10/29 ～ 2025/11/05"},
    "total_assault_84": {"season": "S84", "boss": "クロカゲ", "period": "2025/11/26 ～ 2025/12/03"},
    "total_assault_85": {"season": "S85", "boss": "ホバークラフト", "period": "2025/12/31 ～ 2026/01/07"},
    "total_assault_86": {"season": "S86", "boss": "ビナー", "period": "2026/02/18 ～ 2026/02/25"},
    "total_assault_87": {"season": "S87", "boss": "ゴズ", "period": "2026/03/25 ～ 2026/04/01"},
    "total_assault_88": {"season": "S88", "boss": "KAITEN FX Mk.0", "period": "2026/04/29 ～ 2026/05/06"},
    "total_assault_89": {"season": "S89", "boss": "ドラム缶ガニ", "period": "2026/06/03 ～ 2026/06/10"},
    "grand_assault_25": {"season": "S25", "boss": "ビナー", "period": "2025/08/13 ～ 2025/08/20"},
    "grand_assault_26": {"season": "S26", "boss": "ケセド", "period": "2025/09/10 ～ 2025/09/17"},
    "grand_assault_27": {"season": "S27", "boss": "シロ＆クロ", "period": "2025/10/15 ～ 2025/10/22"},
    "grand_assault_28": {"season": "S28", "boss": "ヒエロニムス", "period": "2025/11/12 ～ 2025/11/19"},
    "grand_assault_29": {"season": "S29", "boss": "KAITEN FX Mk.0", "period": "2025/12/10 ～ 2025/12/17"},
    "grand_assault_30": {"season": "S30", "boss": "シロ＆クロ", "period": "2026/01/14 ～ 2026/01/20"},
    "grand_assault_31": {"season": "S31", "boss": "ヒエロニムス", "period": "2026/03/04 ～ 2026/03/11"},
    "grand_assault_32": {"season": "S32", "boss": "ペロロジラ", "period": "2026/04/08 ～ 2026/04/15"},
    "grand_assault_33": {"season": "S33", "boss": "クロカゲ", "period": "2026/05/13 ～ 2026/05/20"},
    "grand_assault_34": {"season": "S34", "boss": "ホバークラフト", "period": "2026/06/17 ～ 2026/06/24"},
}

diff_translation = {
    "Lunatic": "ルナティック (Lunatic)",
    "Torment": "トーメント (Torment)",
    "Insane": "インセイン (Insane)",
    "Extreme": "エクストリーム (Extreme)",
    "Hardcore": "ハードコア (Hardcore)",
    "VeryHard": "ベリーハード (VeryHard)",
    "Hard": "ハード (Hard)",
    "Normal": "ノーマル (Normal)",
    "Unknown": "不明 (Unknown)"
}

block_translation = {
    "High": "上位ブロック (High)",
    "Mid": "中位ブロック (Mid)",
    "Low": "下位ブロック (Low)"
}

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

def to_legacy_event_id(event_id):
    """
    新しい命名規則のイベントIDを、yuzutrendsのデータ取得などで使用する
    R89 や E34 などの旧イベントID形式（大文字）に変換します。
    """
    if not event_id:
        return ""
    eid = str(event_id).lower().strip()
    if eid.startswith("total_assault_"):
        return f"R{eid[len('total_assault_'):]}".upper()
    if eid.startswith("grand_assault_"):
        return f"E{eid[len('grand_assault_'):]}".upper()
    if eid.startswith("r") or eid.startswith("e"):
        return event_id.upper()
    return event_id

def get_display_name(event_id):
    """
    イベントIDを、S89 (ドラム缶ガニ) [2026/06] のようなわかりやすい表示名に変換します。
    """
    if not event_id:
        return ""
    event_id = normalize_event_id(event_id)
    meta = EVENT_META.get(event_id)
    if meta:
        return f"{meta['season']} ({meta['boss']}) [{meta['period']}]"
    
    # 辞書にない場合の簡易フォールバック
    match = re.match(r"^(?:total_assault_|grand_assault_)(\d+)$", event_id)
    if match:
        return f"S{match.group(1)}"
    
    match_legacy = re.match(r"^([RE])(\d+)$", event_id.upper())
    if match_legacy:
        return f"S{match_legacy.group(2)}"
    return event_id

def score_to_clear_time(score, event_id):
    """
    総力戦スコアから、難易度およびクリアタイム（戦闘時間）を逆算します。
    """
    if not event_id:
        return "Unknown", None
    
    event_id = normalize_event_id(event_id)
    if not event_id.startswith("total_assault_"):
        return "Unknown", None
        
    meta = EVENT_META.get(event_id)
    boss_name = meta["boss"] if meta else ""
    
    limit_type = 4
    if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
        limit_type = 3
    elif boss_name in ["イェソド", "ドラム缶ガニ"]:
        limit_type = 4.5
        
    if limit_type == 3:
        params = {
            "Lunatic": (43235000, 2880),
            "Torment": (31076000, 2400),
            "Insane": (19249600, 1920),
            "Extreme": (9392000, 1440),
            "Hardcore": (3832000, 960),
            "VeryHard": (1916000, 480),
            "Hard": (958000, 240),
            "Normal": (479000, 120),
        }
    elif limit_type == 4:
        params = {
            "Lunatic": (44025000, 2880),
            "Torment": (31708000, 2400),
            "Insane": (21016000, 1920),
            "Extreme": (10160000, 1440),
            "Hardcore": (4216000, 960),
            "VeryHard": (2108000, 480),
            "Hard": (1054000, 240),
            "Normal": (527000, 120),
        }
    else: # 4.5
        params = {
            "Lunatic": (44664000, 2880),
            "Torment": (32502000, 2400),
            "Insane": (21741016, 1920),
            "Extreme": (10578880, 1440),
            "Hardcore": (4437600, 960),
            "VeryHard": (2218800, 480),
            "Hard": (1109400, 240),
            "Normal": (554700, 120),
        }
        
    for diff, (base_score, k) in params.items():
        if score >= base_score:
            time_score = score - base_score
            t_seconds = 3600 - (time_score / k)
            max_limit = limit_type * 60 * 2
            if 0 <= t_seconds <= max_limit:
                return diff, t_seconds
            if t_seconds >= 0:
                return diff, t_seconds
                
    return "Unknown", None

def format_time(t_seconds):
    """
    戦闘秒数を分・秒・ミリ秒にフォーマットします (例: 2分15秒340)。
    """
    if t_seconds is None or t_seconds < 0:
        return "N/A"
    minutes = int(t_seconds // 60)
    seconds = int(t_seconds % 60)
    ms = int(round((t_seconds % 1) * 1000))
    if ms >= 1000:
        seconds += 1
        ms -= 1000
    if seconds >= 60:
        minutes += 1
        seconds -= 60
    return f"{minutes}分{seconds:02d}秒{ms:03d}" if minutes > 0 else f"{seconds}秒{ms:03d}"

def format_time_short(t_seconds):
    """
    戦闘秒数を M:SS.ms にフォーマットします (例: 6:31.767)。
    """
    if t_seconds is None or t_seconds < 0:
        return "N/A"
    minutes = int(t_seconds // 60)
    seconds = int(t_seconds % 60)
    ms = int(round((t_seconds % 1) * 1000))
    if ms >= 1000:
        seconds += 1
        ms -= 1000
    if seconds >= 60:
        minutes += 1
        seconds -= 60
    return f"{minutes}:{seconds:02d}.{ms:03d}"

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

def vectorize_score_to_clear_time(scores, event_id):
    """
    Pandas Series を受け取り、各スコアに対応する難易度とタイムを一括計算します。
    """
    if scores is None or len(scores) == 0 or not event_id:
        return [], []
        
    event_id = normalize_event_id(event_id)
    meta = EVENT_META.get(event_id)
    boss_name = meta["boss"] if meta else ""
    limit_type = 4.0
    if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
        limit_type = 3.0
    elif boss_name in ["イェソド", "ドラム缶ガニ"]:
        limit_type = 4.5
        
    if limit_type == 3.0:
        params = {
            "Lunatic": (43235000, 2880), "Torment": (31076000, 2400), "Insane": (19249600, 1920),
            "Extreme": (9392000, 1440), "Hardcore": (3832000, 960), "VeryHard": (1916000, 480), "Hard": (958000, 240), "Normal": (479000, 120)
        }
    elif limit_type == 4.0:
        params = {
            "Lunatic": (44025000, 2880), "Torment": (31708000, 2400), "Insane": (21016000, 1920),
            "Extreme": (10160000, 1440), "Hardcore": (4216000, 960), "VeryHard": (2108000, 480), "Hard": (1054000, 240), "Normal": (527000, 120)
        }
    else:
        params = {
            "Lunatic": (44664000, 2880), "Torment": (32502000, 2400), "Insane": (21741016, 1920),
            "Extreme": (10578880, 1440), "Hardcore": (4437600, 960), "VeryHard": (2218800, 480), "Hard": (1109400, 240), "Normal": (554700, 120)
        }
        
    diff_list = ["Unknown"] * len(scores)
    time_list = ["N/A"] * len(scores)
    
    scores_np = scores.to_numpy()
    sorted_diffs = sorted(params.keys(), key=lambda d: params[d][0], reverse=True)
    
    for diff in sorted_diffs:
        base_score, k = params[diff]
        for i, val in enumerate(scores_np):
            if diff_list[i] == "Unknown" and val >= base_score:
                time_score = val - base_score
                t_sec = 3600 - (time_score / k)
                
                minutes = int(t_sec // 60)
                seconds = int(t_sec % 60)
                ms = int(round((t_sec % 1) * 1000))
                if ms >= 1000:
                    seconds += 1
                    ms -= 1000
                if seconds >= 60:
                    minutes += 1
                    seconds -= 60
                    
                diff_list[i] = diff
                time_list[i] = f"{minutes}:{seconds:02d}.{ms:03d}"
                
    return diff_list, time_list
