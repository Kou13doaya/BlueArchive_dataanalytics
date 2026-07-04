# -*- coding: utf-8 -*-
import numpy as np
from common.event_metadata import EVENT_META, normalize_event_id

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
