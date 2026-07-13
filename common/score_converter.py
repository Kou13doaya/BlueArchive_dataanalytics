# -*- coding: utf-8 -*-
import numpy as np
from common.event_metadata import EVENT_META, normalize_event_id

def grand_assault_score_to_clear_time(score, event_id):
    """
    大決戦スコアから、難易度組み合わせおよび合計クリアタイム（戦闘時間）を逆算します。
    """
    event_id = normalize_event_id(event_id)
    meta = EVENT_META.get(event_id)
    boss_name = meta["boss"] if meta else "ビナー"
    
    limit_sec = 240
    if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
        limit_sec = 180
    elif boss_name in ["イェソド", "ドラム缶ガニ"]:
        limit_sec = 270
    limit_type = limit_sec / 60.0

    # 制限時間ごとの難易度理論値とk値 (a, b)
    diff_params = {
        3.0: {
            'L': (53603000, 2880), 'T': (39716000, 2400), 'I': (26161600, 1920),
            'E': (14576000, 1440), 'H': (7288000, 960), 'V': (3644000, 480),
            'A': (1822000, 240), 'N': (911000, 120)
        },
        4.5: {
            'L': (55032000, 2880), 'T': (41142000, 2400), 'I': (28653000, 1920),
            'E': (15760000, 1440), 'H': (7893600, 960), 'V': (3946800, 480),
            'A': (1973400, 240), 'N': (986700, 120)
        },
        4.0: {
            'L': (54393000, 2880), 'T': (40348000, 2400), 'I': (27928000, 1920),
            'E': (15344000, 1440), 'H': (7672000, 960), 'V': (3836000, 480),
            'A': (1918000, 240), 'N': (959000, 120)
        }
    }
    params = diff_params.get(limit_type, diff_params[4.0])

    combinations = ['TTT', 'TTI', 'TII', 'III', 'IIE', 'IEE', 'EEE', 'EEH', 'EHH', 'HHH', 'HHV', 'HVV', 'VVV', 'VVA', 'VAA', 'AAA', 'AAN', 'ANN', 'NNN']
    brackets = []
    for combo in combinations:
        max_score = sum(params[char][0] for char in combo)
        mean_k = sum(params[char][1] for char in combo) / 3.0
        brackets.append({
            'name': combo,
            'max_score': max_score,
            'mean_k': mean_k
        })
    brackets.sort(key=lambda x: x['max_score'], reverse=True)

    selected_bracket = 'Other'
    selected_bracket_info = None
    for i in range(len(brackets)):
        if i < len(brackets) - 1:
            if score > brackets[i+1]['max_score']:
                selected_bracket = brackets[i]['name']
                selected_bracket_info = brackets[i]
                break
        else:
            if score >= 0:
                selected_bracket = brackets[i]['name']
                selected_bracket_info = brackets[i]
                break

    if selected_bracket == 'Other' or not selected_bracket_info:
        return "Unknown", None

    max_score = selected_bracket_info['max_score']
    mean_k = selected_bracket_info['mean_k']
    
    t_seconds = (max_score - score) / mean_k
    if t_seconds < 0:
        t_seconds = 0.0

    return selected_bracket, t_seconds

def score_to_clear_time(score, event_id):
    """
    スコアから難易度およびクリアタイム（戦闘時間）を逆算します（総力戦/大決戦対応）。
    """
    if not event_id:
        return "Unknown", None
    
    event_id = normalize_event_id(event_id)
    if event_id.startswith("grand_assault_"):
        return grand_assault_score_to_clear_time(score, event_id)
        
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
    if event_id.startswith("grand_assault_"):
        diff_list = ["Unknown"] * len(scores)
        time_list = ["N/A"] * len(scores)
        for i, val in enumerate(scores.to_numpy()):
            diff, t_sec = grand_assault_score_to_clear_time(val, event_id)
            if diff != "Unknown" and t_sec is not None:
                diff_list[i] = diff
                minutes = int(t_sec // 60)
                seconds = int(t_sec % 60)
                ms = int(round((t_sec % 1) * 1000))
                if ms >= 1000:
                    seconds += 1
                    ms -= 1000
                if seconds >= 60:
                    minutes += 1
                    seconds -= 60
                time_list[i] = f"{minutes}:{seconds:02d}.{ms:03d}"
        return diff_list, time_list

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
