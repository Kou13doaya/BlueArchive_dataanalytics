# -*- coding: utf-8 -*-
import re

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
    "total_assault_99": {"season": "S99", "boss": "Null（beta）", "period": "2026/06/29 ～ 2026/07/06"},
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

def normalize_event_id(event_id):
    """
    R89 や e34 などの様々な形式のIDを、小文字 of total_assault_89 / grand_assault_34 に正規化します。
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
    
    match = re.match(r"^(?:total_assault_|grand_assault_)(\d+)$", event_id)
    if match:
        return f"S{match.group(1)}"
    
    match_legacy = re.match(r"^([RE])(\d+)$", event_id.upper())
    if match_legacy:
        return f"S{match_legacy.group(2)}"
    return event_id
