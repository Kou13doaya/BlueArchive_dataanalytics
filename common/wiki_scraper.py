# -*- coding: utf-8 -*-
import requests
import re

def scrape_event_info(event_id):
    """
    Scrapes the Blue Archive Wikiru wiki to find the boss and period for the given event_id.
    event_id: total_assault_90, grand_assault_34, R90, E34, etc.
    Returns: (boss_name, period_str) or None
    """
    event_id = event_id.lower().strip()
    is_total = "total_assault" in event_id or event_id.startswith("r")
    
    match_num = re.search(r'\d+', event_id)
    if not match_num:
        return None
    season_num = int(match_num.group(0))
    
    if is_total:
        url = "https://bluearchive.wikiru.jp/?%E7%B7%8F%E5%8A%9B%E6%88%A6"
    else:
        url = "https://bluearchive.wikiru.jp/?%E5%A4%A7%E6%B1%BA%E6%88%A6"
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        html = r.text
    except Exception:
        return None
        
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    
    for row in rows:
        # Pukiwiki table cell structure parsing
        row_clean = re.sub(r'</(td|th)>', '\t', row)
        row_clean = re.sub(r'<[^>]+>', '', row_clean)
        cols = [c.strip() for c in row_clean.split('\t') if c.strip()]
        if len(cols) < 5:
            continue
            
        col0_clean = re.sub(r'\D+', '', cols[0])
        if col0_clean == str(season_num):
            boss = cols[1]
            period = ""
            for col in cols:
                # Look for format: YYYY/MM/DD ～ YYYY/MM/DD
                match_date = re.search(r'\d{4}[/\-]\d{2}[/\-]\d{2}.*?[～~].*?\d{4}[/\-]\d{2}[/\-]\d{2}', col)
                if match_date:
                    period = match_date.group(0)
                    break
            if not period:
                for col in cols:
                    if re.search(r'\d{4}/\d{2}/\d{2}', col):
                        period = col
                        break
            
            # Normalize dates to half-width tilde
            period = re.sub(r'\s*[~～〜]\s*', ' ~ ', period).strip()
            return boss, period
            
    return None

def scrape_grand_assault_defenses(event_id):
    """
    大決戦のシーズン番号から、そのシーズンの3つの防御タイプを取得します。
    event_id: grand_assault_34, G34, etc.
    Returns: ['軽装備', '重装甲', '特殊装甲'] などのリスト、失敗時は None
    """
    event_id = event_id.lower().strip()
    match_num = re.search(r'\d+', event_id)
    if not match_num:
        return None
    season_num = int(match_num.group(0))
    
    url = "https://bluearchive.wikiru.jp/?%E5%A4%A7%E6%B1%BA%E6%88%A6"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        html = r.text
    except Exception:
        return None
        
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    
    for row in rows:
        row_clean = row.replace('&ensp;', ' ').replace('&nbsp;', ' ').replace('&amp;', '&')
        row_clean = re.sub(r'</(td|th)>', '\t', row_clean)
        row_clean = re.sub(r'<[^>]+>', '', row_clean)
        cols = [c.strip() for c in row_clean.split('\t') if c.strip()]
        if len(cols) < 5:
            continue
            
        col0_clean = re.sub(r'\D+', '', cols[0])
        if col0_clean == str(season_num):
            # 5列目(インデックス4)に「軽装備 / 重装甲 / 特殊装甲」の形式で書かれている
            defense_str = cols[4]
            # スラッシュや読点、スペース等で分割
            defenses = [d.strip() for d in re.split(r'[/,、・\s\x20\u3000]+', defense_str) if d.strip()]
            if defenses:
                return defenses
                
    return None

