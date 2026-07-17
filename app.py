# -*- coding: utf-8 -*-
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("styler.render.max_elements", 500000)

from data_loader import DataLoader
from analytics import total_assault
from analytics import grand_assault

# ページ基本設定（ワイドモード、美しいUI）
st.set_page_config(
    page_title="ブルアカ 総力戦・大決戦 スコア分布分析ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ブラウザの誤翻訳（英語判定）を防ぐための言語属性(lang="ja")設定、notranslateおよびtranslate="no"属性の注入
st.markdown("""
    <script>
        (function() {
            function applySettings(doc) {
                if (!doc) return;
                var html = doc.documentElement;
                if (html) {
                    html.lang = 'ja';
                    html.setAttribute('xml:lang', 'ja');
                    html.setAttribute('translate', 'no');
                    html.classList.add('notranslate');
                }
                
                // meta tagの注入
                var head = doc.head || doc.getElementsByTagName('head')[0];
                if (head) {
                    if (!head.querySelector('meta[name="google"]')) {
                        var meta = doc.createElement('meta');
                        meta.name = 'google';
                        meta.content = 'notranslate';
                        head.appendChild(meta);
                    }
                }
            }

            // 1. 自身のドキュメントに適用
            applySettings(window.document);

            // 2. 親のドキュメントに適用（アクセス可能な場合）
            try {
                if (window.parent && window.parent.document) {
                    applySettings(window.parent.document);
                }
            } catch (e) {
                console.warn("Could not access parent document due to same-origin policy:", e);
            }
        })();
    </script>
""", unsafe_allow_html=True)

# スタイリング (ダーク/ブルー系のカスタムCSS)
st.markdown("""
<style>
    /* アプリ最上部・コンテンツ全体の余白を極限まで削り、上部に引き上げる */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    header[data-testid="stHeader"] {
        height: 2.0rem !important;
        background-color: transparent !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 5px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Blue Archive Data Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">総力戦・大決戦データ分析ツール (Preview4)</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# ユーティリティ関数のインポート
# ----------------------------------------------------
from common.event_metadata import EVENT_META, normalize_event_id, get_display_name
from common.score_converter import score_to_clear_time, format_time_short, vectorize_score_to_clear_time
from analytics.utils import (
    make_total_assault_summary,
    make_grand_assault_summary,
    translate_diff,
    translate_block,
    get_rank_scores,
    find_nearest_player
)

# data_loader の初期化
loader = DataLoader()

import base64

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return ""

platinum_base64 = get_base64_image("image/platinum.png")

@st.cache_data
def load_cached_data(event_id, suffix=None):
    """
    メモリ上にロード結果をキャッシュし、不要なディスク読み込みを防ぎます。
    """
    return loader.load_data(event_id, suffix=suffix)

def cached_total_assault_graph(df, event_id, draw_mode, selected_zones_tuple, compress_tuple, bin_tuple):
    return total_assault.draw_parametric_graph(
        df=df,
        event_id=event_id,
        draw_mode=draw_mode,
        selected_zones=list(selected_zones_tuple),
        compress_settings=dict(compress_tuple),
        bin_settings=dict(bin_tuple),
        save_path=None,
        show=False
    )

def cached_grand_assault_graph(df, event_id, suffix, draw_mode, selected_zones_tuple, compress_tuple, bin_tuple):
    return grand_assault.draw_grand_assault_parametric_graph(
        df=df,
        event_id=event_id,
        suffix=suffix,
        draw_mode=draw_mode,
        selected_zones=list(selected_zones_tuple),
        compress_settings=dict(compress_tuple),
        bin_settings=dict(bin_tuple),
        save_path=None,
        show=False
    )

@st.cache_data(show_spinner=False)
def get_portal_card_stats(eid, suffix=None):
    # ポータルカード統計用のデータ読み込み時にも最新のサフィックスを使用する
    df_event = load_cached_data(eid, suffix=suffix)
    total_players = len(df_event) if df_event is not None else 0
    
    plat_score_portal = None
    plat_time_str = ""
    if df_event is not None and not df_event.empty:
        sorted_df_portal = df_event.sort_values('score', ascending=False).reset_index(drop=True)
        plat_score_portal = sorted_df_portal.iloc[19999]['score'] if len(sorted_df_portal) > 19999 else None
        
    is_total = normalize_event_id(eid).startswith("total_assault_")
    if is_total and plat_score_portal is not None:
        diff, t_sec = score_to_clear_time(plat_score_portal, eid)
        t_str = format_time_short(t_sec)
        plat_time_str = f"{diff} {t_str}"
        
    return total_players, plat_score_portal, plat_time_str

# rank_data ディレクトリ内のParquetファイルを自動取得
data_dir = "rank_data"
event_suffix_map = {} # { event_id: [suffix1, suffix2, ...] }
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    for f in files:
        # サフィックス付き形式 (例: rank_data_total_assault_99_last.parquet / rank_data_total_assault_99_20260603_1100.parquet)
        match = re.match(r"rank_data_(total_assault_\d+|grand_assault_\d+)(?:_(.+))?\.parquet", f)
        if match:
            eid = match.group(1)
            suffix = match.group(2) or ""
            
            if eid not in event_suffix_map:
                event_suffix_map[eid] = []
            event_suffix_map[eid].append(suffix)
        else:
            # 移行期間用に旧形式のJSON/Parquetも検知
            match_old = re.match(r"rank_data_(R\d+|E\d+)\.(?:json|parquet)", f)
            if match_old:
                eid = normalize_event_id(match_old.group(1))
                if eid not in event_suffix_map:
                    event_suffix_map[eid] = []
                event_suffix_map[eid].append("")

# 重複を排除しソート。各イベント内のサフィックスは "last" を先頭にし、他は降順（新しい時間順）にする
available_events = sorted(list(event_suffix_map.keys()), reverse=True)

for eid in event_suffix_map:
    suffixes = list(set(event_suffix_map[eid]))
    # "last" があれば最優先、それ以外は降順ソート
    other_suffixes = [s for s in suffixes if s not in ["", "last"]]
    other_suffixes.sort(reverse=True)
    
    sorted_suffixes = []
    if "last" in suffixes:
        sorted_suffixes.append("last")
    if "" in suffixes:
        sorted_suffixes.append("")
    sorted_suffixes.extend(other_suffixes)
    
    event_suffix_map[eid] = sorted_suffixes

# Session State とクエリパラメータの同期
query_params = st.query_params
if "event_id" in query_params:
    st.session_state['selected_event_id'] = query_params["event_id"]

event_id = st.session_state.get('selected_event_id')

if event_id:
    # URLに反映されていなければ書き込む
    if st.query_params.get("event_id") != event_id:
        st.query_params["event_id"] = event_id
        
    app_mode = "総力戦 (Total Assault)" if normalize_event_id(event_id).startswith("total_assault_") else "大決戦 (Grand Assault)"
    
    # 戻るボタンをサイドバー最上部に配置
    if st.sidebar.button("← 一覧に戻る", key="back_to_portal_btn", use_container_width=True):
        st.session_state['selected_event_id'] = None
        st.query_params.clear()
        st.rerun()
    st.sidebar.markdown("---")
    
    # サフィックスの選択UI
    selected_suffix = None
    if event_id in event_suffix_map:
        suffixes = event_suffix_map[event_id]
        def format_suffix(s):
            if s == "last":
                return "最終結果"
            elif s == "":
                return "デフォルト"
            else:
                # 20260603_1100 -> 2026/06/03 11:00 のように整形
                match_dt = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})$", s)
                if match_dt:
                    return f"{match_dt.group(1)}/{match_dt.group(2)}/{match_dt.group(3)} {match_dt.group(4)}:{match_dt.group(5)}"
                return s
        
        if len(suffixes) > 1:
            selected_suffix = st.sidebar.selectbox(
                "データ取得時期",
                suffixes,
                format_func=format_suffix,
                key="data_version_suffix"
            )
        elif len(suffixes) == 1:
            # 選択肢が1つだけの場合はセレクトボックスを無効化（disabled）して表示
            selected_suffix = suffixes[0]
            st.sidebar.selectbox(
                "データ取得時期",
                suffixes,
                index=0,
                format_func=format_suffix,
                disabled=True,
                key="data_version_suffix"
            )
        else:
            selected_suffix = None

    # データ読み込み
    df = None
    with st.spinner("データを取得・解析中..."):
        df = load_cached_data(event_id, suffix=selected_suffix)
else:
    app_mode = None
    df = None

# ----------------------------------------------------
# メイン表示エリア (ポータル画面 または 詳細ダッシュボード)
# ----------------------------------------------------
if not event_id:
    # ====================================================
    # A. ポータル画面: 総力戦・大決戦ポータル
    # ====================================================
    # レスポンシブグリッドと高さ統一のためのカスタムCSS
    st.markdown("""
        <style>
        /* columns のフレックスボックス折り返し設定 */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        div[data-testid="column"] {
            min-width: 215px !important;
            flex: 1 1 215px !important;
            margin-bottom: 16px !important; /* 上下のカード同士に間隔を持たせる */
        }
        
        /* カード全体のアンカーリンクスタイル */
        .portal-card {
            position: relative; /* 絶対配置リンクの基準点 */
            background-color: #1a202c; /* ダーク系のプレミアムな背景 */
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 14px; /* 余白を狭めて引き締める */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 205px; /* 縦幅を少しコンパクトに */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }
        .portal-card:hover {
            transform: translateY(-4px);
            border-color: #3b82f6;
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.2), 0 4px 6px -4px rgba(59, 130, 246, 0.2);
        }
        
        /* カード全体を覆う透明なアンカーリンク */
        .portal-card-link-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 10;
            cursor: pointer;
            background-color: rgba(0, 0, 0, 0); /* 完全透明 */
            border-radius: 8px;
        }
        
        /* カード上部のヘッダー情報 */
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px; /* 余白を狭める */
        }
        .card-badge {
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: bold;
        }
        .card-period {
            color: #94a3b8;
            font-size: 0.72rem;
            text-align: right;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 140px;
        }
        
        /* ボス名とシーズン番号 */
        .card-title-row {
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin: 4px 0 6px 0;
            min-height: 1.8rem;
        }
        .card-season {
            color: #94a3b8;
            font-size: 0.72rem;
            white-space: nowrap;
        }
        .card-boss {
            color: #f8fafc;
            font-size: 1.25rem;
            font-weight: bold;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* チナトロボーダー領域 */
        .card-border-area {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin-bottom: 6px; /* 下マージンを狭める */
        }
        .card-border-img {
            width: 24px; /* 少し小さく */
            height: auto;
            margin-top: 2px;
        }
        .card-border-info {
            display: flex;
            flex-direction: column;
        }
        .card-border-score {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: bold;
            line-height: 1.1;
        }
        .card-border-time {
            color: #cbd5e1;
            font-size: 0.78rem;
            font-weight: bold;
            margin-top: 1px; /* マージンを詰める */
        }
        .card-border-time-placeholder {
            height: 1.0rem; /* 高さを詰める */
        }
        
        /* 参加者数 */
        .card-players {
            color: #94a3b8;
            font-size: 0.78rem;
            margin-top: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 一回り小さくしたタイトル
    st.markdown("<h2 style='font-size: 2.0rem; font-weight: 800; color: #1E88E5; margin-bottom: 5px;'>総力戦・大決戦</h2>", unsafe_allow_html=True)
    # ここの文字は不要
    # st.markdown("<p style='color: #888; font-size: 1.0rem; margin-top: -5px; margin-bottom: 20px;'>過去に開催された総力戦・大決戦の統計ダッシュボード一覧</p>", unsafe_allow_html=True)
    
    # 検索窓とカテゴリ切り替えを横並びに配置
    col_search, col_tabs = st.columns([1, 1])
    with col_search:
        search_query = st.text_input(
            "ボス検索",
            value="",
            placeholder="ボス名またはIDで検索 (例: ビナー, S90)",
            label_visibility="collapsed",
            key="portal_search_input"
        )
    with col_tabs:
        st.write('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        portal_tab = st.radio(
            "表示カテゴリ:",
            ["すべて", "総力戦", "大決戦"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        
    # 時系列（開催期間の開始日）の新しい順に並び替え
    from datetime import datetime
    
    def get_event_start_date(eid):
        meta = EVENT_META.get(normalize_event_id(eid), {})
        period = meta.get("period", "")
        if period:
            start_date_str = period.split(" ～ ")[0].strip()
            try:
                return datetime.strptime(start_date_str, "%Y/%m/%d")
            except Exception:
                pass
        return datetime.min
        
    sorted_events = sorted(available_events, key=get_event_start_date, reverse=True)
    
    # フィルタリング
    display_events = []
    for eid in sorted_events:
        meta = EVENT_META.get(normalize_event_id(eid), {})
        boss = meta.get("boss", "").lower()
        season = meta.get("season", "").lower()
        eid_lower = eid.lower()
        
        # 検索マッチ
        q = search_query.strip().lower()
        if q:
            if q not in boss and q not in season and q not in eid_lower:
                continue
                
        # カテゴリマッチ
        is_total = normalize_event_id(eid).startswith("total_assault_")
        if portal_tab == "総力戦" and not is_total:
            continue
        if portal_tab == "大決戦" and is_total:
            continue
            
        display_events.append(eid)
        
    # 4列 of グリッドでカードを描画
    if display_events:
        cols = st.columns(4)
        for i, eid in enumerate(display_events):
            col = cols[i % 4]
            meta = EVENT_META.get(normalize_event_id(eid), {})
            boss_name = meta.get("boss", "Unknown")
            season_num = meta.get("season", eid)
            period = meta.get("period", "")
            is_total = normalize_event_id(eid).startswith("total_assault_")
            
            # 最新のサフィックス（時期）を取得してボーダースコア算出に使用する
            suffixes = event_suffix_map.get(eid, [])
            latest_suffix = suffixes[0] if suffixes else None
            
            total_players, plat_score_portal, plat_time_str = get_portal_card_stats(eid, suffix=latest_suffix)
                
            plat_score_str_portal = f"{int(plat_score_portal):,}" if plat_score_portal is not None else "データ不足"

            # 最終更新情報表示用の文字列を作成
            if latest_suffix == "last":
                update_status_str = "最終結果"
            elif latest_suffix == "":
                update_status_str = ""
            else:
                match_dt = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})$", latest_suffix)
                if match_dt:
                    update_status_str = f"最終更新日: {match_dt.group(1)}/{match_dt.group(2)}/{match_dt.group(3)} {match_dt.group(4)}:{match_dt.group(5)}"
                else:
                    update_status_str = f"最終更新日: {latest_suffix}"

            with col:
                badge_color = "#3b82f6" if is_total else "#10b981"
                type_label = "総力戦" if is_total else "大決戦"
                
                time_display_html = f"<div class='card-border-time'>{plat_time_str}</div>" if plat_time_str else "<div class='card-border-time-placeholder'></div>"
                
                status_html = f"<div style='color: #94a3b8; font-size: 0.78rem; margin-top: 10px;'>{update_status_str}</div>" if update_status_str else ""
                
                card_html = f"""<div class="portal-card"><a href="?event_id={eid}" target="_self" class="portal-card-link-overlay"></a><div><div class="card-header"><span class="card-badge" style="background-color: {badge_color};">{type_label}</span><span class="card-period">{period}</span></div><div class="card-title-row"><span class="card-season">{season_num}</span><span class="card-boss">{boss_name}</span></div><div class="card-border-area"><img class="card-border-img" src="data:image/png;base64,{platinum_base64}" /><div class="card-border-info"><div class="card-border-score">{plat_score_str_portal}</div>{time_display_html}</div></div></div>{status_html}</div>"""
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("該当するシーズンが見つかりませんでした。")

else:
    # ====================================================
    # B. 詳細ダッシュボード表示
    # ====================================================
    if df is None or df.empty:
        st.info("👈 左側のサイドバーから有効なイベントIDを選択または入力してください。")
        st.stop()
        
    # 選択イベントのタイトル表示
    st.subheader(get_display_name(event_id))
    
    # 各種ボーダースコアの事前計算
    # 各ボーダー：ocr、boundaryで該当のrankに位置するもの
    def get_border_score(rank_val):
        if rank_val in df.index:
            row = df.loc[rank_val]
            st = row.get('status') if 'status' in df.columns else 'ocr'
            st_str = str(st) if (not pd.isna(st) and st is not None) else 'ocr'
            if st_str in ['ocr', 'boundary_border']:
                return row['score']
        return None

    plat_score = get_border_score(20000)
    gold_score = get_border_score(120000)
    silver_score = get_border_score(240000)
    
    plat_score_str = f"{int(plat_score):,}" if (plat_score is not None and not pd.isna(plat_score)) else "データ不足"
    gold_score_str = f"{int(gold_score):,}" if (gold_score is not None and not pd.isna(gold_score)) else "データ不足"
    silver_score_str = f"{int(silver_score):,}" if (silver_score is not None and not pd.isna(silver_score)) else "データ不足"

    # 総参加者数の取得
    total_participants = None
    if 'status' in df.columns:
        total_rows = df[df['status'] == 'boundary_total']
        if not total_rows.empty:
            total_participants = total_rows.index.max()
    if total_participants is None:
        total_participants = df.index.max() if not df.empty else None


    # ====================================================
    # 2. チナトロ・ゴルドロ・シルトロボーダー (上から2番目 - 横並びカード)
    # ====================================================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            sub_img, sub_title = st.columns([1, 4])
            with sub_img:
                if os.path.exists("image/platinum.png"):
                    st.image("image/platinum.png", width=35)
            with sub_title:
                html_content = f"<div style='margin-bottom: 0px;'><span style='color: #a855f7; font-size: 1.6rem; font-weight: bold; line-height: 1.1;'>{plat_score_str}</span></div>"
                if app_mode.startswith("総力戦") and plat_score is not None and not pd.isna(plat_score):
                    diff, t_sec = score_to_clear_time(plat_score, event_id)
                    t_str = format_time_short(t_sec)
                    html_content += f"<div style='margin-top: -2px; color: #888; font-size: 0.95rem; font-weight: bold;'>{diff} {t_str}</div>"
                st.markdown(html_content, unsafe_allow_html=True)
            
    with col2:
        with st.container(border=True):
            sub_img, sub_title = st.columns([1, 4])
            with sub_img:
                if os.path.exists("image/gold.png"):
                    st.image("image/gold.png", width=35)
            with sub_title:
                html_content = f"<div style='margin-bottom: 0px;'><span style='color: #D4AF37; font-size: 1.6rem; font-weight: bold; line-height: 1.1;'>{gold_score_str}</span></div>"
                if app_mode.startswith("総力戦") and gold_score is not None and not pd.isna(gold_score):
                    diff, t_sec = score_to_clear_time(gold_score, event_id)
                    t_str = format_time_short(t_sec)
                    html_content += f"<div style='margin-top: -2px; color: #888; font-size: 0.95rem; font-weight: bold;'>{diff} {t_str}</div>"
                st.markdown(html_content, unsafe_allow_html=True)
            
    with col3:
        with st.container(border=True):
            sub_img, sub_title = st.columns([1, 4])
            with sub_img:
                if os.path.exists("image/silver.png"):
                    st.image("image/silver.png", width=35)
            with sub_title:
                html_content = f"<div style='margin-bottom: 0px;'><span style='color: #A8A8A8; font-size: 1.6rem; font-weight: bold; line-height: 1.1;'>{silver_score_str}</span></div>"
                if app_mode.startswith("総力戦") and silver_score is not None and not pd.isna(silver_score):
                    diff, t_sec = score_to_clear_time(silver_score, event_id)
                    t_str = format_time_short(t_sec)
                    html_content += f"<div style='margin-top: -2px; color: #888; font-size: 0.95rem; font-weight: bold;'>{diff} {t_str}</div>"
                st.markdown(html_content, unsafe_allow_html=True)
        
    total_ocr_count = len(df[df['status'] == 'ocr']) if 'status' in df.columns else len(df)
    
    # 総参加者数の取得 (boundary_total)
    total_participants = None
    if 'status' in df.columns:
        # boundary_total のみから取得
        total_rows = df[df['status'] == 'boundary_total']
        if not total_rows.empty:
            total_participants = total_rows.index.max()

    participants_str = f"{total_participants:,} 人" if total_participants is not None else "データ不足"
    st.markdown(
        f"<h3 style='text-align: center; font-weight: bold; margin-top: 10px;'>"
        f"総参加者数 {participants_str}"
        f"</h3>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ====================================================
    # 3. クリア状況サマリー・プレイヤー検索 (上から3番目 - 折りたたみ)
    # ====================================================
    with st.expander("クリア状況サマリー・プレイヤー検索", expanded=True):
        # 検索用データ: status == 'ocr' かつ score が NaN でないもの（元の順位を保持する）
        search_df = df.copy()
        if 'status' in search_df.columns:
            search_df = search_df[search_df['status'] == 'ocr']
        search_df = search_df[search_df['score'].notna()]
        search_df.index.name = 'rank'
        sorted_search_df = search_df.sort_values('score', ascending=False).reset_index()
        
        # 統合検索文字列の入力 (プレースホルダーに情報を集約)
        if app_mode.startswith("総力戦"):
            search_label = "順位・スコア・タイム検索"
            search_placeholder = "順位、スコア、またはタイムを入力してください (例: 20000, 31076000, 2:17.833, Torment 2:15.000)"
        else:
            search_label = "順位・スコア検索"
            search_placeholder = "順位、またはスコアを入力してください (例: 20000, 104800000)"

        search_query = st.text_input(
            search_label,
            value="",
            placeholder=search_placeholder,
            label_visibility="collapsed",
            key="unified_search_input"
        )
        
        target_records = []
        
        if search_query:
            query = search_query.strip()
            # 判別ロジック
            difficulty_keywords = ["lunatic", "torment", "insane", "extreme", "hardcore", "veryhard", "hard", "normal",
                                   "ルナティック", "トーメント", "インセイン", "エクストリーム", "ハードコア", "ベリーハード", "ハード", "ノーマル"]
            
            is_time = False
            if ":" in query or "分" in query or "秒" in query or ("." in query and not query.replace(".", "").isdigit()):
                is_time = True
            for kw in difficulty_keywords:
                if kw in query.lower():
                    is_time = True
                    break
                    
            if is_time:
                if not app_mode.startswith("総力戦"):
                    st.warning("⚠️ 大決戦ではクリアタイムでの検索は行えません。順位またはスコアを入力してください。")
                else:
                    detected_diff = None
                    for kw in difficulty_keywords:
                        if kw in query.lower():
                            if kw in ["lunatic", "ルナティック"]: detected_diff = "Lunatic"
                            elif kw in ["torment", "トーメント"]: detected_diff = "Torment"
                            elif kw in ["insane", "インセイン"]: detected_diff = "Insane"
                            elif kw in ["extreme", "エクストリーム"]: detected_diff = "Extreme"
                            elif kw in ["hardcore", "ハードコア"]: detected_diff = "Hardcore"
                            elif kw in ["veryhard", "ベリーハード"]: detected_diff = "VeryHard"
                            elif kw in ["hard", "ハード"]: detected_diff = "Hard"
                            elif kw in ["normal", "ノーマル"]: detected_diff = "Normal"
                            break
                    
                    time_match = re.search(r"(\d+)[:分](\d+)(?:[:.秒](\d+))?", query)
                    total_seconds = None
                    if time_match:
                        minutes = int(time_match.group(1))
                        seconds = int(time_match.group(2))
                        ms = int(time_match.group(3)) if time_match.group(3) else 0
                        if ms > 0:
                            ms_str = time_match.group(3)
                            ms_val = float(f"0.{ms_str}")
                            total_seconds = minutes * 60 + seconds + ms_val
                        else:
                            total_seconds = minutes * 60 + seconds
                    else:
                        sec_match = re.search(r"(\d+(?:\.\d+)?)", query)
                        if sec_match:
                            total_seconds = float(sec_match.group(1))
                            
                    if total_seconds is not None:
                        meta = EVENT_META.get(normalize_event_id(event_id))
                        boss_name = meta["boss"] if meta else "ビナー"
                        limit_sec = 240
                        if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
                            limit_sec = 180
                        elif boss_name in ["イェソド", "ドラム缶ガニ"]:
                            limit_sec = 270
                        limit_type = limit_sec / 60.0
                        
                        if limit_type == 3.0:
                            params = {"Lunatic": (43235000, 2880), "Torment": (31076000, 2400), "Insane": (19249600, 1920), "Extreme": (9392000, 1440), "Hardcore": (3832000, 960), "VeryHard": (1916000, 480), "Hard": (958000, 240), "Normal": (479000, 120)}
                        elif limit_type == 4.0:
                            params = {"Lunatic": (44025000, 2880), "Torment": (31708000, 2400), "Insane": (21016000, 1920), "Extreme": (10160000, 1440), "Hardcore": (4216000, 960), "VeryHard": (2108000, 480), "Hard": (1054000, 240), "Normal": (527000, 120)}
                        else:
                            params = {"Lunatic": (44664000, 2880), "Torment": (32502000, 2400), "Insane": (21741016, 1920), "Extreme": (10578880, 1440), "Hardcore": (4437600, 960), "VeryHard": (2218800, 480), "Hard": (1109400, 240), "Normal": (554700, 120)}
                        
                        target_diffs = [detected_diff] if detected_diff else ["Lunatic", "Torment", "Insane", "Extreme", "Hardcore", "VeryHard", "Hard", "Normal"]
                        
                        for diff_candidate in target_diffs:
                            if diff_candidate in params:
                                base_score, k = params[diff_candidate]
                                est_score = base_score + (3600 - total_seconds) * k
                                
                                act_rank, act_score = find_nearest_player(sorted_search_df, est_score)
                                if act_rank in df.index:
                                    matched_row = df.loc[act_rank]
                                    matched_score_val = matched_row['score']
                                    act_diff, act_t_sec = score_to_clear_time(matched_score_val, event_id)
                                    
                                    if not detected_diff:
                                        if act_diff == diff_candidate:
                                            target_records.append({
                                                "順位": f"{int(act_rank):,} 位",
                                                "クリア難易度": act_diff,
                                                "スコア": f"{int(matched_score_val):,}",
                                                "クリアタイム": format_time_short(act_t_sec),
                                                "sort_key": int(act_rank) - 1
                                            })
                                    else:
                                        target_records.append({
                                            "順位": f"{int(act_rank):,} 位",
                                            "クリア難易度": act_diff,
                                            "スコア": f"{int(matched_score_val):,}",
                                            "クリアタイム": format_time_short(act_t_sec),
                                            "sort_key": int(act_rank) - 1
                                        })
                        
                        if not detected_diff and not target_records:
                            for diff_candidate in params.keys():
                                base_score, k = params[diff_candidate]
                                est_score = base_score + (3600 - total_seconds) * k
                                act_rank, act_score = find_nearest_player(sorted_search_df, est_score)
                                if act_rank in df.index:
                                    matched_row = df.loc[act_rank]
                                    matched_score_val = matched_row['score']
                                    act_diff, act_t_sec = score_to_clear_time(matched_score_val, event_id)
                                    target_records.append({
                                        "順位": f"{int(act_rank):,} 位",
                                        "クリア難易度": act_diff,
                                        "スコア": f"{int(matched_score_val):,}",
                                        "クリアタイム": format_time_short(act_t_sec),
                                        "sort_key": int(act_rank) - 1
                                    })
            
            if not target_records:
                clean_query = query.replace(",", "").replace("位", "").replace("人", "").strip()
                if clean_query.isdigit():
                    val = int(clean_query)
                    max_rank = df.index.max() if (df is not None and not df.empty) else 0
                    if val <= max_rank:
                        if val in df.index:
                            matched_row = df.loc[val]
                            matched_score_val = matched_row['score']
                            
                            if pd.isna(matched_score_val) or matched_score_val is pd.NA:
                                rec = {
                                    "順位": f"{val:,} 位",
                                    "スコア": "欠損",
                                    "sort_key": val - 1
                                }
                                if app_mode.startswith("総力戦"):
                                    rec["クリア難易度"] = "不明"
                                    rec["クリアタイム"] = "不明"
                            else:
                                rec = {
                                    "順位": f"{val:,} 位",
                                    "スコア": f"{int(matched_score_val):,}",
                                    "sort_key": val - 1
                                }
                                if app_mode.startswith("総力戦"):
                                    diff, t_sec = score_to_clear_time(matched_score_val, event_id)
                                    rec["クリア難易度"] = diff
                                    rec["クリアタイム"] = format_time_short(t_sec)
                            target_records.append(rec)
                    else:
                        act_rank, act_score = find_nearest_player(sorted_search_df, val)
                        if act_rank in df.index:
                            matched_row = df.loc[act_rank]
                            matched_score_val = matched_row['score']
                            rec = {
                                "順位": f"{int(act_rank):,} 位",
                                "スコア": f"{int(matched_score_val):,}",
                                "sort_key": int(act_rank) - 1
                            }
                            if app_mode.startswith("総力戦"):
                                diff, t_sec = score_to_clear_time(matched_score_val, event_id)
                                rec["クリア難易度"] = diff
                                rec["クリアタイム"] = format_time_short(t_sec)
                            target_records.append(rec)
                else:
                    if app_mode.startswith("総力戦"):
                        st.error("⚠️ 入力された形式を理解できませんでした。順位、スコア、またはタイムを入力してください。")
                    else:
                        st.error("⚠️ 入力された形式を理解できませんでした。順位、またはスコアを入力してください。")
            
            if target_records:
                display_df = pd.DataFrame(target_records)
                cols_order = ["順位"]
                if "クリア難易度" in display_df.columns:
                    cols_order.append("クリア難易度")
                cols_order.append("スコア")
                if "クリアタイム" in display_df.columns:
                    cols_order.append("クリアタイム")
                    
                display_df = display_df[cols_order]
                st.dataframe(display_df, width="stretch", hide_index=True)
        
        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
        col_tab1, col_tab2 = st.columns(2)
        
        with col_tab1:
            st.markdown("##### 難易度別クリア状況")
            if app_mode.startswith("総力戦"):
                summary_df = make_total_assault_summary(df, event_id)
            else:
                summary_df = make_grand_assault_summary(df, event_id)
            st.dataframe(summary_df, width="stretch", hide_index=True)
            
        with col_tab2:
            st.markdown("##### 主要順位別スコア状況")
            rank_df = get_rank_scores(df, event_id)
            st.dataframe(rank_df, width="stretch", hide_index=True)

    st.markdown("---")

    # ====================================================
    # 4. スコア分布グラフ (上から4番目)
    # ====================================================
    is_total_assault = app_mode.startswith("総力戦")
    expander_label = "スコア・タイム分布グラフ" if is_total_assault else "スコア分布グラフ"
    with st.expander(expander_label, expanded=True):
        
        def format_bracket(zone):
            abbrev_map = {'T': 'TMT', 'I': 'INS', 'E': 'Ext', 'H': 'Hco', 'V': 'Vha', 'A': 'Hrd', 'N': 'Nor', 'L': 'Lun'}
            if zone in ['Lunatic', 'Torment', 'Insane', 'Extreme', 'Hardcore', 'VeryHard', 'Hard', 'Normal']:
                return zone
            if zone == 'Other':
                return 'Other'
            if all(c in abbrev_map for c in zone):
                return '・'.join(abbrev_map[c] for c in zone)
            return zone
            
        if is_total_assault:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                graph_draw_mode = st.radio(
                    "表示データ形式",
                    ["スコア", "タイム"],
                    index=0,
                    horizontal=True,
                    key="graph_draw_mode_select"
                )
            options_zones = ['Lunatic', 'Torment', 'Insane', 'Extreme', 'Hardcore', 'VeryHard', 'Hard', 'Normal']
            default_zones = ['Lunatic', 'Torment']
            with col_g2:
                selected_zones = st.multiselect(
                    "表示する難易度帯 (複数選択可)",
                    options=options_zones,
                    default=default_zones,
                    format_func=format_bracket
                )
        else:
            graph_draw_mode = "スコア"
            options_zones = ['TTT', 'TTI', 'TII', 'III', 'IIE', 'IEE', 'EEE', 'EEH', 'EHH', 'HHH', 'HHV', 'HVV', 'VVV', 'VVA', 'VAA', 'AAA', 'AAN', 'ANN', 'NNN']
            
            # 動的に上位20,000位以内のスコア帯ブロックを計算
            from common.score_converter import grand_assault_score_to_clear_time
            sorted_df = df.sort_values('score', ascending=False)
            top_20k_df = sorted_df.iloc[:20000]
            top_brackets = set()
            for score in top_20k_df['score'].dropna():
                bracket, _ = grand_assault_score_to_clear_time(score, event_id)
                if bracket and bracket != 'Unknown':
                    top_brackets.add(bracket)
            
            default_zones = [z for z in options_zones if z in top_brackets]
            if not default_zones:
                default_zones = ['TTI']
                
            selected_zones = st.multiselect(
                "表示する難易度帯 (複数選択可)",
                options=options_zones,
                default=default_zones,
                format_func=format_bracket
            )
            
        # ボス別のしきい値範囲の上限・下限を厳密に計算
        meta = EVENT_META.get(normalize_event_id(event_id))
        boss_name = meta["boss"] if meta else "ビナー"
        
        limit_sec = 240
        if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
            limit_sec = 180
        elif boss_name in ["イェソド", "ドラム缶ガニ"]:
            limit_sec = 270
            
        limit_type = limit_sec / 60.0
        
        # 各難易度のパラメータ (base_score, k)
        if is_total_assault:
            if limit_type == 3.0:
                calc_params = {
                    "Lunatic": (43235000, 2880), "Torment": (31076000, 2400), "Insane": (19249600, 1920), 
                    "Extreme": (9392000, 1440), "Hardcore": (3832000, 960), "VeryHard": (1916000, 480), "Hard": (958000, 240), "Normal": (479000, 120)
                }
            elif limit_type == 4.0:
                calc_params = {
                    "Lunatic": (44025000, 2880), "Torment": (31708000, 2400), "Insane": (21016000, 1920), 
                    "Extreme": (10160000, 1440), "Hardcore": (4216000, 960), "VeryHard": (2108000, 480), "Hard": (1054000, 240), "Normal": (527000, 120)
                }
            else:
                calc_params = {
                    "Lunatic": (44664000, 2880), "Torment": (32502000, 2400), "Insane": (21741016, 1920), 
                    "Extreme": (10578880, 1440), "Hardcore": (4437600, 960), "VeryHard": (2218800, 480), "Hard": (1109400, 240), "Normal": (554700, 120)
                }
            ordered_zones = ["Lunatic", "Torment", "Insane", "Extreme", "Hardcore", "VeryHard", "Hard", "Normal"]
        else:
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
            calc_params = {}
            for i, combo in enumerate(options_zones):
                max_score = sum(params[char][0] for char in combo)
                mean_k = sum(params[char][1] for char in combo) / 3.0
                if i < len(options_zones) - 1:
                    next_combo = options_zones[i+1]
                    border = sum(params[char][0] for char in next_combo)
                else:
                    border = 0
                calc_params[combo] = (border, mean_k)
            ordered_zones = options_zones
            
        # 各難易度ゾーンのスコア範囲をループで動的に算出
        zone_ranges = {}
        for idx, zone in enumerate(ordered_zones):
            base_score, k = calc_params[zone]
            z_min = base_score
            if idx == 0:
                if not is_total_assault:
                    z_max = sum(params[char][0] for char in 'TTT')
                else:
                    z_max = base_score + 3600 * k # 最上位難易度の理論上の最大値
            else:
                # 1つ上の難易度の下限直下を上限とする
                z_max = calc_params[ordered_zones[idx-1]][0] - 1
            zone_ranges[zone] = (z_min, z_max)

        # ============================================
        # 自動しきい値計算（パーセンタイルベース）
        # ============================================
        def auto_compress_threshold(df, score_min, score_max, percentile=3.0):
            zone_scores = df[(df['score'] >= score_min) & (df['score'] < score_max)]['score']
            if zone_scores.empty or len(zone_scores) < 10:
                return int(score_min)
            threshold = int(np.percentile(zone_scores, percentile))
            return max(int(score_min), threshold)

        # 難易度別のパーセンタイル初期設定
        percentile_settings = {z: 30.0 for z in ordered_zones}
        if is_total_assault:
            percentile_settings["Lunatic"] = 5.0
            percentile_settings["Torment"] = 30.0
            percentile_settings["Insane"] = 30.0
        else:
            percentile_settings["TTT"] = 30.0
            percentile_settings["TTI"] = 30.0
            percentile_settings["TII"] = 30.0
            
        auto_defaults = {}
        sorted_df = df.sort_values('score', ascending=False)
        
        # 20,000位のスコアと属する難易度帯を特定
        rank_20k_idx = min(19999, len(sorted_df) - 1)
        score_20k = int(sorted_df.iloc[rank_20k_idx]['score']) if rank_20k_idx >= 0 else 0
        from common.score_converter import score_to_clear_time
        bracket_20k, _ = score_to_clear_time(score_20k, event_id)
        
        # 21,000位のスコア
        rank_21k_idx = min(20999, len(sorted_df) - 1)
        score_21k = int(sorted_df.iloc[rank_21k_idx]['score']) if rank_21k_idx >= 0 else 0
        
        for zone in ordered_zones:
            z_min, z_max = zone_ranges[zone]
            if zone == bracket_20k:
                # 条件①: 20,000位が該当する難易度帯の新しい計算方法
                # チナトロボーダー (score_20k) - グラフ1本当たりの幅 * 6
                
                # デフォルトのスコア幅
                default_score_bins = {z: 10000 for z in ordered_zones}
                if is_total_assault:
                    default_score_bins["Lunatic"] = 150000
                    default_score_bins["Torment"] = 1500
                    default_score_bins["Insane"] = 1500
                    default_score_bins["Extreme"] = 30000
                    default_score_bins["Hardcore"] = 30000
                
                # モードに応じて現在の幅(bin)を特定
                if graph_draw_mode == "タイム" and is_total_assault:
                    # タイムモード時のデフォルト秒数幅
                    default_time_bins = {z: 1.0 for z in ordered_zones}
                    default_time_bins["Lunatic"] = 60.0
                    default_time_bins["Torment"] = 0.5
                    default_time_bins["Insane"] = 0.5
                    default_time_bins["Extreme"] = 10.0
                    default_time_bins["Hardcore"] = 10.0
                    
                    time_bin_sec = default_time_bins.get(zone, 1.0)
                    bin_session_key = f"{zone}_t_bin_{selected_suffix}"
                    if bin_session_key in st.session_state:
                        time_bin_sec = st.session_state[bin_session_key]
                    
                    base_score, k = calc_params[zone]
                    current_bin = int(time_bin_sec * k)
                else:
                    # スコアモード時
                    current_bin = default_score_bins.get(zone, 10000)
                    bin_session_key = f"{zone}_s_bin_{selected_suffix}"
                    if bin_session_key in st.session_state:
                        current_bin = st.session_state[bin_session_key]
                
                val = score_20k - (current_bin * 6)
                if val < z_min:
                    val = z_min
            else:
                # 条件②: 該当以外は変更前のパーセンタイル計算（Lunatic: 5%, その他: 30%）
                pct = percentile_settings.get(zone, 30.0)
                val = auto_compress_threshold(df, z_min, z_max, percentile=pct)
            
            # 条件③: 下4桁切り捨て
            val = (val // 10000) * 10000
            auto_defaults[zone] = max(int(z_min), val)


        with st.expander("難易度別詳細パラメータ設定", expanded=False):
            if graph_draw_mode == "タイム" and is_total_assault:
                max_min = 60
                max_sec = 0
                limit_text = "60分00秒"
                
                default_time_bins = {z: 1.0 for z in ordered_zones}
                default_time_bins["Lunatic"] = 60.0
                default_time_bins["Torment"] = 0.5
                default_time_bins["Insane"] = 0.5
                default_time_bins["Extreme"] = 10.0
                default_time_bins["Hardcore"] = 10.0
                
                active_zones = [z for z in ordered_zones if z in selected_zones]
                compress_settings = {}
                bin_settings = {}
                
                import math
                for zone in ordered_zones:
                    base_score, k = calc_params[zone]
                    auto_def = auto_defaults[zone]
                    auto_time = max(0.0, 3600 - (auto_def - base_score) / k)
                    auto_time = min(auto_time, 3600.0)
                    
                    if zone == "Lunatic":
                        auto_time = math.ceil(auto_time / 60.0) * 60.0
                    else:
                        auto_time = math.ceil(auto_time)
                        
                    compress_settings[zone] = base_score + (3600 - auto_time) * k
                    bin_settings[zone] = int(default_time_bins[zone] * k)
                    
                if not active_zones:
                    st.info("表示する難易度帯が選択されていません。")
                else:
                    cols = st.columns(len(active_zones))
                    for col_idx, zone in enumerate(active_zones):
                        with cols[col_idx]:
                            st.markdown(f"**{format_bracket(zone)} 設定**")
                            base_score, k = calc_params[zone]
                            auto_def = auto_defaults[zone]
                            
                            default_bin_sec = default_time_bins[zone]
                            step_val = 0.1 if zone in ["Torment", "Insane"] else (1.0 if default_bin_sec >= 1.0 else 0.5)
                            time_bin = st.number_input("グラフ１本当たりの幅", min_value=0.1, max_value=120.0, value=default_bin_sec, step=step_val, key=f"{zone}_t_bin_{selected_suffix}")
                            bin_settings[zone] = int(time_bin * k)
                            
                            auto_time = max(0.0, 3600 - (auto_def - base_score) / k)
                            auto_time = min(auto_time, 3600.0)
                            
                            if zone == "Lunatic":
                                auto_time = math.ceil(auto_time / 60.0) * 60.0
                            else:
                                auto_time = math.ceil(auto_time)
                            
                            # M:SS.ms 形式にフォーマットして初期値とする
                            from common.score_converter import format_time_short
                            default_time_str = format_time_short(auto_time)
                            
                            time_str_input = st.text_input(
                                "下限タイム",
                                value=default_time_str,
                                key=f"{zone}_t_str_{selected_suffix}_{time_bin}",
                                help="例: 2:20.833 や 12:00.000"
                            )
                            
                            total_sec = None
                            try:
                                parts = time_str_input.split(':')
                                if len(parts) == 2:
                                    minutes = int(parts[0])
                                    sec_parts = parts[1].split('.')
                                    if len(sec_parts) == 2:
                                        seconds = int(sec_parts[0])
                                        ms = int(sec_parts[1])
                                        total_sec = minutes * 60 + seconds + ms / 1000.0
                                    else:
                                        seconds = float(parts[1])
                                        total_sec = minutes * 60 + seconds
                                else:
                                    total_sec = float(time_str_input)
                            except Exception:
                                pass
                                
                            if total_sec is None or total_sec < 0:
                                total_sec = auto_time
                                if time_str_input:
                                    st.caption("⚠️ 正しい形式で入力してください。")
                                    
                            compress_settings[zone] = base_score + (3600 - total_sec) * k

            else:
                # スコアモード（大決戦は常にこちら、総力戦もスコア選択時）
                default_score_bins = {z: 10000 for z in ordered_zones}
                if is_total_assault:
                    default_score_bins["Lunatic"] = 150000
                    default_score_bins["Torment"] = 1500
                    default_score_bins["Insane"] = 1500
                    default_score_bins["Extreme"] = 30000
                    default_score_bins["Hardcore"] = 30000
                
                active_zones = [z for z in ordered_zones if z in selected_zones]
                compress_settings = {}
                bin_settings = {}
                
                for zone in ordered_zones:
                    compress_settings[zone] = auto_defaults[zone]
                    bin_settings[zone] = default_score_bins.get(zone, 10000)
                    
                if not active_zones:
                    st.info("表示する難易度帯が選択されていません。")
                else:
                    cols = st.columns(len(active_zones))
                    for col_idx, zone in enumerate(active_zones):
                        z_min, z_max = zone_ranges[zone]
                        auto_def = auto_defaults[zone]
                        
                        with cols[col_idx]:
                            st.markdown(f"**{format_bracket(zone)} 設定**")
                            
                            default_bin_val = default_score_bins.get(zone, 10000)
                            bin_val = st.number_input(
                                "グラフ１本当たりの幅",
                                min_value=10,
                                max_value=1000000,
                                value=default_bin_val,
                                step=100 if default_bin_val < 5000 else 1000,
                                key=f"{zone}_s_bin_{selected_suffix}"
                            )
                            bin_settings[zone] = bin_val
                            
                            comp_val = st.number_input(
                                "下限スコア",
                                min_value=int(z_min),
                                max_value=int(z_max),
                                value=int(auto_def),
                                step=1000 if (z_max - z_min) < 100000 else 10000,
                                help=f"範囲: {int(z_min):,} ～ {int(z_max):,}",
                                key=f"{zone}_s_compress_{selected_suffix}_{bin_val}"
                            )
                            compress_settings[zone] = comp_val

        # グラフ用データ: ocrのみ
        graph_df = df[df['status'] == 'ocr'] if 'status' in df.columns else df

        # 描画
        if is_total_assault:
            fig = cached_total_assault_graph(
                df=graph_df,
                event_id=event_id,
                draw_mode=graph_draw_mode,
                selected_zones_tuple=tuple(selected_zones),
                compress_tuple=tuple(compress_settings.items()),
                bin_tuple=tuple(bin_settings.items())
            )
        else:
            fig = cached_grand_assault_graph(
                df=graph_df,
                event_id=event_id,
                suffix=selected_suffix,
                draw_mode=graph_draw_mode,
                selected_zones_tuple=tuple(selected_zones),
                compress_tuple=tuple(compress_settings.items()),
                bin_tuple=tuple(bin_settings.items())
            )
        if fig:
            st.pyplot(fig)
            plt.close(fig)
                



