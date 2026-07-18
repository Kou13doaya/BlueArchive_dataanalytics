# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common.event_metadata import EVENT_META, normalize_event_id

# ==========================================
# 設定：デザイン
# ==========================================
BAR_COLOR = '#a0d8ef'
BG_COLOR = 'white'
TEXT_COLOR = 'black'
COMPRESSED_COLOR = '#b0c4de'
GRID_COLOR = '#cccccc'

# デフォルト設定値
DEFAULT_SETTINGS = {
    'High': {'range': [98860000, 121044000], 'compress': 107000000, 'bin': 10000},
    'Mid': {'range': [46500000, 83784000], 'compress': 69932800, 'bin': 10000},
    'Low': {'range': [43440000, 46032000], 'compress': 44995200, 'bin': 10000}
}

def create_single_block_histogram(df_target, settings, block_name):
    """
    大決戦の特定スコア帯（ブロック）のヒストグラムデータを生成します。
    """
    if df_target.empty:
        return pd.DataFrame(), 0

    combined_data = []

    range_vals = settings['range']
    lower_limit = range_vals[0] # End (Min)
    upper_limit = range_vals[1] # Start (Max)
    comp_below = settings['compress']
    bin_size = settings['bin']

    if upper_limit <= lower_limit:
        return pd.DataFrame(), 0

    # 1. 範囲抽出
    block_df = df_target[
        (df_target['score'] < upper_limit) &
        (df_target['score'] >= lower_limit)
    ].copy()

    total_in_range = len(block_df)

    if block_df.empty:
        return pd.DataFrame(), 0

    # 2. Detailed Zone
    detail_df = block_df[block_df['score'] >= comp_below].copy()
    if not detail_df.empty:
        actual_max = detail_df['score'].max()
        # ビン数の安全チェック（300個以上に増えすぎないよう自動スケーリング）
        estimated_bins = (actual_max - comp_below) / bin_size
        if estimated_bins > 300:
            bin_size = int(np.ceil((actual_max - comp_below) / 300))
            
        grid_min = comp_below
        grid_max = (actual_max // bin_size) * bin_size
        full_index = range(int(grid_min), int(grid_max) + 1, bin_size)

        detail_df['binned'] = comp_below + ((detail_df['score'] - comp_below) // bin_size) * bin_size
        counts = detail_df['binned'].value_counts()
        counts_filled = counts.reindex(full_index, fill_value=0).sort_index(ascending=True)

        for score, count in counts_filled.items():
            if score < comp_below:
                continue
            combined_data.append({
                'label': f"{score:,}",
                'count': count,
                'type': 'detail',
                'sort_key': score,
                'color': BAR_COLOR,
                'group': block_name
            })

    # 3. Compressed Zone
    compressed_df = block_df[block_df['score'] < comp_below]
    if not compressed_df.empty:
        count = len(compressed_df)
        combined_data.append({
            'label': f"{block_name} Low",
            'count': count,
            'type': 'compressed',
            'sort_key': lower_limit,
            'color': COMPRESSED_COLOR,
            'group': block_name
        })

    result_df = pd.DataFrame(combined_data)
    if not result_df.empty:
        result_df = result_df.sort_values('sort_key', ascending=True)

    return result_df, total_in_range


def draw_grand_assault_graph(df, event_id=None, suffix=None, view_mode='High', settings=None, save_path=None, show=False):
    """
    大決戦データを可視化したグラフを作成します。
    """
    if df is None or df.empty:
        print("[ERROR] データがありません。")
        return

    # 設定構築
    if settings is None:
        target_settings = DEFAULT_SETTINGS.get(view_mode)
    else:
        target_settings = settings

    target_name = view_mode

    # 全データ使用
    df_display = df.copy()

    # データ生成
    graph_data, total_count = create_single_block_histogram(df_display, target_settings, target_name)

    if graph_data.empty:
        print("[ERROR] 表示データがありません。範囲を広げるか設定を確認してください。")
        return

    # 天井の空行追加
    last_row = graph_data.iloc[-1]
    if last_row['type'] == 'detail':
        b_size = target_settings['bin']
        next_score = last_row['sort_key'] + b_size
        new_row = pd.DataFrame([{
            'label': f"{next_score:,}", 'count': 0, 'type': 'detail',
            'sort_key': next_score, 'color': BAR_COLOR, 'group': target_name
        }])
        graph_data = pd.concat([graph_data, new_row], ignore_index=True)

    # 描画設定
    max_detail_val = graph_data[graph_data['type'] == 'detail']['count'].max()
    if pd.isna(max_detail_val) or max_detail_val == 0:
        max_detail_val = 1
    clip_limit = int(max_detail_val * 1.1)

    plot_counts = graph_data['count'].copy().astype(float)
    mask_comp = graph_data['type'] == 'compressed'
    plot_counts.loc[mask_comp] = plot_counts.loc[mask_comp].clip(upper=clip_limit)

    num_bars = len(graph_data)
    
    # 本数に応じて動的に表示サイズを決定する
    if num_bars <= 80:
        BAR_HEIGHT = 0.6
        height_per_data = 0.22
        font_size_label = 10
        font_size_ytick = 10
    else:
        ratio = max(0.0, min(1.0, (num_bars - 80) / 220)) # 80〜300の比率
        BAR_HEIGHT = 0.6 - (ratio * 0.2)        # 0.6 -> 0.4
        height_per_data = 0.22 - (ratio * 0.08)  # 0.22 -> 0.14
        font_size_label = max(6.5, 10.0 - (ratio * 3.5))
        font_size_ytick = max(6.5, 10.0 - (ratio * 3.5))

    fig_height = (num_bars * height_per_data) + 2

    # 非GUI環境用の設定
    plt.close('all')
    fig, ax = plt.subplots(figsize=(16, max(6, fig_height)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos_grid = np.arange(num_bars)
    y_pos_bar = y_pos_grid + 0.5

    ax.barh(y_pos_bar, plot_counts, color=graph_data['color'], edgecolor='gray', linewidth=0.3, height=BAR_HEIGHT, alpha=1.0)

    ax.grid(axis='y', color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_yticks(y_pos_grid)
    ax.set_yticklabels(graph_data['label'], fontsize=font_size_ytick)
    ax.grid(axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.8)
    ax.axhline(y=num_bars, color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    # ラベル
    for i, (_, row) in enumerate(graph_data.iterrows()):
        count = row['count']
        plot_val = plot_counts.iloc[i]
        bar_y = y_pos_bar[i]
        if count > 0:
            font_w = 'bold' if row['type'] == 'compressed' else 'normal'
            ax.text(plot_val + (clip_limit * 0.01), bar_y, f"{int(count):,}", va='center', color=TEXT_COLOR, fontsize=font_size_label, fontweight=font_w)
        elif row['type'] == 'compressed':
             ax.text(clip_limit * 0.01, bar_y, "Gap", va='center', color='gray', fontsize=font_size_label - 1, fontstyle='italic')

    # ボーダーライン
    sorted_df_all = df.sort_values('score', ascending=False).reset_index(drop=True)
    idx_plat = 19999
    score_plat = sorted_df_all.iloc[idx_plat]['score'] if len(sorted_df_all) > idx_plat else None
    idx_gold = 119999
    score_gold = sorted_df_all.iloc[idx_gold]['score'] if len(sorted_df_all) > idx_gold else None

    def get_y(target_score):
        return np.interp(target_score, graph_data['sort_key'].values, y_pos_grid)

    visible_min = graph_data['sort_key'].min()
    visible_max = graph_data['sort_key'].max()

    if score_plat is not None and visible_min <= score_plat <= visible_max:
        y = get_y(score_plat)
        if 0 <= y <= num_bars:
            ax.axhline(y, color='purple', linewidth=1.5, linestyle='-', alpha=0.9)
            ax.text(ax.get_xlim()[1], y, f" Platinum (20,000th): {score_plat:,} ",
                    va='bottom', ha='right', color='white', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="purple", ec="none", alpha=0.9))

    if score_gold is not None and visible_min <= score_gold <= visible_max:
        y = get_y(score_gold)
        if 0 <= y <= num_bars:
            ax.axhline(y, color='#daa520', linewidth=1.5, linestyle='-', alpha=0.9)
            ax.text(ax.get_xlim()[1], y, f" Gold (120,000th): {score_gold:,} ",
                    va='bottom', ha='right', color='black', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffd700", ec="none", alpha=0.9))

    ax.set_ylim(0, num_bars)
    ax.set_ylabel('Score Zones', fontsize=12, color=TEXT_COLOR)

    # EVENT_METAからタイトル文字列を作成
    title_suffix = ""
    if event_id:
        meta = EVENT_META.get(normalize_event_id(event_id))
        if meta:
            season = meta.get('season', '')
            time_info = ""
            if suffix == "last":
                time_info = "Final Result"
            elif suffix:
                match_dt = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})$", str(suffix))
                if match_dt:
                    time_info = f"{match_dt.group(1)}/{match_dt.group(2)}/{match_dt.group(3)} {match_dt.group(4)}:{match_dt.group(5)}"
                else:
                    time_info = str(suffix)
            
            if time_info:
                title_suffix = f" - {season} ({time_info})"
            else:
                title_suffix = f" - {season}"

    ax.set_title(f'Grand Assault{title_suffix} | {view_mode} Block View', fontsize=16, pad=15)
    ax.set_xlabel('Player Count', fontsize=12)

    # 合計人数表示
    info_text = f"Count in Range: {total_count:,}"
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=BG_COLOR)
        print(f"[SUCCESS] 大決戦のグラフ画像を保存しました: {save_path}")

    if show:
        plt.show()
        plt.close(fig)

    return fig

def draw_grand_assault_parametric_graph(df, event_id, suffix=None, draw_mode='スコア', selected_zones=['TTT', 'TTI', 'TII'],
                                        compress_settings=None, bin_settings=None,
                                        save_path=None, show=False,
                                        **kwargs):
    """
    大決戦データを可視化したグラフを作成します（総力戦と同様のダイナミックヒストグラム）。
    """
    from analytics.total_assault import create_dynamic_histogram
    from common.score_converter import format_time_short, score_to_clear_time
    
    if df is None or df.empty:
        print("[ERROR] データがありません。")
        return None
        
    event_id = normalize_event_id(event_id)
    meta = EVENT_META.get(event_id)
    boss_name = meta["boss"] if meta else "ビナー"
    
    limit_sec = 240
    if boss_name in ["KAITEN FX Mk.0", "ビナー"]:
        limit_sec = 180
    elif boss_name in ["イェソド", "ドラム缶ガニ"]:
        limit_sec = 270
    limit_type = limit_sec / 60.0

    # 制限時間ごとの各難易度の理論値
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
    brackets_dict = {}
    for i, combo in enumerate(combinations):
        max_score = sum(params[char][0] for char in combo)
        mean_k = sum(params[char][1] for char in combo) / 3.0
        
        if i < len(combinations) - 1:
            next_combo = combinations[i+1]
            min_score = sum(params[char][0] for char in next_combo)
        else:
            min_score = 0
            
        brackets_dict[combo] = {
            'border': min_score,
            'max_score': max_score,
            'mean_k': mean_k
        }

    # 1. 設定構築
    dynamic_settings = {}
    for zone in combinations:
        if zone not in selected_zones:
            continue
        border = brackets_dict[zone]['border']
        compress_below = compress_settings.get(zone, border) if compress_settings else border
        bin_val = bin_settings.get(zone, 10000) if bin_settings else 10000
        
        dynamic_settings[zone] = {
            'border': border,
            'compress_below': compress_below,
            'bin': bin_val
        }

    df_display = df.copy()

    # 3. グラフデータの生成
    graph_data = create_dynamic_histogram(df, df_display, dynamic_settings)

    if graph_data.empty:
        print("[ERROR] 表示データがありません。範囲を広げるか設定を確認してください。")
        return None

    # 最上位の空欄追加
    if len(selected_zones) > 0:
        top_zone = selected_zones[0]
        if top_zone in selected_zones:
            last_row = graph_data.iloc[-1]
            if last_row['difficulty'] == top_zone and last_row['type'] == 'detail':
                t_bin = bin_settings.get(top_zone, 10000) if bin_settings else 10000
                top_score = last_row['sort_key']
                next_score = top_score + t_bin
                new_row = pd.DataFrame([{
                    'label': f"{next_score:,}",
                    'count': 0,
                    'type': 'detail',
                    'sort_key': next_score,
                    'color': BAR_COLOR,
                    'difficulty': top_zone
                }])
                graph_data = pd.concat([graph_data, new_row], ignore_index=True)

    # タイム表示の場合はラベルをクリアタイムに変換
    if draw_mode == 'タイム':
        new_labels = []
        for i, row in graph_data.iterrows():
            if row['type'] == 'detail':
                diff_name, t_sec = score_to_clear_time(row['sort_key'], event_id)
                if t_sec is not None:
                    time_str = format_time_short(t_sec)
                    new_labels.append(time_str)
                else:
                    new_labels.append(row['label'])
            else:
                diff_name, t_sec = score_to_clear_time(row['sort_key'], event_id)
                if t_sec is not None:
                    time_str = format_time_short(t_sec)
                    new_labels.append(f"{row['difficulty']} Low")
                else:
                    new_labels.append(row['label'])
        graph_data['label'] = new_labels

    # 4. 集計 & ボーダー
    sorted_df = df.sort_values('score', ascending=False).reset_index(drop=True)
    idx_plat = 19999
    score_plat = sorted_df.iloc[idx_plat]['score'] if len(sorted_df) > idx_plat else None
    idx_gold = 119999
    score_gold = sorted_df.iloc[idx_gold]['score'] if len(sorted_df) > idx_gold else None

    # 5. 描画処理
    max_detail_val = graph_data[graph_data['type'] == 'detail']['count'].max()
    if pd.isna(max_detail_val) or max_detail_val == 0:
        max_detail_val = 1
    clip_limit = int(max_detail_val * 1.1)
    plot_counts = graph_data['count'].copy().astype(float)
    mask_comp = graph_data['type'] == 'compressed'
    plot_counts.loc[mask_comp] = plot_counts.loc[mask_comp].clip(upper=clip_limit)

    num_bars = len(graph_data)
    BAR_HEIGHT = 0.6
    height_per_data = 0.22
    fig_height = (num_bars * height_per_data) + 2

    plt.close('all')
    fig, ax = plt.subplots(figsize=(16, max(6, fig_height)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos_grid = np.arange(num_bars)
    y_pos_bar = y_pos_grid + 0.5

    ax.barh(
        y_pos_bar,
        plot_counts,
        color=graph_data['color'],
        edgecolor='gray',
        linewidth=0.3,
        height=BAR_HEIGHT,
        alpha=1.0
    )

    ax.grid(axis='y', color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_yticks(y_pos_grid)
    ax.set_yticklabels(graph_data['label'])
    ax.grid(axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.8)
    ax.axhline(y=num_bars, color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    for i, (_, row) in enumerate(graph_data.iterrows()):
        count = row['count']
        plot_val = plot_counts.iloc[i]
        bar_y = y_pos_bar[i]
        if count > 0:
            font_w = 'bold' if row['type'] == 'compressed' else 'normal'
            ax.text(plot_val + (clip_limit * 0.01), bar_y, f"{int(count):,}", va='center', color=TEXT_COLOR, fontsize=10, fontweight=font_w)
        elif row['type'] == 'compressed':
            ax.text(clip_limit * 0.01, bar_y, "Gap", va='center', color='gray', fontsize=9, fontstyle='italic')

    # ボーダーライン描画
    def get_exact_y_pos(target_score):
        idx_match = graph_data[graph_data['sort_key'] <= target_score].index
        if idx_match.empty:
            return 0
        match_idx = idx_match[-1]
        
        row_this = graph_data.iloc[match_idx]
        if match_idx == len(graph_data) - 1:
            return y_pos_grid[match_idx] + 0.5
            
        row_next = graph_data.iloc[match_idx + 1]
        
        score_this = row_this['sort_key']
        score_next = row_next['sort_key']
        
        y_this = y_pos_grid[match_idx]
        y_next = y_pos_grid[match_idx + 1]
        
        if score_next == score_this:
            return y_this + 0.5
            
        ratio = (target_score - score_this) / (score_next - score_this)
        return y_this + ratio * (y_next - y_this)

    visible_min = graph_data['sort_key'].min()
    visible_max = graph_data['sort_key'].max()

    if score_plat is not None and visible_min <= score_plat <= visible_max:
        y = get_exact_y_pos(score_plat)
        if 0 <= y <= num_bars:
            ax.axhline(y, color='purple', linewidth=1.5, linestyle='-', alpha=0.9)
            ax.text(ax.get_xlim()[1], y, f" Platinum (20,000th): {score_plat:,} ",
                    va='bottom', ha='right', color='white', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="purple", ec="none", alpha=0.9))

    if score_gold is not None and visible_min <= score_gold <= visible_max:
        y = get_exact_y_pos(score_gold)
        if 0 <= y <= num_bars:
            ax.axhline(y, color='#daa520', linewidth=1.5, linestyle='-', alpha=0.9)
            ax.text(ax.get_xlim()[1], y, f" Gold (120,000th): {score_gold:,} ",
                    va='bottom', ha='right', color='black', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffd700", ec="none", alpha=0.9))

    ax.set_ylim(0, num_bars)
    ax.set_ylabel('Difficulty / Score Zones', fontsize=12, color=TEXT_COLOR)

    title_suffix = ""
    if event_id:
        meta = EVENT_META.get(normalize_event_id(event_id))
        if meta:
            season = meta.get('season', '')
            time_info = ""
            if suffix == "last":
                time_info = "Final Result"
            elif suffix:
                match_dt = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})$", str(suffix))
                if match_dt:
                    time_info = f"{match_dt.group(1)}/{match_dt.group(2)}/{match_dt.group(3)} {match_dt.group(4)}:{match_dt.group(5)}"
                else:
                    time_info = str(suffix)
            if time_info:
                title_suffix = f" - {season} ({time_info})"
            else:
                title_suffix = f" - {season}"

    mode_eng = 'Score' if draw_mode == 'スコア' else 'Time'
    ax.set_title(f'Grand Assault{title_suffix} | Parametric View ({mode_eng} Mode)', fontsize=16, pad=15)
    ax.set_xlabel('Player Count', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=BG_COLOR)

    if show:
        plt.show()
        plt.close(fig)

    return fig

