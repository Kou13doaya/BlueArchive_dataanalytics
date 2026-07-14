# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from common.event_metadata import EVENT_META, normalize_event_id
from common.score_converter import score_to_clear_time, format_time_short

# ==========================================
# 固定設定
# ==========================================
BAR_COLOR = '#a0d8ef'
BG_COLOR = 'white'
TEXT_COLOR = 'black'
COMPRESSED_COLOR = '#b0c4de'
GRID_COLOR = '#cccccc'

FIXED_BORDERS = {
    'Lunatic': 44025000,
    'Torment': 31076000,
    'Insane':  19249602,
    'Extreme': 9392000
}

FIXED_SETTINGS_LOWER = {
    'Extreme': {'compress_below': 11984000, 'bin': 50000}
}

def create_dynamic_histogram(df, df_target, dynamic_settings):
    """
    総力戦スコアのダイナミックヒストグラムデータを生成します。
    """
    if df_target.empty:
        return pd.DataFrame()
    cutoff_score = df_target['score'].min()
    combined_data = []

    diff_names = list(dynamic_settings.keys())
    actual_max_scores = {}
    global_max = int(df['score'].max())

    # 各難易度のトップスコア特定
    for i, name in enumerate(diff_names):
        current_border = dynamic_settings[name]['border']
        if i == 0:
            upper = global_max + 1
        else:
            upper = dynamic_settings[diff_names[i-1]]['border']

        zone_data = df[(df['score'] >= current_border) & (df['score'] < upper)]
        actual_max_scores[name] = int(zone_data['score'].max()) if not zone_data.empty else current_border

    current_ceiling = global_max + dynamic_settings[diff_names[0]]['bin']

    for i, name in enumerate(diff_names):
        settings = dynamic_settings[name]
        compress_threshold = settings['compress_below']
        bin_size = settings['bin']
        border = settings['border']

        if i < len(diff_names) - 1:
            next_name = diff_names[i+1]
            compression_floor = actual_max_scores.get(next_name, 0) + 1
        else:
            compression_floor = border

        if current_ceiling < cutoff_score:
            break

        # 1. Detailed Zone
        detail_df = df_target[(df_target['score'] >= compress_threshold) & (df_target['score'] < current_ceiling)].copy()
        if current_ceiling > compress_threshold:
            # ビン数の安全チェック（150個以上に増えすぎないよう自動スケーリング）
            estimated_bins = (current_ceiling - compress_threshold) / bin_size
            if estimated_bins > 150:
                bin_size = int(np.ceil((current_ceiling - compress_threshold) / 150))
                
            # compress_threshold を起点としてグリッド分割を行う
            # （旧: border 起点だと compress_threshold が border の bin_size 倍数でない場合にズレが生じるため修正）
            grid_min = compress_threshold
            grid_max = compress_threshold + int(np.ceil((current_ceiling - compress_threshold) / bin_size)) * bin_size
            full_index = range(int(grid_min), int(grid_max), bin_size)

            if not detail_df.empty:
                detail_df['binned'] = compress_threshold + ((detail_df['score'] - compress_threshold) // bin_size) * bin_size
                counts = detail_df['binned'].value_counts()
                counts_filled = counts.reindex(full_index, fill_value=0).sort_index(ascending=True)
            else:
                counts_filled = pd.Series(0, index=full_index)

            for score, count in counts_filled.items():
                if score < cutoff_score:
                    continue
                # 難易度ラベル(difficulty)を追加
                combined_data.append({
                    'label': f"{score:,}",
                    'count': count,
                    'type': 'detail',
                    'sort_key': score,
                    'color': BAR_COLOR,
                    'difficulty': name
                })

        # 2. Compressed Zone
        if compress_threshold <= cutoff_score:
            current_ceiling = compression_floor
            continue

        compressed_count = len(df_target[(df_target['score'] >= compression_floor) & (df_target['score'] < compress_threshold)])
        label_text = f"{name} (Low) ~ Gap"
        combined_data.append({
            'label': label_text,
            'count': compressed_count,
            'type': 'compressed',
            'sort_key': compression_floor,
            'color': COMPRESSED_COLOR,
            'difficulty': name
        })
        current_ceiling = compression_floor

    result_df = pd.DataFrame(combined_data)
    if not result_df.empty:
        result_df = result_df.sort_values('sort_key', ascending=True)
    return result_df


def draw_parametric_graph(df, event_id, suffix=None, draw_mode='スコア', selected_zones=['Lunatic', 'Torment'],
                          compress_settings=None, bin_settings=None,
                          save_path=None, show=False,
                          **kwargs):
    """
    総力戦データを可視化したグラフを作成します。
    """
    if df is None or df.empty:
        print("[ERROR] データがありません。")
        return

    # ボス別の境界スコア（border）を動的に取得
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
        borders = {
            "Lunatic": 43235000, "Torment": 31076000, "Insane": 19249600, 
            "Extreme": 9392000, "Hardcore": 3832000, "VeryHard": 1916000, "Hard": 958000, "Normal": 479000
        }
    elif limit_type == 4.0:
        borders = {
            "Lunatic": 44025000, "Torment": 31708000, "Insane": 21016000, 
            "Extreme": 10160000, "Hardcore": 4216000, "VeryHard": 2108000, "Hard": 1054000, "Normal": 527000
        }
    else:
        borders = {
            "Lunatic": 44664000, "Torment": 32502000, "Insane": 21741016, 
            "Extreme": 10578880, "Hardcore": 4437600, "VeryHard": 2218800, "Hard": 1109400, "Normal": 554700
        }

    # 後方互換性のための個別引数抽出
    l_compress = kwargs.get('l_compress', 50900000)
    l_bin = kwargs.get('l_bin', 150000)
    t_compress = kwargs.get('t_compress', 39484000)
    t_bin = kwargs.get('t_bin', 3000)
    i_compress = kwargs.get('i_compress', 27467000)
    i_bin = kwargs.get('i_bin', 3000)

    # 1. 設定構築
    dynamic_settings = {}
    ordered_zones = ["Lunatic", "Torment", "Insane", "Extreme", "Hardcore", "VeryHard", "Hard", "Normal"]
    
    for zone in ordered_zones:
        border = borders.get(zone, 0)
        # 各難易度のデフォルトしきい値とビンサイズ
        if zone == 'Lunatic':
            default_comp = l_compress
            default_bin = l_bin
        elif zone == 'Torment':
            default_comp = t_compress
            default_bin = t_bin
        elif zone == 'Insane':
            default_comp = i_compress
            default_bin = i_bin
        else:
            fallback_settings = FIXED_SETTINGS_LOWER.get(zone, {'compress_below': border, 'bin': 30000})
            default_comp = fallback_settings.get('compress_below', border)
            default_bin = fallback_settings.get('bin', 30000)
            
        comp = compress_settings.get(zone, default_comp) if compress_settings else default_comp
        bsize = bin_settings.get(zone, default_bin) if bin_settings else default_bin
        dynamic_settings[zone] = {'border': border, 'compress_below': comp, 'bin': bsize}

    # 2. 選択された難易度の境界スコアによる足切り計算
    sorted_df = df.sort_values('score', ascending=False).reset_index(drop=True)
    min_score_limit = 0

    if selected_zones:
        zone_borders = [borders[z] for z in selected_zones if z in borders]
        if zone_borders:
            min_score_limit = min(zone_borders)

    # 足切り適用
    df_display = sorted_df[sorted_df['score'] >= min_score_limit].copy()

    # 3. データ生成 (全難易度分を作成)
    graph_data = create_dynamic_histogram(df, df_display, dynamic_settings)

    if graph_data.empty:
        print("[ERROR] 表示データなし")
        return

    # フィルタリング：選択された難易度帯のみ残す
    if selected_zones:
        graph_data = graph_data[graph_data['difficulty'].isin(selected_zones)].copy()

    if graph_data.empty:
        print("[ERROR] 選択された難易度のデータはありません")
        return

    # 最上位の空欄追加 (Lunaticが選択されている場合のみ)
    if 'Lunatic' in selected_zones:
        if not graph_data.empty:
            last_row = graph_data.iloc[-1]
            if last_row['difficulty'] == 'Lunatic' and last_row['type'] == 'detail':
                top_score = last_row['sort_key']
                next_score = top_score + l_bin
                new_row = pd.DataFrame([{
                    'label': f"{next_score:,}",
                    'count': 0,
                    'type': 'detail',
                    'sort_key': next_score,
                    'color': BAR_COLOR,
                    'difficulty': 'Lunatic'
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
                    new_labels.append(f"{time_str} ({diff_name})")
                else:
                    new_labels.append(row['label'])
            else:
                diff_name, t_sec = score_to_clear_time(row['sort_key'], event_id)
                if t_sec is not None:
                    time_str = format_time_short(t_sec)
                    new_labels.append(f"{row['difficulty']} (Low) ～ {time_str}")
                else:
                    new_labels.append(row['label'])
        graph_data['label'] = new_labels

    # 4. 集計 & ボーダー
    global_counts = {}
    search_ceiling = int(df['score'].max()) + 1
    for name, settings in dynamic_settings.items():
        count = len(df[(df['score'] >= settings['border']) & (df['score'] < search_ceiling)])
        global_counts[name] = count
        search_ceiling = settings['border']

    idx_plat = 19999
    score_plat = sorted_df.iloc[idx_plat]['score'] if len(sorted_df) > idx_plat else None
    idx_gold = 119999
    score_gold = sorted_df.iloc[idx_gold]['score'] if len(sorted_df) > idx_gold else None

    # ==========================================
    # 描画処理
    # ==========================================
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

    # 非GUI環境用の設定
    plt.close('all')
    fig, ax = plt.subplots(figsize=(16, max(6, fig_height)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos_grid = np.arange(num_bars)
    y_pos_bar = y_pos_grid + 0.5

    # 棒グラフ
    bars = ax.barh(
        y_pos_bar,
        plot_counts,
        color=graph_data['color'],
        edgecolor='gray',
        linewidth=0.3,
        height=BAR_HEIGHT,
        alpha=1.0
    )

    # グリッド & 天井線
    ax.grid(axis='y', color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_yticks(y_pos_grid)
    ax.set_yticklabels(graph_data['label'])
    ax.grid(axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.8)
    ax.axhline(y=num_bars, color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    # 数値ラベル
    for i, (_, row) in enumerate(graph_data.iterrows()):
        count = row['count']
        plot_val = plot_counts.iloc[i]
        bar_y = y_pos_bar[i]

        if count > 0:
            font_w = 'bold' if row['type'] == 'compressed' else 'normal'
            ax.text(
                plot_val + (clip_limit * 0.01),
                bar_y,
                f"{int(count):,}",
                va='center',
                color=TEXT_COLOR,
                fontsize=10,
                fontweight=font_w
            )
        elif row['type'] == 'compressed':
             ax.text(clip_limit * 0.01, bar_y, "Gap", va='center', color='gray', fontsize=9, fontstyle='italic')

    # ボーダーライン (描画範囲内にある場合のみ表示)
    def get_exact_y_pos(target_score):
        scores = graph_data['sort_key'].values
        ys = y_pos_grid
        return np.interp(target_score, scores, ys)

    # 現在の表示範囲の最小スコア
    visible_min_score = graph_data['sort_key'].min()

    if score_plat is not None and score_plat >= visible_min_score:
        y_plat = get_exact_y_pos(score_plat)
        if 0 <= y_plat <= num_bars:
            ax.axhline(y=y_plat, color='purple', linewidth=1.5, linestyle='-', alpha=0.9)
            if draw_mode == 'タイム':
                _, t_sec = score_to_clear_time(score_plat, event_id)
                label_text = f" Platinum (20k): {format_time_short(t_sec)} "
            else:
                label_text = f" Platinum (20k): {score_plat:,} "
            ax.text(ax.get_xlim()[1], y_plat, label_text,
                    va='bottom', ha='right', color='white', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="purple", ec="none", alpha=0.9))

    if score_gold is not None and score_gold >= visible_min_score:
        y_gold = get_exact_y_pos(score_gold)
        if 0 <= y_gold <= num_bars:
            ax.axhline(y=y_gold, color='#daa520', linewidth=1.5, linestyle='-', alpha=0.9)
            if draw_mode == 'タイム':
                _, t_sec = score_to_clear_time(score_gold, event_id)
                label_text = f" Gold (120k): {format_time_short(t_sec)} "
            else:
                label_text = f" Gold (120k): {score_gold:,} "
            ax.text(ax.get_xlim()[1], y_gold, label_text,
                    va='bottom', ha='right', color='black', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffd700", ec="none", alpha=0.9))

    ax.set_ylim(0, num_bars)

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

    if draw_mode == 'タイム':
        ax.set_title(f'Total Assault (Clear Time){title_suffix} | Zones: {", ".join(selected_zones)}', fontsize=16, color=TEXT_COLOR, pad=15)
        ax.set_ylabel('Clear Time (M:SS.ms) / Score Zones', fontsize=12, color=TEXT_COLOR)
    else:
        ax.set_title(f'Total Assault (Score){title_suffix} | Zones: {", ".join(selected_zones)}', fontsize=16, color=TEXT_COLOR, pad=15)
        ax.set_ylabel('Score Zones', fontsize=12, color=TEXT_COLOR)

    ax.set_xlabel('Player Count', fontsize=12, color=TEXT_COLOR)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=BG_COLOR)
        print(f"[SUCCESS] 総力戦のグラフ画像を保存しました: {save_path}")

    if show:
        plt.show()
        plt.close(fig)
    
    return fig
