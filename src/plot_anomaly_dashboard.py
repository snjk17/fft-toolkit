import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_CONFIG

def main():
    config = DEFAULT_CONFIG
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # ディレクトリ設定
    dirs = config["dirs"]
    binned_stats_dirname = dirs.get("binned_stats", "stats-binned")
    output_dirname = os.path.join("がたつき解析", "帯域別統計量の時間推移")
    output_dir = os.path.join(base_dir, output_dirname)
    
    ref_stats_path = os.path.join(output_dir, "reference_stats.csv")
    if not os.path.exists(ref_stats_path):
        print(f"[エラー] {ref_stats_path} が見つかりません。先に plot_combined_interactive.py を実行してください。")
        return

    # リファレンス統計の読み込み
    ref_df = pd.read_csv(ref_stats_path)
    
    print(f"=== 総合ダッシュボード復元開始 ===")
    
    # 1. データの収集
    all_rows = []
    for folder_rel_path in config["target_folders"]:
        target_dir = os.path.join(base_dir, folder_rel_path)
        folder_name = os.path.basename(folder_rel_path)
        stats_dir = os.path.join(target_dir, binned_stats_dirname)
        
        if not os.path.exists(stats_dir):
            continue
            
        csv_files = glob.glob(os.path.join(stats_dir, "*Hz-*Hz.csv"))
        for csv_path in csv_files:
            bin_name = os.path.basename(csv_path).replace(".csv", "")
            try:
                df = pd.read_csv(csv_path)
                if df.empty: continue
                
                df['timestamp'] = pd.to_datetime(
                    df['source_filename'].str.split('_').str[0],
                    format='%Y%m%dT%H%M%S'
                )
                df['folder'] = folder_name
                df['bin_name'] = bin_name
                all_rows.append(df)
            except Exception as e:
                print(f"[警告] {csv_path} の読み込み失敗: {e}")

    if not all_rows:
        print("データが見つかりませんでした。")
        return
        
    full_df = pd.concat(all_rows, ignore_index=True)
    stats_to_plot = [
        'mean_amplitude', 'max_amplitude', 'energy', 
        'energy_ratio', 'std_amplitude', 'variance'
    ]
    
    folders = ["全フォルダ統合"] + sorted(full_df['folder'].unique().tolist())
    
    # 2. Plotly Figure の作成
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("閾値超過帯域数の推移", "全帯域 Z-score ヒートマップ"),
        row_heights=[0.3, 0.7]
    )
    
    # 3. トレースの生成
    trace_indices = []
    
    print("トレースを生成中...")
    for stat in stats_to_plot:
        ref_subset = ref_df[ref_df['statistic'] == stat].set_index('bin_name')
        stat_folder_indices = {}

        for fld in folders:
            if fld == "全フォルダ統合":
                df_fld = full_df.copy()
            else:
                df_fld = full_df[full_df['folder'] == fld].copy()
            
            if df_fld.empty:
                stat_folder_indices[fld] = []
                continue

            # --- トレンドグラフ ---
            counts_by_time = []
            for ts, group in df_fld.groupby('timestamp'):
                ex_s, ex_2s, ex_3s = 0, 0, 0
                for _, row in group.iterrows():
                    bn = row['bin_name']
                    if bn not in ref_subset.index: continue
                    mu = ref_subset.loc[bn, 'mean']
                    sigma = ref_subset.loc[bn, 'std']
                    if pd.isna(mu) or pd.isna(sigma) or sigma <= 0: continue
                    
                    val = row[stat]
                    if val > mu + 3*sigma:
                        ex_s += 1; ex_2s += 1; ex_3s += 1
                    elif val > mu + 2*sigma:
                        ex_s += 1; ex_2s += 1
                    elif val > mu + sigma:
                        ex_s += 1
                
                counts_by_time.append({'timestamp': ts, 'σ': ex_s, '2σ': ex_2s, '3σ': ex_3s})
            
            count_df = pd.DataFrame(counts_by_time).sort_values('timestamp')
            
            idx_start = len(fig.data)
            fig.add_trace(go.Scatter(
                x=count_df['timestamp'], y=count_df['σ'], name="σ超過数",
                line=dict(color='orange', width=1), visible=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=count_df['timestamp'], y=count_df['2σ'], name="2σ超過数",
                line=dict(color='red', width=1.5), visible=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=count_df['timestamp'], y=count_df['3σ'], name="3σ超過数",
                line=dict(color='purple', width=2), visible=False
            ), row=1, col=1)
            
            # --- ヒートマップ ---
            def calc_z(row):
                bn = row['bin_name']
                if bn not in ref_subset.index: return 0
                mu, sigma = ref_subset.loc[bn, 'mean'], ref_subset.loc[bn, 'std']
                return (row[stat] - mu) / sigma if sigma > 0 else 0
            
            df_fld['zscore'] = df_fld.apply(calc_z, axis=1)
            pivot_df = df_fld.groupby(['bin_name', 'timestamp'])['zscore'].mean().unstack()
            
            def bin_sort_key(name):
                try: return int(name.split('Hz')[0])
                except: return 0
            sorted_bins = sorted(pivot_df.index, key=bin_sort_key)
            pivot_df = pivot_df.loc[sorted_bins]
            
            fig.add_trace(go.Heatmap(
                z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index,
                colorscale='Viridis', zmin=0, zmax=5,
                colorbar=dict(title="Z-score", len=0.4, y=0.3),
                visible=False,
                hovertemplate="時刻: %{x}<br>帯域: %{y}<br>Z-score: %{z:.2f}<extra></extra>"
            ), row=2, col=1)
            
            idx_end = len(fig.data)
            stat_folder_indices[fld] = list(range(idx_start, idx_end))
            
        trace_indices.append(stat_folder_indices)

    # 4. メニュー (以前の形式に戻す)
    combined_buttons = []
    for i, stat in enumerate(stats_to_plot):
        for fld in folders:
            visible_mask = [False] * len(fig.data)
            for idx in trace_indices[i].get(fld, []):
                visible_mask[idx] = True
            
            combined_buttons.append(dict(
                label=f"{stat} [{fld}]",
                method="update",
                args=[{"visible": visible_mask},
                      {"title": f"総合ダッシュボード: {stat} ({fld})"}]
            ))

    if combined_buttons:
        fig.update_layout(title=combined_buttons[0]['args'][1]['title'])
        for idx in trace_indices[0]["全フォルダ統合"]:
            fig.data[idx].visible = True

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=combined_buttons,
                direction="down",
                showactive=True,
                x=0.0, xanchor="left", y=1.2, yanchor="top"
            ),
        ],
        xaxis2_title="日時",
        yaxis_title="超過帯域数",
        yaxis2_title="周波数帯 (Hz)",
        height=1000,
        margin=dict(t=150, b=100, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    save_path = os.path.join(output_dir, "総合ダッシュボード.html")
    fig.write_html(save_path)
    print(f"  - 統合ダッシュボードを復元・保存しました: {save_path}")

if __name__ == "__main__":
    main()
