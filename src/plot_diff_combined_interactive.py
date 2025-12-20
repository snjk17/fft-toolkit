"""
複数フォルダのバンド別差分統計量を統合し、フォルダごとに色分けしたインタラクティブなプロットを作成するスクリプト。
Plotlyを使用し、ホバー時にファイル名を表示するようにします。
"""

import os
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import DEFAULT_CONFIG

def main():
    config = DEFAULT_CONFIG
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dirs = config["dirs"]
    # 差分統計量の集約ディレクトリ (デフォルト値を使用)
    diff_binned_stats_dirname = dirs.get("diff_binned_stats", "diff-stats-binned")
    output_dirname = dirs.get("fig_diff_combined_interactive", "fig-diff-combined-interactive")
    
    output_dir = os.path.join(base_dir, output_dirname)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== 差分統合インタラクティブプロット生成開始 ===")
    
    # 1. データの収集
    # bin_name -> list of dataframes
    all_data_by_bin = {}
    
    for folder_rel_path in config["target_folders"]:
        target_dir = os.path.join(base_dir, folder_rel_path)
        folder_name = os.path.basename(folder_rel_path)
        
        stats_dir = os.path.join(target_dir, diff_binned_stats_dirname)
        if not os.path.exists(stats_dir):
            continue
            
        csv_files = glob.glob(os.path.join(stats_dir, "diff_binned_stats_*.csv"))
        for csv_path in csv_files:
            bin_name = os.path.basename(csv_path).replace(".csv", "")
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                
                # 差分ファイル名からタイムスタンプを抽出
                # 例: 20240117T124323_..._vs_20231208T153920_..._diff.csv
                # 最初の（新しい方の）タイムスタンプをプロット用に使用する
                df['timestamp'] = pd.to_datetime(
                    df['source_filename'].str.split('_').str[0],
                    format='%Y%m%dT%H%M%S'
                )
                # フォルダ識別子を追加
                df['folder'] = folder_name
                
                if bin_name not in all_data_by_bin:
                    all_data_by_bin[bin_name] = []
                all_data_by_bin[bin_name].append(df)
                
            except Exception as e:
                print(f"[エラー] {csv_path} の読み込み失敗: {e}")

    # 2. ビンごとに統合プロット作成
    stats_to_plot = [
        'sum_abs_diff',
        'mean_abs_diff',
        'max_abs_diff',
        'mean_diff',
        'std_diff',
        'variance_diff'
    ]
    
    for bin_name, df_list in all_data_by_bin.items():
        if not df_list:
            continue
            
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df = combined_df.sort_values(['folder', 'timestamp'])
        
        # サブプロットの作成
        fig = make_subplots(
            rows=len(stats_to_plot), 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=stats_to_plot
        )
        
        bin_info = f"{combined_df['bin_start'].iloc[0]}Hz - {combined_df['bin_end'].iloc[0]}Hz"
        
        # 各統計量ごとにプロット
        for i, stat in enumerate(stats_to_plot):
            if stat not in combined_df.columns:
                continue
                
            unique_folders = sorted(combined_df['folder'].unique())
            colors = px.colors.qualitative.Plotly
            
            for f_idx, folder in enumerate(unique_folders):
                f_df = combined_df[combined_df['folder'] == folder]
                
                fig.add_trace(
                    go.Scatter(
                        x=f_df['timestamp'],
                        y=f_df[stat],
                        name=folder,
                        mode='lines+markers',
                        legendgroup=folder,
                        showlegend=(i == 0), 
                        marker=dict(color=colors[f_idx % len(colors)]),
                        line=dict(color=colors[f_idx % len(colors)]),
                        hovertext=f_df['source_filename'],
                        hovertemplate="<b>%{hovertext}</b><br>Basis Time: %{x}<br>Value: %{y}<extra></extra>"
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            height=300 * len(stats_to_plot),
            title_text=f"Combined Binned Diff-Statistics Evolution ({bin_info})",
            xaxis_title="Basis Timestamp (Newer File)",
            hovermode="closest"
        )
        
        # HTMLとして保存
        save_path = os.path.join(output_dir, f"combined_{bin_name}.html")
        fig.write_html(save_path)
        print(f"  - 統合差分インタラクティブグラフを保存しました: {save_path}")

    print(f"\n=== 全ての処理が完了しました ===")

if __name__ == "__main__":
    main()
