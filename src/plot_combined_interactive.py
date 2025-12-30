"""
複数フォルダのバンド別統計量を統合し、フォルダごとに色分けしたインタラクティブなプロットを作成するスクリプト。
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
    binned_stats_dirname = dirs.get("binned_stats", "stats-binned")
    output_dirname = os.path.join("がたつき解析", "帯域別統計量の時間推移")
    
    output_dir = os.path.join(base_dir, output_dirname)
    html_output_dir = os.path.join(output_dir, "html") # HTML専用サブフォルダ
    
    os.makedirs(html_output_dir, exist_ok=True)
    
    print(f"=== 統合インタラクティブプロット生成開始 ===")
    
    # 1. データの収集
    # bin_name -> list of dataframes
    all_data_by_bin = {}
    
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
                if df.empty:
                    continue
                
                # タイムスタンプ抽出
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
        'mean_amplitude',
        'max_amplitude',
        'energy',
        'energy_ratio',
        'std_amplitude',
        'variance'
    ]
    
    # 正常データの基準統計、異常カウント、詳細ログを保持するためのリスト
    reference_stats_list = []
    anomaly_counts_map = {} # (timestamp, folder, stat) -> {'σ':0, '2σ':0, '3σ':0}
    anomaly_details = []
    
    for bin_name, df_list in all_data_by_bin.items():
        if not df_list:
            continue
            
        combined_df = pd.concat(df_list, ignore_index=True)
        # タイムスタンプ順にソート（必要に応じて）
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
        
        # 3. 正常データの統計量計算 (基準線用)
        normal_folders = config.get("normal_folders", [])
        normal_df = combined_df[combined_df['folder'].isin(normal_folders)]
        
        # 各統計量ごとにプロット
        for i, stat in enumerate(stats_to_plot):
            if stat not in combined_df.columns:
                continue
            
            # 基準線の追加
            if not normal_df.empty and stat in normal_df.columns:
                mu = normal_df[stat].mean()
                sigma = normal_df[stat].std()
                
                if not pd.isna(mu):
                    # 統計量の保存用にリストへ追加
                    reference_stats_list.append({
                        'bin_name': bin_name,
                        'statistic': stat,
                        'mean': mu,
                        'std': sigma
                    })
                    
                    # 閾値判定とカウント (2σ 追加)
                    if not pd.isna(sigma) and sigma > 0:
                        thresh1 = mu + sigma
                        thresh2 = mu + 2 * sigma
                        thresh3 = mu + 3 * sigma
                        
                        for _, row in combined_df.iterrows():
                            val = row[stat]
                            if pd.isna(val): continue
                            
                            ex_s = val > thresh1
                            ex_2s = val > thresh2
                            ex_3s = val > thresh3
                            
                            if ex_s:
                                key = (row['timestamp'], row['folder'], stat)
                                if key not in anomaly_counts_map:
                                    anomaly_counts_map[key] = {'σ': 0, '2σ': 0, '3σ': 0}
                                
                                if ex_s: anomaly_counts_map[key]['σ'] += 1
                                if ex_2s: anomaly_counts_map[key]['2σ'] += 1
                                if ex_3s: anomaly_counts_map[key]['3σ'] += 1
                                
                                # 詳細ログ
                                type_str = "3σ" if ex_3s else ("2σ" if ex_2s else "σ")
                                anomaly_details.append({
                                    'timestamp': row['timestamp'],
                                    'folder': row['folder'],
                                    'statistic': stat,
                                    'bin_name': bin_name,
                                    'value': val,
                                    'mean': mu,
                                    'std': sigma,
                                    'exceed_type': type_str
                                })

                    # 平均線の追加 (太め・濃いめ)
                    fig.add_shape(
                        type="line", line=dict(color="rgba(100, 100, 100, 0.6)", width=2),
                        xref=f"x{i+1}", yref=f"y{i+1}",
                        x0=combined_df['timestamp'].min(), x1=combined_df['timestamp'].max(),
                        y0=mu, y1=mu,
                        layer="below"
                    )
                    
                    if not pd.isna(sigma) and sigma > 0:
                        # sigma, 2sigma, 3sigma の追加
                        # 細め・薄めの実線
                        for mult in [1, 2, 3]:
                            for sign in [1, -1]:
                                y_val = mu + sign * mult * sigma
                                fig.add_shape(
                                    type="line", 
                                    line=dict(
                                        color="rgba(150, 150, 150, 0.3)", 
                                        width=1
                                    ),
                                    xref=f"x{i+1}", yref=f"y{i+1}",
                                    x0=combined_df['timestamp'].min(), x1=combined_df['timestamp'].max(),
                                    y0=y_val, y1=y_val,
                                    layer="below"
                                )

            # フォルダごとに色分けしてトレースを追加
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
                        showlegend=(i == 0), # 最初のサブプロットのみ凡例を表示
                        marker=dict(color=colors[f_idx % len(colors)]),
                        line=dict(color=colors[f_idx % len(colors)]),
                        hovertext=f_df['source_filename'],
                        hovertemplate="<b>%{hovertext}</b><br>Time: %{x}<br>Value: %{y}<extra></extra>"
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            height=300 * len(stats_to_plot),
            title_text=f"Combined Binned Statistics Evolution ({bin_info})",
            xaxis_title="Timestamp",
            hovermode="closest",
            # y軸の範囲を基準線に合わせて調整（オプション。現状はデータに合わせる）
        )
        
        # HTMLとして保存 (htmlサブフォルダへ)
        save_path = os.path.join(html_output_dir, f"{bin_name}.html")
        fig.write_html(save_path)
        print(f"  - 統合インタラクティブグラフを保存しました: {save_path}")

    # 4. 正常データの統計量を CSV として保存
    if reference_stats_list:
        ref_stats_df = pd.DataFrame(reference_stats_list)
        # 見やすいようにソート
        ref_stats_df = ref_stats_df.sort_values(['bin_name', 'statistic'])
        csv_save_path = os.path.join(output_dir, "reference_stats.csv")
        ref_stats_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
        print(f"  - 正常データのリファレンス統計量を保存しました: {csv_save_path}")

    # 5. 異常検知カウントの保存
    if anomaly_counts_map:
        summary_rows = []
        for (ts, fld, st), counts in anomaly_counts_map.items():
            summary_rows.append({
                'timestamp': ts,
                'folder': fld,
                'statistic': st,
                'exceed_sigma_count': counts['σ'],
                'exceed_2sigma_count': counts['2σ'],
                'exceed_3sigma_count': counts['3σ']
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(['timestamp', 'folder', 'statistic'])
        summary_path = os.path.join(output_dir, "anomaly_counts.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"  - 異常検知サマリー（超過数）を保存しました: {summary_path}")

    # 6. 異常詳細ログの保存
    if anomaly_details:
        details_df = pd.DataFrame(anomaly_details)
        details_df = details_df.sort_values(['timestamp', 'folder', 'statistic', 'bin_name'])
        details_path = os.path.join(output_dir, "anomaly_details.csv")
        details_df.to_csv(details_path, index=False, encoding='utf-8-sig')
        print(f"  - 異常詳細ログ（超過帯域）を保存しました: {details_path}")

    print(f"\n=== 全ての処理が完了しました ===")

if __name__ == "__main__":
    main()
