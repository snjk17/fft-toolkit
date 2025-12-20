"""
各フォルダの stats-binned 内にある統計量CSVから時系列プロットを生成するスクリプト。
"""

import os
import glob
from config import DEFAULT_CONFIG
from visualizer import DataVisualizer

def main():
    config = DEFAULT_CONFIG
    visualizer = DataVisualizer()
    
    # プロジェクトルートディレクトリ（srcの親）を特定
    # i:\マイドライブ\WALC\PoC\data\周波数解析データセット
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dirs = config["dirs"]
    binned_stats_dirname = dirs.get("binned_stats", "stats-binned")
    output_fig_dirname = dirs.get("fig_binned_stats", "fig-stats-binned")
    
    print(f"=== バンド別統計量プロット生成開始 ===")
    
    for folder_rel_path in config["target_folders"]:
        target_dir = os.path.join(base_dir, folder_rel_path)
        if not os.path.exists(target_dir):
            print(f"[警告] フォルダが見つかりません: {target_dir}")
            continue
            
        stats_dir = os.path.join(target_dir, binned_stats_dirname)
        output_dir = os.path.join(target_dir, output_fig_dirname)
        
        if not os.path.exists(stats_dir):
            print(f"[スキップ] 統計ディレクトリがありません: {stats_dir}")
            continue
            
        print(f"\n解析中: {folder_rel_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        # binned_stats_*.csv を探す
        csv_files = glob.glob(os.path.join(stats_dir, "binned_stats_*.csv"))
        
        if not csv_files:
            print(f"  - CSVファイルが見つかりません")
            continue
            
        for csv_path in sorted(csv_files):
            filename = os.path.basename(csv_path)
            # binned_stats_0Hz_10Hz.csv -> binned_stats_0Hz_10Hz.png
            fig_filename = filename.replace(".csv", ".png")
            output_fig_path = os.path.join(output_dir, fig_filename)
            
            visualizer.plot_binned_statistics_evolution(csv_path, output_fig_path)

    print(f"\n=== 全ての処理が完了しました ===")

if __name__ == "__main__":
    main()
