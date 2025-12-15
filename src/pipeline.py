"""
解析ワークフローを調整するパイプラインモジュール。
精緻化されたパイプライン構成（前処理、基本解析、差分解析、可視化）を実装。
"""

import os
import glob
import pandas as pd
from .config import DEFAULT_CONFIG
from .io_handler import BinaryConverter, trim_csv_files_in_folder
from .analyzer import UnifiedAnalyzer
from .visualizer import DataVisualizer

# --- 1. 前処理 (Preparation) ---


def run_conversion(target_dir, config):
    """
    ステップ1-1: バイナリ変換を実行。
    """
    print(
        f"\n=== ステップ1-1: バイナリ変換 (Binary Conversion) [{os.path.basename(target_dir)}] ===")
    converter = BinaryConverter(target_dir)
    # デフォルトでは target_dir/csv に保存される
    converter.convert_all_to_individual_csvs()


def run_trimming(target_dir, config):
    """
    ステップ1-2: CSVトリミングを実行。
    """
    print(
        f"\n=== ステップ1-2: CSVトリミング (Trimming) [{os.path.basename(target_dir)}] ===")

    dirs = config["dirs"]
    trim_params = config["trim"]

    # 入力: target_dir/csv (バイナリ変換の出力先)
    csv_input_dir = os.path.join(target_dir, 'csv')

    if not os.path.exists(csv_input_dir):
        print("  [エラー] 入力CSVディレクトリが見つかりません。先にバイナリ変換を行ってください。")
        return

    # 出力先: target_dir/csv_trim/csv (configの source_csv に合わせる)
    trim_output_dir = os.path.join(target_dir, dirs["source_csv"])

    trim_csv_files_in_folder(
        target_folder_path=csv_input_dir,
        skip_top=trim_params["rows_from_top"],
        skip_bottom=trim_params["rows_from_bottom"],
        output_folder_path=trim_output_dir
    )


# --- 2. 基本解析 (Basic Analysis) ---
def run_basic_analysis(target_dir, config):
    """
    基本解析ステップ: FFT計算と基本統計量の算出。
    """
    print(
        f"\n=== ステップ2: 基本解析 (Basic Analysis) [{os.path.basename(target_dir)}] ===")

    dirs = config["dirs"]
    filenames = config["filenames"]
    sampling_rate = config["sampling_rate"]

    # 入力: トリミング済みCSV
    source_csv_dir = os.path.join(target_dir, dirs["source_csv"])
    if not os.path.exists(source_csv_dir):
        print(f"  [エラー] 入力ディレクトリが見つかりません: {source_csv_dir}")
        return

    # 出力先
    fft_output_dir = os.path.join(target_dir, dirs["fft_csv"])
    stats_summary_dir = os.path.join(target_dir, dirs["stats_summary"])
    os.makedirs(fft_output_dir, exist_ok=True)
    os.makedirs(stats_summary_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(source_csv_dir, "*.csv")))
    all_stats = []

    # バンド別統計量を集約するためのリスト
    all_binned_stats_list = []

    print(f"  [情報] {len(csv_files)} ファイルを解析します...")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            if 'value' not in df.columns:
                continue

            analyzer = UnifiedAnalyzer(df)

            # A. FFT計算と保存
            fft_save_path = os.path.join(
                fft_output_dir, f"{os.path.splitext(filename)[0]}_fft.csv")
            analyzer.save_fft_to_csv(
                sampling_rate=sampling_rate, save_path=fft_save_path)

            # B. 統計量計算 (基本統計 + FFT統計)
            stats = analyzer.calculate_statistics().to_dict()
            stats['file_source'] = filename

            freq_range = config.get("fft_stats_freq_range")
            fft_stats = analyzer.get_fft_stats_dict(
                sampling_rate=sampling_rate,
                freq_min=freq_range.get("min"),
                freq_max=freq_range.get("max")
            )
            stats.update(fft_stats)
            all_stats.append(stats)

            # C. バンド別統計量の計算 (10Hz区切り)
            if "fft_binned_stats" in config:
                binned_params = config["fft_binned_stats"]
                binned_df = analyzer.get_binned_stats(
                    bin_size=binned_params.get("bin_size", 10),
                    max_freq=binned_params.get("max_freq", 500),
                    sampling_rate=sampling_rate
                )

                # 集約用にファイル名を追加してリストに格納
                binned_df.insert(0, 'source_filename', filename)
                all_binned_stats_list.append(binned_df)

        except Exception as e:
            print(f"  [エラー] {filename} の解析失敗: {e}")

    # D. 統計サマリー保存
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        if 'file_source' in summary_df.columns:
            cols = summary_df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('file_source')))
            summary_df = summary_df[cols]

        summary_path = os.path.join(
            stats_summary_dir, filenames["stats_summary"])
        summary_df.round(config["rounding_digits"]).to_csv(
            summary_path, index=False, encoding='utf-8-sig')
        print(f"  [完了] 統計サマリーを保存: {summary_path}")

    # E. バンド別統計量の集約保存
    if all_binned_stats_list:
        binned_stats_dir = os.path.join(
            target_dir, config["dirs"].get("binned_stats", "stats-binned"))
        os.makedirs(binned_stats_dir, exist_ok=True)

        # 全データを結合
        full_binned_df = pd.concat(all_binned_stats_list, ignore_index=True)

        # ビンごとにグルーピングして保存
        # bin_start, bin_end でグループ化
        grouped = full_binned_df.groupby(['bin_start', 'bin_end'])

        print(f"  [情報] {len(grouped)} 個の周波数帯域の統計ファイルを保存します...")

        for (bin_start, bin_end), group in grouped:
            # ファイル名: binned_stats_0Hz_10Hz.csv
            save_filename = f"binned_stats_{int(bin_start)}Hz_{int(bin_end)}Hz.csv"
            save_path = os.path.join(binned_stats_dir, save_filename)

            # 不要な列を整理（bin_start, bin_endはファイル名にあるので削除してもいいが、念のため残すか、indexにする）
            # ここではそのまま保存する

            group.round(config["rounding_digits"]).to_csv(
                save_path, index=False, encoding='utf-8-sig')

        print(f"  [完了] バンド別統計量を保存: {binned_stats_dir}")


# --- 3. 差分解析 (Differential Analysis) ---
def run_diff_analysis(target_dir, config):
    """
    差分解析ステップ: FFT差分計算、差分統計量、およびバンド別差分統計の算出。
    """
    print(
        f"\n=== ステップ3: 差分解析 (Differential Analysis) [{os.path.basename(target_dir)}] ===")

    dirs = config["dirs"]
    freq_range = config["fft_diff_freq_range"]
    bin_size = config["fft_diff_bin_size"]  # 1Hz (Updated)

    fft_csv_dir = os.path.join(target_dir, dirs["fft_csv"])
    diff_summary_dir = os.path.join(target_dir, dirs["diff_fft_summary"])
    diff_stats_dir = os.path.join(target_dir, dirs["diff_stats_summary"])
    diff_binned_stats_dir = os.path.join(target_dir, dirs.get(
        "diff_binned_stats", "diff-stats-binned"))  # New

    if not os.path.exists(fft_csv_dir):
        print(f"  [エラー] FFTディレクトリが見つかりません: {fft_csv_dir}")
        return

    os.makedirs(diff_summary_dir, exist_ok=True)
    os.makedirs(diff_stats_dir, exist_ok=True)
    os.makedirs(diff_binned_stats_dir, exist_ok=True)

    analyzer = UnifiedAnalyzer(None)
    file_list = sorted(glob.glob(os.path.join(fft_csv_dir, "*_fft.csv")))

    previous_binned_fft = None
    previous_filename = None
    diff_stats_list = []

    # バンド別差分統計量を集約するためのリスト
    all_binned_diff_stats_list = []

    print(f"  [情報] {len(file_list)} ファイル間の差分を計算します (Bin Size: {bin_size}Hz)...")

    for file_path in file_list:
        filename = os.path.basename(file_path)
        try:
            fft_df = pd.read_csv(file_path)
            # ビニング (またはそのまま扱う)
            current_binned_fft = analyzer.get_binned_fft_from_df(
                fft_df, freq_range, bin_size)

            if previous_binned_fft is not None:
                # 差分計算
                diff_df = pd.merge(
                    current_binned_fft, previous_binned_fft,
                    on='frequency_bin', how='outer', suffixes=('_current', '_previous')
                ).fillna(0)

                # 単純差分 (Current - Previous)
                diff_df['amplitude_diff'] = diff_df['amplitude_sum_current'] - \
                    diff_df['amplitude_sum_previous']

                # ファイル名生成
                prev_base = previous_filename.replace('_fft.csv', '')
                curr_base = filename.replace('_fft.csv', '')
                diff_filename = f"{curr_base}_vs_{prev_base}_diff.csv"
                diff_save_path = os.path.join(diff_summary_dir, diff_filename)

                diff_df[['frequency_bin', 'amplitude_diff']].to_csv(
                    diff_save_path, index=False, encoding='utf-8-sig')

                # 1. 差分統計量の計算 (全体)
                d_stats = UnifiedAnalyzer.calculate_statistics_from_diff_csv(
                    diff_save_path)
                if d_stats:
                    diff_stats_list.append(d_stats)

                # 2. バンド別差分統計 (10Hz区切り集約)
                # diff_df を使って計算する
                # diff_df は 'frequency_bin' (1Hz刻み) と 'amplitude_diff' を持っている
                binned_diff_df = UnifiedAnalyzer.aggregate_binned_diff_stats(
                    diff_df, bin_size=10, max_freq=config["fft_binned_stats"]["max_freq"]
                )

                # 集約用にソースファイル名を追加してリストに格納
                binned_diff_df.insert(0, 'source_filename', diff_filename)
                all_binned_diff_stats_list.append(binned_diff_df)

            previous_binned_fft = current_binned_fft
            previous_filename = filename

        except Exception as e:
            print(f"  [エラー] {filename} の差分計算失敗: {e}")

    # 差分統計サマリー保存
    if diff_stats_list:
        summary_stats_df = pd.DataFrame(diff_stats_list)
        summary_stats_path = os.path.join(
            diff_stats_dir, config["filenames"]["diff_stats_summary"])
        summary_stats_df.round(config["rounding_digits"]).to_csv(
            summary_stats_path, index=False, encoding='utf-8-sig')
        print(f"  [完了] 差分統計サマリーを保存: {summary_stats_path}")

    # 3. バンド別差分統計量の集約保存
    if all_binned_diff_stats_list:
        full_diff_binned_df = pd.concat(
            all_binned_diff_stats_list, ignore_index=True)

        # ビンごとにグルーピングして保存
        grouped = full_diff_binned_df.groupby(['bin_start', 'bin_end'])

        print(f"  [情報] {len(grouped)} 個の周波数帯域の差分統計ファイルを保存します...")

        for (bin_start, bin_end), group in grouped:
            # ファイル名: diff_binned_stats_0Hz_10Hz.csv
            save_filename = f"diff_binned_stats_{int(bin_start)}Hz_{int(bin_end)}Hz.csv"
            save_path = os.path.join(diff_binned_stats_dir, save_filename)

            group.round(config["rounding_digits"]).to_csv(
                save_path, index=False, encoding='utf-8-sig')

        print(f"  [完了] バンド別差分統計量を保存: {diff_binned_stats_dir}")


# --- 4. 可視化 (Visualization) ---
def run_visualization(target_dir, config):
    """
    可視化ステップ: 個別グラフ、時系列推移、複合グラフの作成。
    """
    print(
        f"\n=== ステップ4: 可視化 (Visualization) [{os.path.basename(target_dir)}] ===")

    dirs = config["dirs"]
    visualizer = DataVisualizer()

    # 4-1. 個別グラフ (時系列, FFT)
    # これらは数が多くなるので、必要に応じてスキップオプションがあっても良いが、今回は全て実行
    source_csv_dir = os.path.join(target_dir, dirs["source_csv"])
    fig_ts_dir = os.path.join(target_dir, dirs["fig_timeseries"])
    os.makedirs(fig_ts_dir, exist_ok=True)

    if os.path.exists(source_csv_dir):
        print("  - 個別時系列グラフ作成中...")
        for f in glob.glob(os.path.join(source_csv_dir, "*.csv")):
            try:
                df = pd.read_csv(f, index_col='sample_index')
                save_path = os.path.join(
                    fig_ts_dir, f"{os.path.splitext(os.path.basename(f))[0]}.png")
                visualizer.plot_timeseries(
                    df, title=os.path.basename(f), save_path=save_path,
                    y_limit=config.get("timeseries_y_limit")
                )
            except Exception:
                pass

    fft_csv_dir = os.path.join(target_dir, dirs["fft_csv"])
    fig_fft_dir = os.path.join(target_dir, dirs["fig_fft"])
    fig_time_fft_dir = os.path.join(target_dir, dirs.get(
        "fig_time_fft", "fig-time-fft"))  # Configにキーがない場合のフォールバック
    os.makedirs(fig_fft_dir, exist_ok=True)
    os.makedirs(fig_time_fft_dir, exist_ok=True)

    if os.path.exists(fft_csv_dir):
        print("  - 個別FFTグラフ & 複合グラフ(Time+FFT)作成中...")
        fft_files = glob.glob(os.path.join(fft_csv_dir, "*_fft.csv"))

        for fft_file in fft_files:
            try:
                fft_df = pd.read_csv(fft_file)
                base_name = os.path.basename(fft_file).replace('_fft.csv', '')

                # 1. 個別FFTグラフ
                save_path_fft = os.path.join(
                    fig_fft_dir, f"{base_name}_fft.png")
                visualizer.plot_fft(
                    fft_df,
                    title=f"FFT Spectrum: {base_name}",
                    save_path=save_path_fft,
                    x_limit=config.get("fft_x_limit"),
                    y_limit=config.get("fft_y_limit")
                )

                # 2. Time + FFT 複合グラフ
                # 対応する時系列CSVを探す
                # 時系列ファイルの拡張子は.csv, 名前は base_name と一致するはず
                time_csv_path = os.path.join(
                    source_csv_dir, f"{base_name}.csv")

                if os.path.exists(time_csv_path):
                    time_df = pd.read_csv(
                        time_csv_path, index_col=None)  # index_colは適宜
                    # index_col='sample_index' がある場合はそれを使うが、
                    # 先ほどの plot_timeseries では index_col='sample_index' で読んでいた。
                    # ここでは汎用的に読む。

                    save_path_combined = os.path.join(
                        fig_time_fft_dir, f"{base_name}_time_fft.png")
                    visualizer.plot_time_and_fft_combined(
                        time_df,
                        fft_df,
                        title=f"Time Series & FFT: {base_name}",
                        save_path=save_path_combined,
                        timeseries_y_limit=config.get("timeseries_y_limit"),
                        fft_x_limit=config.get("fft_x_limit"),
                        fft_y_limit=config.get("fft_y_limit")
                    )

            except Exception as e:
                print(
                    f"  [エラー] FFTグラフ作成失敗 ({os.path.basename(fft_file)}): {e}")

    stats_summary_path = os.path.join(
        target_dir, dirs["stats_summary"], config["filenames"]["stats_summary"])
    fig_stats_path = os.path.join(
        target_dir, dirs["fig_statistics"], "statistics_evolution.png")
    os.makedirs(os.path.dirname(fig_stats_path), exist_ok=True)

    if os.path.exists(stats_summary_path):
        print("  - 統計推移グラフ作成中...")
        visualizer.plot_statistics_evolution(
            stats_summary_path, fig_stats_path)

    # 4-3. 差分推移グラフ (Diff Evolution - Specific Bins)
    diff_summary_dir = os.path.join(target_dir, dirs["diff_fft_summary"])
    fig_diff_dir = os.path.join(target_dir, dirs["fig_diff_evolution"])

    if os.path.exists(diff_summary_dir):
        print("  - 差分推移グラフ（特定ビン）作成中...")
        visualizer.plot_fft_diff_evolution(
            diff_summary_dir, fig_diff_dir, config["plot_diff_bins"])

        # 個別差分グラフ
        fig_diff_ind_dir = os.path.join(target_dir, dirs["fig_diff_summary"])
        os.makedirs(fig_diff_ind_dir, exist_ok=True)
        for f in glob.glob(os.path.join(diff_summary_dir, "*.csv")):
            visualizer.plot_single_fft_diff(
                f, os.path.join(
                    fig_diff_ind_dir, f"{os.path.splitext(os.path.basename(f))[0]}.png"),
                title=os.path.basename(f), y_limit=config.get("plot_diff_summary_y_limit")
            )

    # 4-4. 複合グラフ (Combined Evolution)
    # 統計サマリーと差分統計サマリーを使用
    diff_stats_path = os.path.join(
        target_dir, dirs["diff_stats_summary"], config["filenames"]["diff_stats_summary"])
    fig_combined_dir = os.path.join(target_dir, dirs["fig_combined"])

    if os.path.exists(stats_summary_path) and os.path.exists(diff_stats_path):
        print("  - 複合グラフ（時系列＋差分）作成中...")
        try:
            stats_df = pd.read_csv(stats_summary_path)
            diff_stats_df = pd.read_csv(diff_stats_path)
            visualizer.plot_combined_evolution(
                stats_df, diff_stats_df, fig_combined_dir,
                target_stats=['energy', 'max_amplitude', 'spectral_entropy']
            )
        except Exception as e:
            print(f"  [エラー] 複合グラフ作成失敗: {e}")
