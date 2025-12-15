"""
時系列、FFT、および差分解析を可視化するためのモジュール。
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataVisualizer:
    """
    解析済みのデータを受け取り、可視化（グラフ作成）を行うクラス。
    """

    def __init__(self):
        # sns.set_style("whitegrid")  # seabornへの依存を削除
        plt.style.use('seaborn-v0_8-whitegrid')  # 代替スタイル

    def plot_timeseries(self, df, title="Time Series Data", save_path=None, y_limit=None):
        """
        時系列データを可視化し、ファイルに保存する。
        """
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['value'], alpha=0.8)
        plt.title(title, fontsize=16)
        plt.xlabel("Sample Index")
        plt.ylabel("Value")

        if y_limit and y_limit.get("min") is not None and y_limit.get("max") is not None:
            plt.ylim(y_limit["min"], y_limit["max"])

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"  - 時系列グラフを保存しました: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_fft(self, df, title="FFT Spectrum", save_path=None, x_limit=None, y_limit=None):
        """
        FFTデータを可視化し、ファイルに保存する。
        """
        plt.figure(figsize=(15, 7))
        plt.plot(df['frequency_hz'], df['amplitude'],
                 alpha=0.8, color='orange')
        plt.title(title, fontsize=16)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")

        if x_limit and x_limit.get("min") is not None and x_limit.get("max") is not None:
            plt.xlim(x_limit["min"], x_limit["max"])

        if y_limit:
            if isinstance(y_limit, dict):
                if y_limit.get("min") is not None and y_limit.get("max") is not None:
                    plt.ylim(y_limit["min"], y_limit["max"])
            elif isinstance(y_limit, (int, float)):
                plt.ylim(0, y_limit)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"  - FFTグラフを保存しました: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_time_and_fft_combined(self, time_df, fft_df, title="Time Series & FFT", save_path=None,
                                   timeseries_y_limit=None, fft_x_limit=None, fft_y_limit=None):
        """
        時系列データ（上段）とFFTデータ（下段）を縦に並べてプロットする。
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle(title, fontsize=18)

        # 上段: 時系列
        ax1.plot(time_df.index, time_df['value'], alpha=0.8, color='blue')
        ax1.set_title("Time Series", fontsize=14)
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Value")
        ax1.grid(True, linestyle='--', alpha=0.6)

        if timeseries_y_limit and timeseries_y_limit.get("min") is not None and timeseries_y_limit.get("max") is not None:
            ax1.set_ylim(timeseries_y_limit["min"], timeseries_y_limit["max"])

        # 下段: FFT
        ax2.plot(fft_df['frequency_hz'], fft_df['amplitude'],
                 alpha=0.8, color='orange')
        ax2.set_title("FFT Spectrum", fontsize=14)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, linestyle='--', alpha=0.6)

        if fft_x_limit and fft_x_limit.get("min") is not None and fft_x_limit.get("max") is not None:
            ax2.set_xlim(fft_x_limit["min"], fft_x_limit["max"])

        if fft_y_limit:
            if isinstance(fft_y_limit, dict):
                if fft_y_limit.get("min") is not None and fft_y_limit.get("max") is not None:
                    ax2.set_ylim(fft_y_limit["min"], fft_y_limit["max"])
            elif isinstance(fft_y_limit, (int, float)):
                ax2.set_ylim(0, fft_y_limit)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # タイトル分のスペース確保

        if save_path:
            plt.savefig(save_path)
            print(f"  - 時系列+FFT複合グラフを保存しました: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_statistics_evolution(self, stats_csv_path, output_fig_path):
        """
        統計情報サマリーCSVを読み込み、各統計量の時系列変化をグラフ化して保存する。
        """
        try:
            df = pd.read_csv(stats_csv_path)
        except FileNotFoundError:
            print(f"エラー: 統計サマリーが見つかりません: {stats_csv_path}")
            return

        try:
            # Assumes file_source structure matches expected pattern
            df['timestamp'] = pd.to_datetime(
                df['file_source'].str.split('_').str[0],
                format='%Y%m%dT%H%M%S'
            )
            df = df.set_index('timestamp').sort_index()
        except Exception as e:
            print(f"統計推移のためのタイムスタンプ解析中にエラー: {e}")
            return

        stats_to_plot = [
            'mean',
            'std',
            'variance',
            'max',
            'min'
        ]

        fig, axes = plt.subplots(
            nrows=len(stats_to_plot),
            ncols=1,
            figsize=(15, 20),
            sharex=True
        )
        fig.suptitle('Evolution of Statistics Over Time', fontsize=20, y=0.99)
        colors = plt.cm.viridis(np.linspace(0, 1, len(stats_to_plot)))

        for i, stat in enumerate(stats_to_plot):
            if stat in df.columns:
                ax = axes[i]
                ax.plot(
                    df.index,
                    df[stat],
                    marker='o',
                    linestyle='-',
                    color=colors[i])
                ax.set_ylabel(stat)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        axes[-1].set_xlabel('Timestamp')
        fig.autofmt_xdate()
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        try:
            plt.savefig(output_fig_path)
            print(f"  - 統計推移グラフを保存しました: {output_fig_path}")
        except Exception as e:
            print(f"グラフの保存エラー: {e}")

        plt.close()

    def plot_fft_diff_evolution(self, diff_csv_dir, output_dir, plot_bins):
        """
        FFT差分CSV群を読み込み、指定された周波数ビンの時間変化をプロットする。
        """
        if not os.path.exists(diff_csv_dir):
            return

        file_list = sorted(
            [f for f in os.listdir(diff_csv_dir) if f.lower().endswith('_diff.csv')])
        if not file_list:
            return

        evolution_data = {bin_val: [] for bin_val in plot_bins}

        for filename in file_list:
            try:
                file_path = os.path.join(diff_csv_dir, filename)
                df = pd.read_csv(file_path)

                for bin_val in plot_bins:
                    diff_row = df[df['frequency_bin'] == bin_val]
                    if not diff_row.empty:
                        amplitude_diff = diff_row['amplitude_diff'].iloc[0]
                        evolution_data[bin_val].append(amplitude_diff)
                    else:
                        evolution_data[bin_val].append(0)

            except Exception:
                continue

        num_data_points = len(evolution_data[plot_bins[0]]) if plot_bins else 0
        if num_data_points == 0:
            return

        os.makedirs(output_dir, exist_ok=True)

        for bin_val in plot_bins:
            if evolution_data[bin_val]:
                plt.figure(figsize=(15, 7))
                indices = range(num_data_points)
                plt.plot(
                    indices,
                    evolution_data[bin_val],
                    marker='o',
                    linestyle='-',
                )
                plt.title(
                    f'FFT Difference Evolution for {bin_val}-{bin_val+10} Hz Bin'
                )
                plt.xlabel('File Index')
                plt.ylabel('Average Amplitude Difference')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()

                output_path = os.path.join(
                    output_dir,
                    f'diff_evolution_{bin_val}Hz.png')
                plt.savefig(output_path)
                print(f"  - 差分推移グラフ ({bin_val}Hz) を保存しました: {output_path}")
                plt.close()

    def plot_single_fft_diff(
        self,
        diff_csv_path,
        save_path,
        title,
        y_limit=None
    ):
        """
        単一のFFT差分CSVファイルを読み込み、棒グラフとしてプロットする。
        """
        try:
            df = pd.read_csv(diff_csv_path)
        except FileNotFoundError:
            return

        if df.empty:
            return

        plt.figure(figsize=(15, 7))
        bin_size = df['frequency_bin'].iloc[1] - \
            df['frequency_bin'].iloc[0] if len(df) > 1 else 10

        plt.bar(
            df['frequency_bin'],
            df['amplitude_diff'],
            width=bin_size * 0.9,
            align='edge')

        plt.title(title, fontsize=16)
        plt.xlabel('Frequency Bin (Hz)')
        plt.ylabel('Average Amplitude Difference')

        if y_limit and y_limit.get("min") is not None and y_limit.get("max") is not None:
            plt.ylim(y_limit["min"], y_limit["max"])

        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()

    def plot_combined_evolution(self, stats_df, diff_df, output_dir, target_stats=['energy', 'max_amplitude']):
        """
        基本統計量の推移（上段）と差分推移（下段）を結合したグラフを作成する。
        """
        if stats_df.empty or diff_df.empty:
            print("  [警告] 複合グラフ作成のためのデータが不足しています。")
            return

        os.makedirs(output_dir, exist_ok=True)

        # タイムスタンプでソート・インデックス化
        try:
            # file_source カラムからタイムスタンプを抽出する共通ロジック
            # diff_dfのsource_fileは "Current_vs_Previous_diff.csv" のような形式で、
            # Current側のタイムスタンプを採用するのが適切。
            stats_df['timestamp'] = pd.to_datetime(
                stats_df['file_source'].str.split('_').str[0],
                format='%Y%m%dT%H%M%S',
                errors='coerce'
            )
            stats_df = stats_df.dropna(subset=['timestamp']).set_index(
                'timestamp').sort_index()

            # diff_dfのsource_file例: "2023..._vs_2023..._diff.csv"
            # 最初のYYYYMMDD...を取得する
            diff_df['timestamp'] = pd.to_datetime(
                diff_df['source_file'].str.split('_').str[0],
                format='%Y%m%dT%H%M%S',
                errors='coerce'
            )
            diff_df = diff_df.dropna(subset=['timestamp']).set_index(
                'timestamp').sort_index()

        except Exception as e:
            print(f"  [エラー] タイムスタンプ解析エラー: {e}")
            return

        # 共通の期間（または結合）を取得
        common_indices = stats_df.index.intersection(diff_df.index)
        if len(common_indices) == 0:
            print("  [警告] 統計データと差分データで共通のタイムスタンプがありません。")
            return

        # プロット用データ
        s_df = stats_df.loc[common_indices]
        d_df = diff_df.loc[common_indices]

        # target_stats 1つにつき1枚の画像を生成
        for stat in target_stats:
            if stat not in s_df.columns:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

            # 上段: 基本統計量
            ax1.plot(s_df.index, s_df[stat], marker='o', color='b', label=stat)
            ax1.set_title(f"Statistics Evolution: {stat}", fontsize=14)
            ax1.set_ylabel(stat)
            ax1.grid(True, linestyle='--')
            ax1.legend()

            # 下段: 差分統計量 (Mean Abs Diff と Total Abs Change を表示)
            if 'mean_abs_diff' in d_df.columns:
                ax2.plot(
                    d_df.index,
                    d_df['mean_abs_diff'],
                    marker='x',
                    color='r',
                    label='Mean Abs Diff'
                )
            if 'max_diff' in d_df.columns:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(
                    d_df.index,
                    d_df['max_diff'],
                    marker='.',
                    color='orange',
                    linestyle='--',
                    label='Max Diff'
                )
                ax2_twin.set_ylabel('Max Diff')
                # legendの統合は少し面倒だが、ここでは簡易的に

            ax2.set_title(
                "Difference Evolution (Changes from previous)",
                fontsize=14
            )
            ax2.set_xlabel("Timestamp")
            ax2.set_ylabel("Mean Abs Difference")
            ax2.grid(True, linestyle='--')
            ax2.legend(loc='upper left')

            plt.tight_layout()
            save_path = os.path.join(
                output_dir, f"combined_evolution_{stat}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"  - 複合グラフを保存しました: {save_path}")
