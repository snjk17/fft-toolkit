"""
時系列データ解析パイプラインの設定モジュール。
"""

DEFAULT_CONFIG = {
    # 解析対象のデータフォルダをリストで指定
    # 注: これらはプロジェクトルートからの相対パスまたは絶対パスです
    "target_folders": [
        "がたつき解析/11170101_交換前",
        "がたつき解析/11170101_交換後",
        "がたつき解析/11170201",
        "がたつき解析/11170301",
    ],

    # 解析パラメータ
    "sampling_rate": 1000,  # サンプリングレート (Hz)
    # 時系列グラフの縦軸範囲（Noneの場合は自動調整）
    # "timeseries_y_limit": {"min": -2000, "max": 2000},
    "timeseries_y_limit": None,
    "fft_y_limit": 2000,    # FFTグラフの縦軸の最大値（Noneの場合は自動調整）
    "fft_x_limit": {"min": 0, "max": 500},  # FFTグラフの横軸の範囲(Hz)
    # FFT統計量を計算する周波数範囲(Hz)
    "fft_stats_freq_range": {"min": 30, "max": 130},
    # 10Hz区切りの統計解析設定
    "fft_binned_stats": {
        "bin_size": 10,
        "max_freq": 500
    },
    "rounding_digits": 4,   # CSV保存時の小数点以下の丸め桁数

    # FFT差分分析のパラメータ
    "fft_diff_freq_range": {"min": 0, "max": 500},  # FFT差分を計算する周波数範囲(Hz)
    # FFTを集約する周波数ビンのサイズ(Hz) - 1Hzに変更
    "fft_diff_bin_size": 1,
    "plot_diff_bins": [],               # 時系列プロットしたい周波数ビン(Hz) - 不要なため空リスト

    # 個別差分グラフのY軸範囲（Noneの場合は自動調整）
    "plot_diff_summary_y_limit": {"min": -5000, "max": 5000},

    # 可視化パラメータ
    "statistics_to_plot": [  # 時系列変化をプロットする統計量
        'mean', 'std', 'variance', 'max', 'min',
        'dominant_frequency', 'max_amplitude'
    ],

    # 生成されるディレクトリ名
    "dirs": {
        "source_csv": "csv_trim",  # バイナリから変換されたCSVの保存先
        "stats_summary": "stats-summary",  # 統計サマリーCSVの保存先
        "fft_csv": "csv-fft",             # FFT結果CSVの保存先
        "diff_fft_summary": "diff-fft-summary",      # FFT差分結果の保存先
        "diff_stats_summary": "diff-stats-summary",  # 差分統計サマリーの保存先
        "fig_timeseries": "fig-timeseries",  # 時系列グラフの保存先
        "fig_fft": "fig-fft",             # FFTグラフの保存先
        "fig_statistics": "fig-statistics",    # 統計量グラフの保存先
        "fig_diff_evolution": "fig-fft-diff",  # 差分時系列グラフの保存先
        "fig_diff_summary": "fig-diff-summary",    # 個別差分グラフの保存先
        "fig_combined": "fig-combined",        # 複合グラフの保存先
        "fig_time_fft": "fig-time-fft",        # 時系列+FFT複合グラフの保存先
    },

    # 生成されるファイル名
    "filenames": {
        "stats_summary": "fft_statistics.csv",  # 統計サマリーファイル名
        "diff_stats_summary": "diff_statistics_summary.csv",  # 差分統計サマリーファイル名
    },

    # トリミングパラメータ
    "trim": {
        "rows_from_top": 4500,
        "rows_from_bottom": 500
    }
}
