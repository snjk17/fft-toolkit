# コードベース解説ドキュメント

このドキュメントでは、周波数解析データセットプロジェクトのコード構造と、各モジュールの役割について解説します。

## ディレクトリ構造

```
project_root/
├── main.py                # エントリーポイント (CLI)
├── src/                   # ソースコードパッケージ
│   ├── __init__.py
│   ├── config.py          # 設定ファイル
│   ├── io_handler.py      # 入出力・前処理 (バイナリ変換, トリミング)
│   ├── analyzer.py        # 解析ロジック (FFT, 統計, バンド解析)
│   ├── pipeline.py        # 処理フローの定義 (各ステップの実行関数)
│   └── visualizer.py      # 可視化・グラフ生成
└── CODE_EXPLANATION.md    # 本ドキュメント
```

---

## 各モジュールの詳細

### 1. `main.py` (エントリーポイント)
解析パイプラインの実行を管理するスクリプトです。コマンドライン引数を受け取り、`src/pipeline.py` 内の適切な関数を呼び出します。

- **主な機能**:
    - コマンドライン引数 (`--convert`, `--trim`, `--analyze` など) の解析。
    - `src/config.py` で定義された「解析対象フォルダ」へのループ処理。
- **使用方法**: `python main.py --all` ですべての工程を実行できます。

### 2. `src/config.py` (設定)
プロジェクト全体の設定パラメータを一元管理します。

- **主な設定項目**:
    - `target_folders`: 解析対象のフォルダパス（リスト）。
    - `sampling_rate`: サンプリング周波数 (基本 1000Hz)。
    - `timeseries_y_limit`: 時系列グラフのY軸固定範囲。
    - `fft_binned_stats`: 10Hz区切りの統計解析の設定 (`bin_size`, `max_freq`)。
    - `fft_diff_bin_size`: 差分解析時の周波数解像度 (1Hz)。
    - ディレクトリ名・ファイル名の定義。

### 3. `src/io_handler.py` (入出力・前処理)
ファイルの読み書きや形式変換を担当します。

- **`BinaryConverter` クラス**:
    - `.bin` 形式のバイナリデータを読み込み、物理量に変換して `.csv` として保存します。
- **`trim_csv_files_in_folder` 関数**:
    - CSVファイルの先頭・末尾から指定行数を削除（トリミング）し、解析用データを準備します。

### 4. `src/analyzer.py` (解析ロジック)
データ解析の中核となるクラス `UnifiedAnalyzer` を提供します。

- **`calculate_fft`**: FFT（高速フーリエ変換）を実行し、周波数と振幅を返します。
- **`calculate_statistics`**: 基本的な統計量（平均、分散、最大・最小値など）を計算します。
    - *Note*: 以前含まれていた「歪度」「尖度」は現在除外されています。
- **`get_binned_stats` (重要)**:
    - 指定された周波数バンド（例: 0-10Hz, 10-20Hz...）ごとの情報を抽出します。
    - パイプライン内で、複数ファイルの統計情報を **帯域ごとに1つのCSVファイル (`binned_stats_XXHz_YYHz.csv`)** に集約する仕組みに変更されました。
- **`aggregate_binned_diff_stats`**:
    - 差分データに対しても同様に、1Hzごとの差分をより大きなビンに集約し、全比較データを帯域ごとのCSVに保存します。

### 5. `src/pipeline.py` (パイプライン)
解析の各工程（ステップ）を定義し、連携させます。

- **`run_conversion` / `run_trimming`**: 前処理（変換・トリミング）を実行。
- **`run_basic_analysis`**:
    - FFT計算、基本統計量に加え、**周波数帯域ごとの集約統計CSV** を生成・保存します。
- **`run_diff_analysis`**:
    - 異なるデータ（交換前vs交換後など）のFFT差分を計算します。
    - **周波数帯域ごとの差分集約統計CSV** を生成・保存します。
- **`run_visualization`**: 可視化処理を実行します。

### 6. `src/visualizer.py` (可視化)
グラフの描画を担当します。

- **`plot_time_and_fft_combined`**:
    - 上段に時系列波形、下段にFFTスペクトルを配置した複合グラフを作成します (`fig-time-fft` フォルダ)。
- **`plot_fft`**:
    - 個別のFFTスペクトルグラフを作成します (`fig-fft` フォルダ)。
- **推移グラフ**: `plot_statistics_evolution` で統計量（平均、分散、エネルギー等）の時系列変化を描画。
- **FFT差分グラフ**: `plot_single_fft_diff` でファイル間の周波数差分を棒グラフで表示します。
