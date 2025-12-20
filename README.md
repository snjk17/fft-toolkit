# 周波数解析データセット (Frequency Analysis Toolkit)

時系列データのバイナリ読み込みから、前処理（トリミング）、FFT解析、統計量算出、そして可視化までを一貫して行うPythonパイプラインツールです。
各解析結果を周波数帯域（ビン）ごとに集計し、時間変化やフォルダ間の比較を視覚的に把握できます。

## 1. 機能概要

### A. データ処理・解析
- **データ変換**: 独自バイナリ形式 (`.bin`) から物理量CSVへの変換。
- **前処理**: 解析区間のトリミング機能。
- **周波数解析**: 高速フーリエ変換 (FFT) によるスペクトル解析。
- **統計解析**:
  - **基本統計量**: 平均, 分散, 最大/最小値, エネルギー, 最大振幅など。
  - **バンド別解析**: 周波数帯域（10Hz区切り）ごとの詳細統計。
  - **差分解析**: 異なる測定データ間のスペクトル差分の算出。

### B. 可視化
- **静的グラフ (PNG)**:
  - 時系列波形 & FFTスペクトルの比較、統計量の時系列推移、FFT差分棒グラフ。
  - 周波数帯域ごとの統計推移プロット。
- **インタラクティブグラフ (HTML)**:
  - 複数フォルダのデータを統合した色分けグラフ。マウスホバーによるファイル名の確認が可能。

---

## 2. ディレクトリ構造とコード解説

### プロジェクト構成
```
project_root/
├── main.py                    # 解析パイプラインの実行用エントリーポイント
├── src/                       # ソースコード
│   ├── config.py              # 全体の設定 (解析対象、パラメータ、出力先)
│   ├── pipeline.py            # 各解析ステップ（変換〜可視化）のフロー制御
│   ├── analyzer.py            # 解析ロジックの中核 (FFT, 統計計算)
│   ├── visualizer.py          # グラフ描画（Matplotlibベースの静的プロット）
│   ├── io_handler.py          # ファイル入出力と変換（バイナリ読込、トリミング）
│   ├── plot_binned_stats.py   # 【追加】帯域別統計画像の自動生成スクリプト
│   ├── plot_combined_interactive.py # 【追加】統合インタラクティブプロット生成
│   └── plot_diff_combined_interactive.py # 【追加】統合差分インタラクティブプロット生成
├── fig-combined-interactive/  # 【出力】統合HTMLグラフの保存先
├── fig-diff-combined-interactive/ # 【出力】統合差分HTMLグラフの保存先
└── README.md                  # 本ドキュメント
```

### 各ファイルの役割詳細

- **`main.py`**: パイプライン全体の司令塔です。引数に応じて、バイナリ変換から可視化までの各工程を順番に実行します。
- **`src/config.py`**: 解析対象フォルダ、サンプリングレート、グラフのY軸範囲など、すべての設定を一括管理します。
- **`src/analyzer.py` (`UnifiedAnalyzer` クラス)**: データ解析の本体です。FFT計算、複数ファイルの統計量を帯域ごとに集計する機能を提供します。
- **`src/visualizer.py` (`DataVisualizer` クラス)**: Matplotlibを使用した各種静的グラフの生成を担当します。
- **`src/plot_binned_stats.py`**: `DataVisualizer` を利用し、全解析フォルダ内の「バンド別統計量」から時系列画像を自動生成します。
- **`src/plot_combined_interactive.py`**: `Plotly` ライブラリを使用し、全フォルダを横断して比較できるHTMLグラフを作成します。
- **`src/plot_diff_combined_interactive.py`**: 同様に、差分解析結果（diff-stats-binned）を統合したインタラクティブプロットを作成します。

---

## 3. 使い方（コマンド一覧）

### パイプラインの一括実行
```bash
python main.py --all
```

### ステップごとの実行
1.  **バイナリ変換**: `python main.py --convert`
2.  **トリミング**: `python main.py --trim`
3.  **解析（FFT/統計）**: `python main.py --analyze`
4.  **差分解析**: `python main.py --diff`
5.  **基本可視化**: `python main.py --visualize`

### 【NEW】追加された可視化ツールの実行
- **帯域別統計画像の生成**: `python src/plot_binned_stats.py`
  - 各フォルダ内にPNG画像を作成します。
- **統合インタラクティブグラフの生成**: `python src/plot_combined_interactive.py`
  - ルート配下の `fig-combined-interactive/` にHTMLを作成します。
- **統合差分インタラクティブグラフの生成**: `python src/plot_diff_combined_interactive.py`
  - ルート配下の `fig-diff-combined-interactive/` にHTMLを作成します。

---

## 4. 出力データと確認方法

各対象フォルダの配下に以下の成果物が生成されます。

- `csv_trim/`: 前処理済みCSV
- `csv-fft/`: FFT解析結果CSV
- `stats-summary/`: 全体統計サマリー
- `stats-binned/`: 10Hz区切りの帯域別統計CSV
- `fig-timeseries/` & `fig-fft/`: 基本波形画像
- `fig-stats-binned/`: 帯域別の統計推移画像（PNG）

### インタラクティブグラフ (HTML) の確認方法
ルートディレクトリの以下のフォルダ内のHTMLファイルをブラウザで開いてください。
- `fig-combined-interactive/`: 通常の統計量推移の比較
- `fig-diff-combined-interactive/`: 差分統計量推移の比較

**確認・操作ポイント**:
- **複数比較**: 全フォルダのデータが色分けされて1つのグラフに表示されます。
- **ファイル確認**: 各点にカーソルを合わせると、元データ名が表示されます。
- **表示切り替え**: 右側の凡例をクリックして、特定フォルダの表示ON/OFFを切り替えられます。
