# 周波数解析データセット (Frequency Analysis Toolkit)

時系列データのバイナリ読み込みから、前処理（トリミング）、FFT解析、統計量算出、そして可視化までを一貫して行うPythonパイプラインツールです。

## 機能概要

- **データ変換**: 独自バイナリ形式 (`.bin`) から物理量CSVへの変換。
- **前処理**: 解析区間のトリミング機能。
- **周波数解析**: 高速フーリエ変換 (FFT) によるスペクトル解析。
- **統計解析**:
  - 基本統計量: 平均, 分散, 最大/最小値, エネルギーなど。
  - バンド別解析: 周波数帯域（10Hz区切り）ごとの詳細統計。
  - 差分解析: 異なる測定データ間のスペクトル差分の算出。
- **可視化**:
  - 時系列波形 & FFTスペクトルのプロット（上下2分割比較）。
  - 統計量の時系列推移グラフ。
  - FFT差分の棒グラフ。

## ディレクトリ構造

```
project_root/
├── main.py                # 実行用スクリプト
├── src/                   # ソースコード
│   ├── config.py          # 設定 (解析パラメータ,出力先)
│   ├── pipeline.py        # 処理フロー制御
│   ├── analyzer.py        # 解析ロジック (FFT, 統計)
│   ├── visualizer.py      # グラフ描画
│   └── io_handler.py      # 入出力・変換
├── CODE_EXPLANATION.md    # 詳細コード解説
└── README.md              # 本ファイル
```

## 必要要件

- Python 3.x
- 依存ライブラリ:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`

## 使い方

`main.py` を実行して各ステップを処理します。

### 全ステップの一括実行
```bash
python main.py --all
```

### 個別ステップの実行
```bash
# 1. バイナリ変換のみ
python main.py --convert

# 2. CSVトリミングのみ
python main.py --trim

# 3. 解析 (FFT & 統計計算)
python main.py --analyze

# 4. 差分解析
python main.py --diff

# 5. 可視化 (グラフ作成)
python main.py --visualize
```

## 設定の変更
`src/config.py` を編集することで、以下の設定を変更可能です。
- 解析対象フォルダ (`target_folders`)
- サンプリングレート
- FFT/時系列グラフのY軸範囲 (`timeseries_y_limit`, `fft_y_limit`)
- 解析対象の周波数範囲 など

## 出力されるデータ
各解析対象フォルダ内に、以下のディレクトリが生成されます。
- `csv_trim/`: トリミング済み時系列CSV
- `csv-fft/`: FFT解析結果CSV
- `stats-summary/`: 統計量サマリーCSV
- `fig-wave-fft/`: 時系列+FFTの複合グラフ画像
- `fig-statistics/`: 統計量の推移グラフ画像
- `fig-diff-summary/`: FFT差分グラフ画像
