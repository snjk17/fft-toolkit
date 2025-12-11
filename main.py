import os
import argparse
from src.config import DEFAULT_CONFIG as CONFIG
from src.pipeline import (
    run_conversion,
    run_trimming,
    run_basic_analysis,
    run_diff_analysis,
    run_visualization
)

def main():
    """
    メイン実行関数。
    コマンドライン引数に応じて、リスト内の各フォルダに指定された処理を実行する。
    """
    parser = argparse.ArgumentParser(
        description="時系列データ解析パイプライン (Refactored)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--convert", action="store_true",
        help="ステップ1-1: バイナリ変換 (.bin -> .csv)"
    )
    parser.add_argument(
        "--trim", action="store_true",
        help="ステップ1-2: CSVトリミング"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="ステップ2: 基本解析 (FFT計算 -> 統計量算出)"
    )
    parser.add_argument(
        "--diff", action="store_true",
        help="ステップ3: 差分解析 (FFT差分計算 -> 差分統計用)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="ステップ4: 可視化 (個別グラフ, 推移グラフ, 複合グラフ)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="全ステップ順次実行"
    )
    
    # 設定ファイルの場所に関するヘルプ
    parser.epilog = "注: 設定ファイルは src/config.py にあります。"

    args = parser.parse_args()

    if not any([args.convert, args.trim, args.analyze, args.diff, args.visualize, args.all]):
        parser.print_help()
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ターゲットフォルダは相対パスまたは絶対パスを許可
    target_folders = CONFIG["target_folders"]
    absolute_target_folders = []
    
    for f in target_folders:
        if os.path.isabs(f):
            absolute_target_folders.append(f)
        else:
            absolute_target_folders.append(os.path.join(base_dir, f))

    for folder_path in absolute_target_folders:
        if not os.path.isdir(folder_path):
            print(f"警告: ディレクトリが見つかりません: {folder_path} スキップします。")
            continue

        print(f"\n{'='*70}\n処理対象フォルダ: {folder_path}\n{'='*70}")

        if args.convert or args.all:
            run_conversion(folder_path, CONFIG)
            
        if args.trim or args.all:
            run_trimming(folder_path, CONFIG)

        if args.analyze or args.all:
            run_basic_analysis(folder_path, CONFIG)

        if args.diff or args.all:
            run_diff_analysis(folder_path, CONFIG)

        if args.visualize or args.all:
            run_visualization(folder_path, CONFIG)

    print(f"\n{'='*70}\n全ての処理が完了しました。\n{'='*70}")


if __name__ == '__main__':
    main()
