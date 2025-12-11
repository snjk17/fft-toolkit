"""
バイナリ変換やCSV操作を含む、ファイル操作のための入出力処理モジュール。
"""

import os
import glob
import struct
import numpy as np
import pandas as pd
from pathlib import Path

class BinaryConverter:
    """
    バイナリデータ（float32）をCSVに変換するクラス
    """
    def __init__(self, input_dir):
        """
        初期化

        Args:
            input_dir (str): 入力バイナリファイルが格納されているディレクトリ
        """
        self.input_dir = input_dir
        # 'csv'サブディレクトリを出力先として設定（従来の動作）
        # 新しいパイプラインでは config["dirs"]["source_csv"] を尊重するべきかもしれませんが、
        # 変換メソッド内でこのパスを確認する方が良いでしょう。
        self.output_dir = os.path.join(self.input_dir, 'csv')
        os.makedirs(self.output_dir, exist_ok=True)

    def _read_binary_file(
        self,
        file_path,
        header_size=128
    ):
        """
        単一のバイナリファイルを読み込み、ヘッダーをスキップする

        Args:
            file_path (str): バイナリファイルのパス
            header_size (int): スキップするヘッダーのバイト数

        Returns:
            np.array or None: 読み込んだデータ配列
        """
        try:
            with open(file_path, 'rb') as f:
                # ヘッダーをスキップ
                f.seek(header_size)
                data = f.read()
                # 4バイトずつfloat32としてアンパック
                float_count = len(data) // 4
                data_array = struct.unpack(f'{float_count}f', data)
                return np.array(data_array)
        except Exception as e:
            print(f"ファイルの読み込みエラー {file_path}: {e}")
            return None

    def convert_all_to_individual_csvs(self, output_dir_override=None):
        """
        ディレクトリ内の各バイナリファイルを個別のCSVファイルに変換して保存する

        Args:
            output_dir_override (str): 出力先ディレクトリを上書きする場合指定

        Returns:
            list: 作成されたCSVファイルのフルパスのリスト
        """
        out_dir = output_dir_override if output_dir_override else self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        binary_files = sorted(glob.glob(os.path.join(self.input_dir, "*.bin")))
        created_csv_paths = []

        for file_path in binary_files:
            data = self._read_binary_file(file_path)
            if data is not None:
                # 異常な巨大値を除外
                data = data[np.abs(data) < 1e10]

                filename = os.path.basename(file_path)
                timestamp_str = filename.split('_')[0]

                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(
                        timestamp_str,
                        format='%Y%m%dT%H%M%S'
                    ),
                    'value': data,
                    'file_source': filename
                })

                # 出力ファイルパスを生成 (例: data.bin -> data.csv)
                csv_filename = os.path.splitext(filename)[0] + ".csv"
                output_path = os.path.join(out_dir, csv_filename)

                df.to_csv(output_path, index_label='sample_index')
                created_csv_paths.append(output_path)

        if not created_csv_paths:
            print("変換するデータがありません。")
            return []

        print(f"{len(created_csv_paths)} 個のCSVファイルを正常に作成しました（保存先: {out_dir}）。")
        return created_csv_paths


def trim_csv_files_in_folder(
    target_folder_path: str,
    skip_top: int,
    skip_bottom: int,
    output_folder_path: str = None
):
    """
    単一のフォルダパスを受け取り、その中の全CSVファイルをトリミングする。

    Args:
        target_folder_path (str): 処理対象のCSVフォルダのパス
        skip_top (int): 削除する先頭の行数
        skip_bottom (int): 削除する末尾の行数
        output_folder_path (str, optional): 出力先ディレクトリ。指定がない場合は親ディレクトリの兄弟フォルダ 'csv_trim' に作成。
    """

    print(f"\n--- トリミング処理: フォルダ '{target_folder_path}' ---")

    input_dir = Path(target_folder_path)

    # 1. 入力フォルダの検証
    if not input_dir.exists():
        print(f"  [エラー] フォルダが見つかりません: {input_dir}")
        return
    if not input_dir.is_dir():
        print(f"  [エラー] ディレクトリではありません: {input_dir}")
        return

    # 2. 出力フォルダのパス生成
    if output_folder_path:
         final_output_dir = Path(output_folder_path)
    else:
        # trim_csv.py のデフォルト動作
        parent_dir = input_dir.parent
        base_output_dir = parent_dir / 'csv_trim'
        # 入力が "csv" だった場合、"csv_trim/csv" のような構造を維持したい場合、
        # または単に csv_trim に入れる。
        # main.py の config "source_csv": "csv_trim/csv" に基づくと、特定の構造が必要。
        # しかし trim_csv.py は base_output_dir / input_dir.name としていました。
        final_output_dir = base_output_dir / input_dir.name

    # 3. 出力フォルダの作成
    try:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [情報] 出力先: {final_output_dir}")
    except OSError as e:
        print(f"  [エラー] 出力ディレクトリの作成に失敗しました: {final_output_dir} ({e})")
        return

    processed_count = 0
    skipped_count = 0

    csv_files = list(input_dir.glob('*.csv'))

    if not csv_files:
        print("  [情報] .csv ファイルが見つかりませんでした。")
        return

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, header=0)
            total_rows = len(df)

            if total_rows <= skip_top + skip_bottom:
                skipped_count += 1
                continue

            trimmed_df = df.iloc[skip_top:-skip_bottom]

            output_file_path = final_output_dir / file_path.name
            trimmed_df.to_csv(output_file_path, index=False, header=True)
            processed_count += 1

        except pd.errors.EmptyDataError:
            print(f"    [スキップ] '{file_path.name}': 空のファイルです。")
            skipped_count += 1
        except Exception as e:
            print(f"    [エラー] '{file_path.name}': 処理エラー ({e})")
            skipped_count += 1

    print(f"  --- トリミング完了: {processed_count} 成功, {skipped_count} スキップ/エラー ---")
    return final_output_dir
