"""
時系列データから統計量とFFTを計算するための解析モジュール。
DataProcessor, AnalysisFFT, time_evolution.py の機能を統合しました。
"""

import os
import numpy as np
import pandas as pd
from scipy import fft
from scipy import stats as scipy_stats
from scipy.signal import butter, filtfilt


class UnifiedAnalyzer:
    """
    データフレームを受け取り、数値解析（統計、FFT）を行うクラス。
    """

    def __init__(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            # 必要であればダミーの空データフレームでの初期化を許可する
            # 前回の差分解析ではダミーのデータフレームが使用されていた: pd.DataFrame({'value': []})
            if dataframe is None:
                self.df = pd.DataFrame({'value': []})
            else:
                raise TypeError("入力は pandas DataFrame である必要があります。")
        else:
            self.df = dataframe

    # --- 基本統計量 ---
    def calculate_statistics(self):
        """
        データの基本統計情報を計算し、Seriesとして返す。
        """
        if 'value' not in self.df.columns:
            return pd.Series(dtype=float)

        stats = self.df['value'].describe()
        return stats

    # --- FFT 計算 ---
    def calculate_fft(self, sampling_rate=1000):
        """
        FFTを計算し、周波数と振幅のデータを返す。
        """
        if 'value' not in self.df.columns or self.df.empty:
            return np.array([]), np.array([])

        data = self.df['value'].values
        n = len(data)
        yf = fft.fft(data)
        xf = fft.fftfreq(n, 1 / sampling_rate)

        # 正の周波数成分のみを抽出
        positive_mask = xf > 0
        return xf[positive_mask], np.abs(yf[positive_mask])

    def bandpass_filter(self, lowcut, highcut, sampling_rate=1000, order=4):
        """
        バンドパスフィルタを適用してフィルタ済み信号を返す。
        """
        if 'value' not in self.df.columns or self.df.empty:
            return np.array([])

        data = self.df['value'].values
        if len(data) == 0:
            return data

        nyq = 0.5 * sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        if low <= 0 or high >= 1 or low >= high:
            raise ValueError('lowcut/highcut の指定が不正です')

        b, a = butter(order, [low, high], btype='band')
        try:
            y = filtfilt(b, a, data)
        except ValueError:
            y = data
        return y

    def save_fft_to_csv(self, sampling_rate=1000, save_path=None):
        """
        FFTを計算し、結果をCSVファイルに保存する。
        """
        frequencies, amplitudes = self.calculate_fft(sampling_rate)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            fft_df = pd.DataFrame({
                'frequency_hz': frequencies,
                'amplitude': amplitudes
            })
            fft_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"  - FFTデータをCSVに保存しました: {save_path}")

        return frequencies, amplitudes

    # --- FFT 統計量 (AnalysisFFT からの移行) ---
    def mean_amplitude(self, amplitudes, sampling_rate=1000):
        if amplitudes.size == 0:
            return float('nan')
        return float(np.mean(amplitudes))

    def std_amplitude(self, amplitudes, sampling_rate=1000):
        if amplitudes.size == 0:
            return float('nan')
        return float(np.std(amplitudes))

    def mean_power(self, amplitudes, sampling_rate=1000):
        if amplitudes.size == 0:
            return float('nan')
        power = amplitudes ** 2
        return float(np.mean(power))

    def std_power(self, amplitudes, sampling_rate=1000):
        if amplitudes.size == 0:
            return float('nan')
        power = amplitudes ** 2
        return float(np.std(power))

    def spectral_entropy(self, amplitudes, sampling_rate=1000):
        if amplitudes.size == 0:
            return float('nan')
        power_spectrum = amplitudes ** 2
        power_spectrum = power_spectrum[power_spectrum > 0]
        if power_spectrum.size == 0:
            return 0.0
        normalized_ps = power_spectrum / np.sum(power_spectrum)
        entropy = -np.sum(normalized_ps * np.log2(normalized_ps))
        return float(entropy)

    def get_fft_stats_dict(self, sampling_rate=1000, freq_min=None, freq_max=None):
        """
        主要統計量を dict で返す。
        """
        freqs, amps = self.calculate_fft(sampling_rate)

        # 周波数範囲でデータをフィルタリング
        if freq_min is not None or freq_max is not None:
            freq_mask = np.ones_like(freqs, dtype=bool)
            if freq_min is not None:
                freq_mask &= (freqs >= freq_min)
            if freq_max is not None:
                freq_mask &= (freqs <= freq_max)
            freqs = freqs[freq_mask]
            amps = amps[freq_mask]

        if amps.size == 0:
            # フィルタで全てのデータが除外された場合はデフォルト/空の統計量を返す
            return {
                'mean_amplitude': 0, 'std_amplitude': 0,
                'mean_power': 0, 'std_power': 0, 'spectral_entropy': 0,
                'dominant_frequency': 0
            }

        stats = {
            'mean_amplitude': self.mean_amplitude(amps),
            'std_amplitude': self.std_amplitude(amps),
            'mean_power': self.mean_power(amps),
            'std_power': self.std_power(amps),
            'spectral_entropy': self.spectral_entropy(amps),
        }

        # 最大振幅の周波数 (Dominant Frequency)
        max_amp_idx = np.argmax(amps)
        stats['dominant_frequency'] = freqs[max_amp_idx]

        # Max/Min/Median 振幅 (Class_Analysis にあったが有用なので計算)
        stats['max_amplitude'] = np.max(amps)
        stats['min_amplitude'] = np.min(amps)
        stats['median_amplitude'] = np.median(amps)
        stats['total_power'] = np.sum(amps ** 2)

        return stats

    def save_fft_stats_to_csv(self, save_path, sampling_rate=1000, source_filename=None, freq_min=None, freq_max=None):
        stats_dict = self.get_fft_stats_dict(sampling_rate, freq_min, freq_max)
        stats_df = pd.DataFrame([stats_dict])

        if source_filename:
            stats_df.insert(0, 'source_filename', source_filename)

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        stats_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"  - FFT統計量をCSVに保存しました: {save_path}")

    def get_binned_fft_from_df(self, df, freq_range, bin_size):
        """
        既存のFFTデータフレームを集約する (差分解析用)。
        """
        min_freq = freq_range.get('min', 0)
        max_freq = freq_range.get('max', df['frequency_hz'].max())
        freq_mask = (df['frequency_hz'] >= min_freq) & (
            df['frequency_hz'] <= max_freq)
        filtered_df = df[freq_mask].copy()

        if filtered_df.empty:
            return pd.DataFrame(columns=['frequency_bin', 'amplitude_sum'])

        filtered_df['frequency_bin'] = (
            filtered_df['frequency_hz'] // bin_size) * bin_size
        binned_df = filtered_df.groupby('frequency_bin')[
            'amplitude'].sum().reset_index()
        binned_df.rename(columns={'amplitude': 'amplitude_sum'}, inplace=True)
        return binned_df

    def get_binned_stats(self, bin_size=10, max_freq=500, sampling_rate=1000):
        """
        0Hzからmax_freqまで、bin_sizeごとに区切って統計量を計算する。
        正規化エネルギー（全体に対する寄与率）も計算する。
        """
        freqs, amps = self.calculate_fft(sampling_rate)

        # 全周波数帯域の総エネルギー（0Hz〜ナイキスト周波数全体）
        total_energy = np.sum(amps ** 2)

        bins = []
        current_freq = 0

        while current_freq < max_freq:
            next_freq = current_freq + bin_size

            # 範囲内のデータを抽出
            mask = (freqs >= current_freq) & (freqs < next_freq)
            bin_amps = amps[mask]

            stats = {
                'bin_start': current_freq,
                'bin_end': next_freq,
            }
            if bin_amps.size > 0:
                bin_energy = np.sum(bin_amps ** 2)
                stats['mean_amplitude'] = np.mean(bin_amps)
                stats['max_amplitude'] = np.max(bin_amps)
                stats['energy'] = bin_energy
                stats['energy_ratio'] = bin_energy / \
                    total_energy if total_energy > 0 else 0
                stats['std_amplitude'] = np.std(bin_amps)
                stats['variance'] = np.var(bin_amps)
            else:
                stats['mean_amplitude'] = 0
                stats['max_amplitude'] = 0
                stats['energy'] = 0
                stats['energy_ratio'] = 0
                stats['std_amplitude'] = 0
                stats['variance'] = 0

            bins.append(stats)
            current_freq = next_freq

        return pd.DataFrame(bins)

    @staticmethod
    def aggregate_binned_diff_stats(diff_df, bin_size=10, max_freq=500):
        """
        1Hzごとの差分データ(diff_df)を受け取り、より大きなbin_size(デフォルト10Hz)で集約統計量を計算する。
        diff_dfは 'frequency_bin' と 'amplitude_diff' を持つことを想定。
        """
        if diff_df is None or diff_df.empty:
            return pd.DataFrame()

        # frequency_bin がインデックスやカラムにある場合を考慮
        data = diff_df.copy()

        bins = []
        current_freq = 0

        while current_freq < max_freq:
            next_freq = current_freq + bin_size

            # 範囲抽出
            mask = (data['frequency_bin'] >= current_freq) & (
                data['frequency_bin'] < next_freq)
            subset = data[mask]

            stats = {
                'bin_start': current_freq,
                'bin_end': next_freq
            }

            if not subset.empty:
                diffs = subset['amplitude_diff'].values
                abs_diffs = np.abs(diffs)

                stats['sum_abs_diff'] = np.sum(abs_diffs)
                stats['mean_abs_diff'] = np.mean(abs_diffs)
                stats['max_abs_diff'] = np.max(abs_diffs)
                stats['mean_diff'] = np.mean(diffs)
                stats['std_diff'] = np.std(diffs)
                stats['variance_diff'] = np.var(diffs)
            else:
                stats['sum_abs_diff'] = 0
                stats['mean_abs_diff'] = 0
                stats['max_abs_diff'] = 0
                stats['mean_diff'] = 0
                stats['std_diff'] = 0
                stats['variance_diff'] = 0

            bins.append(stats)
            current_freq = next_freq

        return pd.DataFrame(bins)

    # --- 時間変化解析ロジック (time_evolution.py からの移行) ---
    @staticmethod
    def calculate_statistics_from_fft_csv(file_path, freq_min, freq_max):
        """
        既存のFFT CSVファイルから統計量を計算するための静的メソッド。
        時間変化解析ステップで使用されます。
        """
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            return None

        freq_col = next((c for c in df.columns if 'freq' in c.lower()), None)
        amp_col = next((c for c in df.columns if 'amp' in c.lower()), None)

        if not freq_col or not amp_col:
            return None

        freq_mask = (df[freq_col] >= freq_min) & (df[freq_col] <= freq_max)
        filtered_df = df[freq_mask]

        if filtered_df.empty:
            return {
                'source_file': os.path.basename(file_path),
                'energy': 0, 'spectral_entropy': 0,
                'mean_amplitude': 0, 'std_amplitude': 0,
                'max_amplitude': 0, 'second_max_amplitude': 0,
                'dominant_frequency': 0,
            }

        amplitudes = filtered_df[amp_col].values
        frequencies = filtered_df[freq_col].values

        energy = np.sum(amplitudes ** 2)

        # スペクトルエントロピーの計算
        total_power = energy
        if total_power > 0:
            normalized_ps = (amplitudes ** 2) / total_power
            spectral_entropy = - \
                np.sum(normalized_ps * np.log2(normalized_ps + 1e-12))
        else:
            spectral_entropy = 0

        second_max = np.sort(amplitudes)[-2] if len(amplitudes) >= 2 else 0

        return {
            'source_file': os.path.basename(file_path),
            'energy': energy,
            'spectral_entropy': spectral_entropy,
            'mean_amplitude': np.mean(amplitudes),
            'std_amplitude': np.std(amplitudes),
            'max_amplitude': np.max(amplitudes),
            'second_max_amplitude': second_max,
            'dominant_frequency': frequencies[np.argmax(amplitudes)] if len(amplitudes) > 0 else 0,
        }

    @staticmethod
    def calculate_statistics_from_diff_csv(file_path):
        """
        差分CSVファイルから統計量を計算する。
        """
        try:
            df = pd.read_csv(file_path)
            if 'amplitude_diff' not in df.columns:
                return None
        except Exception:
            return None

        diffs = df['amplitude_diff'].values
        abs_diffs = np.abs(diffs)

        return {
            'source_file': os.path.basename(file_path),
            'mean_diff': np.mean(diffs),
            'mean_abs_diff': np.mean(abs_diffs),
            'max_diff': np.max(diffs),
            'min_diff': np.min(diffs),
            'std_diff': np.std(diffs),
            'total_abs_change': np.sum(abs_diffs)
        }
