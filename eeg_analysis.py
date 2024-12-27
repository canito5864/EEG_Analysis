import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import pywt
from numpy.fft import fft
import matplotlib.pyplot as plt
import mne
from sklearn.decomposition import PCA

def main():
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    #주파수 필터링링
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    #웨이블릿 변환
    def wavelet_transform(data, wavelet="db4", level=2):
        coeffs = pywt.wavedec(data, wavelet, level=level)
        reconstructed_signal = pywt.waverec(coeffs, wavelet)
        return coeffs, reconstructed_signal

    #푸리에 변환
    def fourier_transform(data, fs):
        data = data - np.mean(data)
        N = len(data)
        fft_result = fft(data)
        freqs = np.fft.fftfreq(N, d=1/fs)
        return freqs[:N // 2], np.abs(fft_result[:N // 2])

    #두 신호의 차이 계산
    def calculate_difference(signal1, signal2):
        return np.abs(signal1 - signal2)

    #topomap 시각화
    def plot_topomap(data, pos, title='EEG Topomap', cmap='Spectral'):
        fig, ax = plt.subplots()
        mne.viz.plot_topomap(data, pos[:, :2], axes=ax, show=False, contours=0, cmap=cmap)
        ax.set_title(title)
        st.pyplot(fig)

    #주파수 대역 분리 및 신호 전처리 한 번에 적용
    def apply_processing(data, fs, preprocessing, selected_bands, frequency_bands, filter_first):
            fft_results = None

            if filter_first == "Frequency Band Filtering → Preprocessing":
                filtered_signals = []
                for band in selected_bands:
                    lowcut, highcut = frequency_bands[band]
                    filtered_signal = butter_bandpass_filter(data, lowcut, highcut, fs)
                    filtered_signals.append(filtered_signal)
                data = np.sum(filtered_signals, axis=0)
        
                if "Fourier Transform" in preprocessing:
                    fft_results = fourier_transform(data, fs)
                if "Wavelet Transform" in preprocessing:
                    _, data = wavelet_transform(data)
        
            elif filter_first == "Preprocessing → Frequency Band Filtering":
                if "Fourier Transform" in preprocessing:
                    fft_results = fourier_transform(data, fs)
                if "Wavelet Transform" in preprocessing:
                    _, data = wavelet_transform(data)
                    
                filtered_signals = []
                for band in selected_bands:
                    lowcut, highcut = frequency_bands[band]
                    filtered_signal = butter_bandpass_filter(data, lowcut, highcut, fs)
                    filtered_signals.append(filtered_signal)
                data = np.sum(filtered_signals, axis=0)
        
            return data, fft_results
    
    st.title('EEG 뇌파 분석')
    
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
    
    if uploaded_file is not None:
        raw_eeg = pd.read_csv(uploaded_file, skiprows=[0])
    
        eeg_columns = ['EEG.AF3', 'EEG.AF4', 'EEG.T7', 'EEG.T8', 'EEG.Pz']
        raw_eeg = raw_eeg[eeg_columns]
    
        mapping = {
            'EEG.AF3': 'AF3',
            'EEG.AF4': 'AF4',
            'EEG.T7': 'T7',
            'EEG.T8': 'T8',
            'EEG.Pz': 'Pz'
        }
    
        raw_eeg = raw_eeg.rename(columns=mapping)
    
        sampling_interval = 0.00780995 
        fs = 1 / sampling_interval
        time = np.arange(0, len(raw_eeg) * sampling_interval, sampling_interval)
    
        analysis_type = st.selectbox('분석할 방법을 선택하세요', (
            'Single Electrode',
            'Comparison',
            'Topomap Visualization',
            'PCA'
        ))

        frequency_bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-45 Hz)': (30, 45)
        }

        if not(analysis_type == 'Comparison'):
            selected_bands = st.multiselect('분석할 주파수 대역을 선택하세요', list(frequency_bands.keys()))

        preprocessing = st.multiselect(
            '적용할 전처리 방법을 선택하세요',
            ['Wavelet Transform', 'Fourier Transform']
        )

        filter_first = st.radio(
            "Select Preprocessing Order",
            ["Frequency Band Filtering → Preprocessing", "Preprocessing → Frequency Band Filtering"]
        )
        
        num_ranges = st.number_input("선택할 구간의 개수를 입력하세요", min_value=0, max_value=100, value=1)
        time_ranges = []

        for i in range(num_ranges):
            start_time = st.text_input(f"구간 {i+1} 시작 시간 (초)", "0.0")
            end_time = st.text_input(f"구간 {i+1} 종료 시간 (초)", str(len(time) * sampling_interval))
            try:
                start_time = float(start_time)
                end_time = float(end_time)
                if 0 <= start_time < end_time <= len(time) * sampling_interval:
                    time_ranges.append((start_time, end_time))
                else:
                    st.warning(f"구간 {i+1}: 유효한 시간을 입력하세요.")
            except ValueError:
                st.warning(f"구간 {i+1}: 올바른 숫자 형식으로 시간을 입력하세요.")
    
        if analysis_type == 'Single Electrode':
            st.subheader('단일 전극 분석')
    
            electrode = st.selectbox('분석할 전극을 선택하세요', raw_eeg.columns)
            raw_data = raw_eeg[electrode]

            processed_data, fft_results = apply_processing(
                raw_data, fs, preprocessing, selected_bands, frequency_bands, filter_first
            )

            st.subheader(f'{electrode} 채널의 전처리된 EEG 신호')
            fig, ax = plt.subplots()
            if len(time)<len(processed_data):
                ax.plot(time, processed_data[:len(time)], label=f'{electrode} - 필터링된 신호', linewidth=0.5)
            else:
                ax.plot(time[:len(processed_data)], processed_data, label=f'{electrode} - 필터링된 신호', linewidth=0.5)
                
            for start_time, end_time in time_ranges:
                ax.axvspan(start_time, end_time, color='lightgray', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            plt.figure(figsize=(100,10))
            st.pyplot(fig)

            if fft_results:
                freqs, magnitude = fft_results
                band_powers = {}
                for band_name, (low, high) in frequency_bands.items():
                    # 주파수 범위 내 값 필터링
                    band_indices = np.where((freqs >= low) & (freqs <= high))
                    band_power = np.sum(magnitude[band_indices] ** 2)  # 에너지 계산
                    band_powers[band_name] = band_power
            
                # 막대 그래프 시각화
                st.subheader('주파수 대역별 신호 에너지')
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(band_powers.keys(), band_powers.values(), color='gray')
                ax.set_ylabel('Band Power')
                ax.set_title('Frequency Band Power Distribution')
                st.pyplot(fig)
                
                st.subheader("Fourier Transform")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(freqs, magnitude, label="FFT Magnitude")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude")
                ax.legend()
                st.pyplot(fig)

            for idx, (start_time, end_time) in enumerate(time_ranges):
                start_idx = int(start_time / sampling_interval)
                end_idx = int(end_time / sampling_interval)

                st.subheader(f'구간 {idx + 1}: {start_time}초 - {end_time}초')
                fig, ax = plt.subplots()
                ax.plot(time[start_idx:end_idx], processed_data[start_idx:end_idx], label=f'{electrode} - 필터링된 신호', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)

                if fft_results:
                    # 구간별 푸리에 변환
                    segment_signal = processed_data[start_idx:end_idx]
                    segment_time = time[start_idx:end_idx]
                    
                    if len(segment_signal) > 0:
                        segment_fft_results = fourier_transform(segment_signal, fs)
                        if segment_fft_results:
                            freqs, magnitude = segment_fft_results
                            band_powers = {}
                            for band_name, (low, high) in frequency_bands.items():
                                # 주파수 범위 내 값 필터링
                                band_indices = np.where((freqs >= low) & (freqs <= high))
                                band_power = np.sum(magnitude[band_indices] ** 2)  # 에너지 계산
                                band_powers[band_name] = band_power
                        
                            # 막대 그래프 시각화
                            st.subheader('주파수 대역별 신호 에너지')
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.bar(band_powers.keys(), band_powers.values(), color='gray')
                            ax.set_ylabel('Band Power')
                            ax.set_title('Frequency Band Power Distribution')
                            st.pyplot(fig)
                            
                            st.subheader(f'구간 {idx + 1}: Fourier Transform ({start_time}초 - {end_time}초)')
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(freqs, magnitude, label="FFT Magnitude")
                            ax.set_xlabel("Frequency (Hz)")
                            ax.set_ylabel("Magnitude")
                            ax.legend()
                            st.pyplot(fig)

        elif analysis_type == 'Comparison':
            st.subheader('비교 분석')
        
            # 비교 모드 선택
            comparison_mode = st.selectbox('비교할 유형을 선택하세요', ['Electrode Comparison', 'Frequency Band Comparison'])
        
            if comparison_mode == 'Electrode Comparison':
                st.subheader('전극 간 비교 분석')

                #주파수 선택
                selected_bands = st.multiselect('분석할 주파수 대역을 선택하세요', list(frequency_bands.keys()))
        
                # 두 전극 선택
                electrode1 = st.selectbox('첫 번째 전극을 선택하세요', raw_eeg.columns, index=0)
                electrode2 = st.selectbox('두 번째 전극을 선택하세요', raw_eeg.columns, index=1)
        
                # 데이터 로드
                data1 = raw_eeg[electrode1]
                data2 = raw_eeg[electrode2]
        
                # 신호 처리
                processed_data1, _ = apply_processing(data1, fs, preprocessing, selected_bands, frequency_bands, filter_first)
                processed_data2, _ = apply_processing(data2, fs, preprocessing, selected_bands, frequency_bands, filter_first)
        
                # 시각화
                st.subheader(f'{electrode1}과 {electrode2}의 신호 비교')
                fig, ax = plt.subplots()
                ax.plot(time[:len(processed_data1)], processed_data1[:len(time)], label=f'{electrode1} - 필터링된 신호', linewidth=0.5)
                ax.plot(time[:len(processed_data2)], processed_data2[:len(time)], label=f'{electrode2} - 필터링된 신호', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)
        
                # 신호 차이 계산
                difference = calculate_difference(processed_data1, processed_data2)
                st.subheader(f'{electrode1}과 {electrode2}의 신호 차이')
                fig, ax = plt.subplots()
                ax.plot(time[:len(difference)], difference[:len(time)], label='Difference', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude Difference')
                ax.legend()
                st.pyplot(fig)
        
            elif comparison_mode == 'Frequency Band Comparison':
                st.subheader('주파수 대역 간 비교')
        
                # 전극 선택
                electrode = st.selectbox('분석할 전극을 선택하세요', raw_eeg.columns)
        
                # 주파수 대역 선택
                st.subheader('비교할 주파수 대역을 선택하세요')
                frequency_bands = {
                    'Delta (0.5-4 Hz)': (0.5, 4),
                    'Theta (4-8 Hz)': (4, 8),
                    'Alpha (8-13 Hz)': (8, 13),
                    'Beta (13-30 Hz)': (13, 30),
                    'Gamma (30-45 Hz)': (30, 45)
                }
        
                band1 = st.selectbox('첫 번째 주파수 대역을 선택하세요', list(frequency_bands.keys()))
                band2 = st.selectbox('두 번째 주파수 대역을 선택하세요', list(frequency_bands.keys()))
        
                # 선택된 대역의 주파수 범위
                low1, high1 = frequency_bands[band1]
                low2, high2 = frequency_bands[band2]
        
                # 선택된 전극 데이터
                raw_data = raw_eeg[electrode]
        
                # 대역별 필터링
                filtered_signal1 = butter_bandpass_filter(raw_data, low1, high1, fs)
                filtered_signal2 = butter_bandpass_filter(raw_data, low2, high2, fs)
        
                # 신호 비교 시각화
                st.subheader(f'{band1} 대역과 {band2} 대역의 필터링된 신호 비교')
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(time[:len(filtered_signal1)], filtered_signal1[:len(time)], label=f'{band1} 대역', linewidth=0.5)
                ax.plot(time[:len(filtered_signal2)], filtered_signal2[:len(time)], label=f'{band2} 대역', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)


    
        elif analysis_type == 'Topomap Visualization':
            st.subheader('뇌파 활동 시각화 (Topomap)')

            colormap_info = {
                "viridis": "파랑 < 초록 < 노랑 / 선형적 증가로 신뢰성 높음",
                "plasma": "보라 < 주황 < 노랑 / 미세한 변화 관찰 용이",
                "inferno": "검정 < 빨강 < 주황 < 노랑 / 데이터의 차이를 강조",
                "magma": "검정 < 보라 < 주황 < 노랑 / 낮은 값을 강조",
                "cividis": "파랑 < 초록 < 노랑 / 선형적 변화로 신뢰성 높음",
                "cool": "파랑 < 하늘 < 분홍 / 디자인",
                "hot": "검정 < 빨강 < 주황 < 노량 < 하양 / 극단적 변화를 강조",
                "RdBu_r": "파랑(-) < 하양(0) < 빨강(+) / 양극 데이터의 차이를 강조",
                "Spectral": "파랑 < 초록 < 노랑 < 빨강 / 넓은 범위의 데이터를 정성적으로 표현",
            }

            colormaps = list(colormap_info.keys())
            selected_cmap = st.selectbox("Select Color Map", colormaps, index=8)

            st.write(f"### Selected Color Map: `{selected_cmap}`")
            st.write(colormap_info[selected_cmap])
    
            selected_time = st.slider(
                '시각화할 시간을 초 단위로 선택하세요',
                min_value=0.0,
                max_value=float(len(time) * sampling_interval),
                value=0.0,
                step=sampling_interval
            )
            selected_idx = int(selected_time / sampling_interval)
    
            montage = mne.channels.make_standard_montage('standard_1020')
            valid_electrodes = [ch for ch in raw_eeg.columns if ch in montage.ch_names]
    
            if len(valid_electrodes) == 0:
                st.error("유효한 전극 이름이 없습니다. 데이터와 채널 이름을 확인하세요.")
            else:
                activity_levels = [raw_eeg[ch].iloc[selected_idx] for ch in valid_electrodes]
        
                pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in valid_electrodes])

                
            plot_topomap(activity_levels, pos, title=f'EEG Activity at {selected_time:.3f} seconds', cmap=selected_cmap)


        if analysis_type == 'PCA':
            st.subheader('PCA 분석')
        
            processed_eeg = pd.DataFrame()
            for col in raw_eeg.columns:
                processed_eeg[col] = apply_processing(
                    raw_eeg[col].values, fs, preprocessing, selected_bands, frequency_bands, filter_first
                )[0]
        
            selected_electrodes = st.multiselect(
                'PCA를 적용할 전극을 선택하세요:',
                options=processed_eeg.columns.tolist(),
                default=processed_eeg.columns.tolist()
            )
        
            if selected_electrodes:
                selected_data = processed_eeg[selected_electrodes]
        
                n_components = len(selected_electrodes)
                pca = PCA(n_components=n_components)
                transformed_data = pca.fit_transform(selected_data)

                st.subheader("각 주요 성분에 대한 전극 기여도")
        
                loadings = pd.DataFrame(
                    pca.components_.T,
                    index=selected_electrodes,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
        
                st.dataframe(loadings)
        
                st.subheader('전극 기여도 시각화')
                selected_pc = st.selectbox(
                    '기여도를 시각화할 주요 성분을 선택하세요:',
                    [f'PC{i+1}' for i in range(n_components)]
                )
        
                selected_pc_index = int(selected_pc.replace('PC', '')) - 1
        
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(loadings.index, loadings.iloc[:, selected_pc_index], color='skyblue')
                ax.set_xlabel('Electrode')
                ax.set_ylabel('Contribution')
                ax.set_title(f'{selected_pc}의 전극 기여도')
                st.pyplot(fig)
        
                max_sum_components = st.slider(
                    '축소할 성분의 개수를 선택하세요',
                    min_value=1,
                    max_value=n_components,
                    value=n_components
                )
                
                st.subheader("모든 주요 성분의 분산 기여도")
                
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                explained_variance_df = pd.DataFrame({
                    '성분': [f'PC{i+1}' for i in range(n_components)],
                    '분산 기여도 (%)': explained_variance_ratio * 100,
                    '누적 분산 기여도 (%)': cumulative_variance * 100
                })
                
                st.dataframe(explained_variance_df.style.apply(
                    lambda x: ['background-color: gray' if i < max_sum_components else '' for i in range(len(x))],
                    axis=0
                ))
                
                st.subheader(f'선택된 주요 성분 (PC1 to PC{max_sum_components}) 개별 시각화')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                min_length = min(len(time), len(transformed_data))
                
                for i in range(max_sum_components):
                    ax.plot(time[:min_length], transformed_data[:min_length, i], label=f'PC{i+1}')
                    
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Selected PCA Components (PC1 to PC{max_sum_components})')
                ax.legend()
                st.pyplot(fig)
                
                summed_signal = np.sum(transformed_data[:, :max_sum_components], axis=1)
                
                st.subheader('선택된 주요 성분 합산 신호 시각화')
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(time[:min_length], summed_signal[:min_length], label='Summed Signal', color='red')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Summed Signal (PC1 to PC{max_sum_components})')
                ax.legend()
                st.pyplot(fig)
        
                st.subheader('모든 주요 성분 시각화')
        
                for i in range(n_components):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(time[:min_length], transformed_data[:min_length, i], label=f'PC{i+1}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'PC{i+1} 시각화')
                    ax.legend()
                    st.pyplot(fig)

                
                if uploaded_file is not None:
                    original_filename = uploaded_file.name
                    base_name = original_filename.split('.')[0] 
                    new_filename = f"{base_name}_PCA.csv"
                else:
                    new_filename = "transformed_data_PCA.csv" 

                transformed_df = pd.DataFrame(
                transformed_data,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
                
                csv = transformed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"축소된 데이터 다운로드 ({new_filename})",
                    data=csv,
                    file_name=new_filename,
                    mime='text/csv'
                )

                        
            else:
                st.warning("전극을 선택하세요.")
            
    else:
        st.write("먼저 CSV 파일을 업로드하세요.")

if __name__ == "__main__":
    main()
