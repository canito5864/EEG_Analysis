import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import pywt
from numpy.fft import fft
import matplotlib.pyplot as plt
import mne

# Helper functions for signal processing
def main():
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def wavelet_transform(data, wavelet="db4", level=2):
        coeffs = pywt.wavedec(data, wavelet, level=level)
        return pywt.waverec(coeffs, wavelet)
    
    def fourier_transform(data):
        return np.abs(fft(data))
    
    def calculate_difference(signal1, signal2):
        return np.abs(signal1 - signal2)
    
    def plot_topomap(data, pos, title='EEG Topomap'):
        fig, ax = plt.subplots()
        mne.viz.plot_topomap(data, pos[:, :2], axes=ax, show=False, contours=0)
        ax.set_title(title)
        st.pyplot(fig, figsize(300, 50))
    
    # Streamlit app
    st.title('EEG 뇌파 분석')
    
    # CSV 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
    
    if uploaded_file is not None:
        # CSV 파일 읽기
        raw_eeg = pd.read_csv(uploaded_file)
    
        # 'eeg.'로 시작하는 특정 전극만 필터링
        eeg_columns = ['eeg.af3', 'eeg.af4', 'eeg.t7', 'eeg.t8', 'eeg.pz']
        raw_eeg = raw_eeg[eeg_columns]
    
        # 전극 이름 매핑 (MNE 표준 채널 이름으로)
        mapping = {
            'eeg.af3': 'AF3',
            'eeg.af4': 'AF4',
            'eeg.t7': 'T7',
            'eeg.t8': 'T8',
            'eeg.pz': 'Pz'
        }
    
        # 전극 이름을 MNE 표준 전극 이름으로 매핑
        raw_eeg = raw_eeg.rename(columns=mapping)
    
        # 타임 시리즈 생성
        sampling_interval = 0.007  # 예시로 0.007초 샘플링 간격을 사용
        fs = 1 / sampling_interval
        time = np.arange(0, len(raw_eeg) * sampling_interval, sampling_interval)
    
        # 분석 옵션 선택
        analysis_type = st.selectbox('분석할 옵션을 선택하세요', (
            'Single Electrode',
            'Electrode Comparison',
            'Topomap Visualization'
        ))
    
        # 주파수 대역 선택 (여러 개 선택 가능)
        frequency_bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-45 Hz)': (30, 45)
        }
        selected_bands = st.multiselect('분석할 주파수 대역을 선택하세요', list(frequency_bands.keys()))
    
        # 전처리 선택 (복수 선택 가능)
        preprocessing = st.multiselect(
            '적용할 전처리 방법을 선택하세요',
            ['None', 'Wavelet Transform', 'Fourier Transform', 'Butterworth Filter'],
            default=['None']
        )
    
        def apply_preprocessing(data, electrode_key):
            # 전처리 단계별로 신호 처리
            if 'Wavelet Transform' in preprocessing:
                data = wavelet_transform(data)
                st.write(f'{electrode_key}: 이산 웨이블릿 변환 적용됨.')
    
            if 'Fourier Transform' in preprocessing:
                data = fourier_transform(data)
                st.write(f'{electrode_key}: 푸리에 변환 적용됨.')
    
            filtered_signals = []
            if 'Butterworth Filter' in preprocessing and selected_bands:
                for band in selected_bands:
                    lowcut, highcut = frequency_bands[band]
                    filtered_signal = butter_bandpass_filter(data, lowcut, highcut, fs)
                    st.write(f'{electrode_key}: {lowcut}-{highcut} Hz {band} 버터워스 필터링 적용됨.')
                    filtered_signals.append(filtered_signal)
    
                # 여러 주파수 대역의 결과를 평균 내어 반환
                data = np.mean(filtered_signals, axis=0)
    
            return data
    
        if analysis_type == 'Single Electrode':
            st.subheader('단일 전극 분석')
    
            # 전극 선택
            electrode = st.selectbox('분석할 전극을 선택하세요', raw_eeg.columns)
            raw_data = raw_eeg[electrode]
    
            # 전처리 적용
            processed_data = apply_preprocessing(raw_data, electrode)
    
            # 전처리된 신호 시각화
            st.subheader(f'{electrode} 채널의 전처리된 EEG 신호')
            fig, ax = plt.subplots()
            ax.plot(time[:len(processed_data)], processed_data, label=f'{electrode} - 필터링된 신호')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            st.pyplot(fig, figsize(300, 50)
    
        elif analysis_type == 'Electrode Comparison':
            st.subheader('전극 간 비교 분석')
    
            # 두 전극 선택
            electrode1 = st.selectbox('첫 번째 전극을 선택하세요', raw_eeg.columns, index=0)
            electrode2 = st.selectbox('두 번째 전극을 선택하세요', raw_eeg.columns, index=1)
    
            data1 = raw_eeg[electrode1]
            data2 = raw_eeg[electrode2]
    
            # 전처리 적용
            processed_data1 = apply_preprocessing(data1, electrode1)
            processed_data2 = apply_preprocessing(data2, electrode2)
    
            # 두 신호를 같은 그래프에 시각화
            st.subheader(f'{electrode1}과 {electrode2}의 전처리된 신호 비교')
            fig, ax = plt.subplots()
            ax.plot(time[:len(processed_data1)], processed_data1, label=f'{electrode1} - 필터링된 신호')
            ax.plot(time[:len(processed_data2)], processed_data2, label=f'{electrode2} - 필터링된 신호')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            st.pyplot(fig, figsize(300, 50)
    
            # 전극 간 차이 계산
            difference = calculate_difference(processed_data1, processed_data2)
    
            st.subheader(f'{electrode1}과 {electrode2}의 EEG 신호 차이')
            fig, ax = plt.subplots()
            ax.plot(time[:len(difference)], difference, label='Difference')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude Difference')
            ax.legend()
            st.pyplot(fig, figsize(300, 50)
    
        elif analysis_type == 'Topomap Visualization':
            st.subheader('뇌파 활동 시각화 (Topomap)')
    
            # 특정 시간 선택 (슬라이더를 사용하여 연속적으로 시간 조절)
            selected_time = st.slider(
                '시각화할 시간을 초 단위로 선택하세요',
                min_value=0.0,
                max_value=float(len(time) * sampling_interval),
                value=0.0,
                step=sampling_interval
            )
            selected_idx = int(selected_time / sampling_interval)
    
            # EEG 채널 위치 정보 가져오기
            montage = mne.channels.make_standard_montage('standard_1020')
            valid_electrodes = [ch for ch in raw_eeg.columns if ch in montage.ch_names]
    
            if len(valid_electrodes) == 0:
                st.error("유효한 전극 이름이 없습니다. 데이터와 채널 이름을 확인하세요.")
            else:
                # 각 전극에서의 신호 세기 추출
                activity_levels = [raw_eeg[ch].iloc[selected_idx] for ch in valid_electrodes]
    
                # EEG 채널 위치 정보
                pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in valid_electrodes])
    
                # Topomap 시각화 (XYZ에서 XY 좌표로 변환)
                plot_topomap(activity_levels, pos, title=f'EEG Activity at {selected_time:.3f} seconds')
    else:
        st.write("먼저 CSV 파일을 업로드하세요.")

if __name__ == "__main__":
    main()
