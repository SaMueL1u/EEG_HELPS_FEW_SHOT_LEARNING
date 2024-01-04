from scipy import signal
import numpy.fft as fft
import numpy as np
import mne
def get_fft_values(y_values, N = 720):
    fft_values_ = fft.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return fft_values

def butter_bandpass_filter(data, lowcut = 5, highcut =95, fs = 128, order=2):
    wn = [lowcut / fs,highcut / fs]
    b, a = signal.butter(order, wn, 'bandpass', analog=False,output='ba')
    output = signal.filtfilt(b, a, data, axis=0)
    return output

def filter_new(x_data_once,highcut,lowcut):
    x_data_once_ = x_data_once
    top1 = x_data_once_.shape[0]
    top2 = x_data_once_.shape[1]
    x_data_once_ = x_data_once_.tolist()
    # 平滑滤波
    for j in range(top1):
        for z in range(top2 - 1):
            a = x_data_once_[j][z + 1]
            b = x_data_once_[j][z]
            del_ = a - b
            del_ = max(del_, -15)
            del_ = min(del_, 15)
            x_data_once_[j][z + 1] = x_data_once_[j][z] + del_

    x_data_once_ = np.asarray(x_data_once_)

    # High pass filter
    a = lowcut  # HPF filter coeffs
    b = highcut
    preVal = np.zeros(top2)
    eeg_filt = np.zeros((top1, top2))
    for j in range(top1):
        preVal = a * x_data_once_[j] + b * preVal
        eeg_filt[j] = x_data_once_[j] - preVal

    IIR_TC = 256
    EEG_data = eeg_filt
    rows, columns = top1, top2
    filtedData = np.zeros((rows, columns))
    back_ = EEG_data[0, :]

    for r in range(1, rows):
        back_ = (back_ * (IIR_TC - 1) + EEG_data[r, :]) / IIR_TC
        filtedData[r, :] = EEG_data[r, :] - back_

    return filtedData

def new_filter(eeg_data,high_pass,low_pass):
    info = mne.create_info(
        ch_names=['AF3', 'AF4', 'T7', 'T8', 'Pz'],
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
        sfreq=128
    )
    custom_raw = mne.io.RawArray(eeg_data, info)

    custom_raw.filter(low_pass, high_pass, fir_design='firwin')

    return custom_raw._data