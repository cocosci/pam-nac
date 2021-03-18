from utilities_n_libraries import *


def cal_psd(data):
    """
    Input: the waveform represented as a vector with the length of 512 
    Output: the power spectral density of the input with the max value of 96 dB
    """
    data = data / np.max(np.abs(data))
    P = np.maximum(20 * ma.log10(np.abs(scipy.fftpack.fft(data)[:(FFT_SIZE/2 + 1)]) / FFT_SIZE), -200)
    # P = np.maximum(np.abs(scipy.fftpack.fft(data)[:(FFT_SIZE / 2 + 1)])**2 / FFT_SIZE, -200)
    P = P.filled()
    P = np.reshape(P, ((FFT_SIZE/2 + 1), -1))
    Delta = 96 - np.max(P)
    P = P + Delta
    return P


def find_tonal_set(psd):
    """
    Description: given the input of the power spectral density, find a set of tonal maskers
    Reference: ISO/IEC 11172-3:1993, Information technology Coding of moving pictures 
                and associated audio for digital storage media at up to about 1,5 Mbit/s
                Part 3: Audio, with the permission of ISO.
    """
    Flags = np.zeros(int(FFT_SIZE / 2)) + NOT_EXAMINED
    local_max_list = np.empty((0, 2))
    counter = 0
    for k in range(int(FFT_SIZE / 2) - 1):
        if psd[k] > psd[k - 1] and psd[k] >= psd[k + 1] and 1 < k <= 249:
            the_pair = [k, psd[k][0]]
            local_max_list = np.concatenate((local_max_list, np.reshape(np.array(the_pair), (-1, 2))))
            counter = counter + 1
    Tonal_list = np.empty((0, 2))
    if local_max_list.shape[0] > 0:
        for i in range(len(local_max_list[:, 0])):
            k = int(local_max_list[i, INDEX])
            is_tonal = 1
            # Examine neighbouring frequencies
            if 2 < k < 63:
                J = [-2, 2]
            elif 63 <= k < 127:
                J = [-3, -2, 2, 3]
            elif 127 <= k < 150:
                J = [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]
            else:
                is_tonal = 0
                J = []
            for j in J:
                is_tonal = is_tonal and (psd[k] - psd[k + j] >= 7)
            if is_tonal:
                the_psd_val = 10 * ma.log10(
                    10 ** (psd[k - 1] / 10) + 10 ** (psd[k] / 10) + 10 ** (psd[k + 1] / 10)).filled(10 ** -10)
                the_pair = [k, the_psd_val[0]]
                Tonal_list = np.concatenate((Tonal_list, np.reshape(np.array(the_pair), (-1, 2))))
                Flags[k] = TONAL
                for j in np.concatenate((J, [-1, 1])):
                    Flags[k + j] = IRRELEVANT
    return Flags, Tonal_list
