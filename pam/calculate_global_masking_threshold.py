from calculate_psd_tonal_maskers import *
from calculate_non_tonal_maskers import *
from decimation import *
from calculate_individual_mask import *
import argparse
from collections import OrderedDict
import sys


def global_threshold(LTq, LTt, LTn):
    """
    Description: calculate the global masking threshold by accumulating the masking curve
                 of the tonal, non-tonal maskers and absolute hearing threshold.
    """
    LTg = np.zeros((len(LTq), len(TH[:, 0]))) + MIN_POWER
    N = len(LTq)
    if len(LTt) > 0:
        m = len(LTt[:, 0])
    n = len(LTn[:, 0])
    for i in range(N):
        # Threshold in quiet
        temp = 10 ** (LTq[i] / 10)
        # Contribution of the tonal component
        if len(LTt) > 0:
            for j in range(m):
                temp = temp + 10 ** (LTt[j, i] / 10)
        # Contribution of the noise components
        for j in range(n):
            temp = temp + 10 ** (LTn[j, i] / 10)
        LTg[i] = 10 * ma.log10(temp)

    return LTg


def plot(global_psd, Tonal_list, LTt, LTn, Non_tonal_list, LTg):
    size = 17
    fig, ax = plt.subplots(figsize=(10., 5))

    plt.ylim((-20, 100))
    plt.xlim((0, 256))
    if sample_rate == 44100:
        the_x_tick = ["{0:.1f}".format(ss) for ss in np.linspace(0, 22, 10)]
    else:
        the_x_tick = ["{0:.1f}".format(ss) for ss in np.linspace(0, 16, 10)]
    the_y_tick = [int(ss) for ss in np.linspace(-20, 100, 6)]
    plt.xticks(np.linspace(0, 256, 10), the_x_tick, size=size)
    plt.yticks(np.linspace(-20, 100, 6), the_y_tick, size=size)
    plt.rc('xtick', labelsize=size)
    plt.rc('ytick', labelsize=size)
    plt.xlabel('frequency (kHz)', fontsize=size + 1)
    plt.ylabel('sound pressure level (dB)', fontsize=size + 1)

    plt.plot(global_psd, color='b', linewidth=3.5, label='power spectral density')
    plt.plot([int(i) for i in Tonal_list[:, 0]], [int(i) for i in Tonal_list[:, 1]], marker='x', markersize=8,
             markeredgewidth=4, markeredgecolor='m', markerfacecolor='m', linestyle='None', label='tonal masker')
    plt.plot([int(i) for i in Non_tonal_list[:, 0]], [int(i) for i in Non_tonal_list[:, 1]], marker='o', markersize=8,
             markeredgecolor='k', markerfacecolor='w', linestyle='None', label='noise masker')
    for j in range(len(Non_tonal_list[:, 0])):
        plt.plot(TH[:, 0], LTn[j, :], 'k:', label='noise masking threshold')
    for j in range(len(Tonal_list[:, 0])):
        plt.plot(TH[:, 0], LTt[j, :], 'm:', linewidth=3, label='tonal masking threshold')
    plt.plot(TH[:, 0], LTg, 'r-', label='global masking threshold', linewidth=2)
    plt.plot(TH[:, 0], LTq, 'r--', linewidth=3, label='absolute hearing threshold')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize='x-large', fancybox=True, framealpha=0.35, loc='best')
    plt.tight_layout()
    plt.show()
    fig.savefig(fname='./output/pam-1-test.png', dpi=100)


def one_pass_calculation(data):
    P = cal_psd(data)
    global_psd = P.flatten()
    Flags, Tonal_list = find_tonal_set(P)
    Flags, Non_tonal_list = find_non_tonal_set(Flags, P)
    Flags, Tonal_list, Non_tonal_list = decimation(Flags, Tonal_list, Non_tonal_list)
    LTt, LTn = individual_masking_thresholds(P, Tonal_list, Non_tonal_list, TH, Map)
    LTg = global_threshold(LTq, LTt, LTn)
    global_threshold_data = np.interp(np.arange(0, (FFT_SIZE / 2 + 1)), TH[:, 0], LTg[:, 1])
    return global_psd, Tonal_list, LTt, LTn, Non_tonal_list, LTg, global_threshold_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str,
                        help='Path of the input data in time domain. The data should either be ?x512 or a vector.')
    parser.add_argument('--sample_rate', type=int,
                        help='The program currently supports two sample rates: 44100 Hz and 32000 Hz.')
    args = parser.parse_args()

    if args.sample_rate not in [44100, 32000]:
        print('Sample rate not supported!')
        sys.exit()
    else:
        sample_rate = args.sample_rate
    CB, C, TH, LTq, Map = acoustic_parameter_setup(sample_rate)

    data = np.load(args.input_data)
    if len(data.shape) == 2 and data.shape[1] == 512:
        global_threshold_data = np.empty((data.shape[0], (FFT_SIZE / 2 + 1)))
        global_psd = np.empty((data.shape[0], (FFT_SIZE / 2 + 1)))
        for i in range(data.shape[0]):
            global_psd, _, _, _, _, _, global_threshold_data = one_pass_calculation(data[i, :])
            global_psd[i, :] = global_psd
            global_threshold_data[i, :] = global_threshold_data
        np.save('./output/global_threshold_data_' + str(args.sample_rate) + 'kHz.npy', global_threshold_data)
        np.save('./output/global_psd_' + str(args.sample_rate) + 'kHz.npy', global_psd)
    else:
        global_psd, Tonal_list, LTt, LTn, Non_tonal_list, LTg, global_threshold_data = one_pass_calculation(data[:FFT_SIZE])
        plot(global_psd, Tonal_list, LTt, LTn, Non_tonal_list, LTg)
