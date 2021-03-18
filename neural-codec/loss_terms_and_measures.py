import tensorflow as tf
import os
from constants import *


def entropy_to_bitrate(total_entropy, the_strides):
    """
    Calculate the bitrate given the entropy
    :param total_entropy: the estimated amount of bits to represent each sample on average
    :param the_strides: by default it's set to be 2, which gives 256 samples per output vector from the encoder.
    :return: estimated bitrate 
    """
    bitrate = ((sample_rate / 1024.0) / (frame_length - overlap_each_side)) * \
              (frame_length / the_strides) * total_entropy
    return bitrate


def bitrate_to_entropy(bitrate, the_strides):
    pre_entropy_rate = (frame_length / the_strides) * (float(frame_length / the_strides) / frame_length)
    entropy = (bitrate / pre_entropy_rate * sample_rate)
    entropy *= (frame_length - overlap_each_side / float(frame_length))
    return entropy


def mse_loss(decoded_sig, original_sig, kai_re_mat=1):
    mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(decoded_sig, original_sig)), axis=-1)
    return tf.sqrt(mse + 1e-07)


def mfcc_transform(the_stft, the_spectrum, is_finetuning=False):
    """
    Warp the linear scale spectrograms into the mel-scale.
    """
    num_spectrogram_bins = the_stft.shape[-1]
    lower_edge_hertz, upper_edge_hertz = 0.0, sample_rate / 2.0
    selected_ind = [8, 16, 32, 128]
    MEL_FILTERBANKS = []
    for num_mel_bins in selected_ind:
        linear_to_mel_weight_matrix = tf.compat.v2.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        MEL_FILTERBANKS.append(linear_to_mel_weight_matrix)
    transform = []
    for i in range(0, int(len(MEL_FILTERBANKS))):
        mel_spectrograms = tf.matmul(the_spectrum, MEL_FILTERBANKS[i])  # axis = 1 means it's just mat mul.
        mel_spectrograms = mel_spectrograms[:, 3:int(mel_spectrograms.shape[1] * 1)]  # 112kbps
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-7)
        transform.append(log_mel_spectrograms)
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    return transform


def mel_scale_loss(decoded_sig, original_sig, is_finetuning=False):
    """
    Calculate the frequency loss in Mel scale. The code is derived from the paper, End-to-End Optimized Speech Coding 
    with Deep Neural Networks, authored by Srihari Kankanahalli. 
    """
    # calculate stft spectrum
    dec_stfts, dec_spectrograms = tf_stft(decoded_sig)
    ori_stfts, ori_spectrograms = tf_stft(original_sig)
    # calculate stft psd
    ori_spectrograms = ori_spectrograms ** 2
    ori_spectrograms = 1.0 / frame_length * ori_spectrograms
    dec_spectrograms = dec_spectrograms ** 2
    dec_spectrograms = 1.0 / frame_length * dec_spectrograms

    pvec_pred = mfcc_transform(dec_stfts, dec_spectrograms, is_finetuning)
    pvec_true = mfcc_transform(ori_stfts, ori_spectrograms, is_finetuning)

    distances = []
    for i in range(0, len(pvec_true)):
        error = tf.expand_dims(mse_loss(pvec_pred[i], pvec_true[i]), axis=-1)
        distances.append(error)
    distances = tf.concat(distances, axis=-1)
    mfcc_loss = tf.reduce_mean(input_tensor=distances, axis=-1)
    return mfcc_loss


def tf_stft(sig, the_frame_length=frame_length):
    dec_stfts = tf.compat.v2.signal.stft(tf.reshape(sig, [-1, frame_length]), frame_length=the_frame_length,
                                         frame_step=int(the_frame_length), fft_length=the_frame_length, window_fn=None)
    dec_stfts = tf.reshape(dec_stfts, (-1, int(the_frame_length / 2) + 1))
    dec_spectrograms = tf.sqrt(tf.square(tf.math.real(dec_stfts)) + tf.square(tf.math.imag(dec_stfts)) + 1e-7)
    return dec_stfts, dec_spectrograms


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def log_psd_to_psd(log_psd):
    return tf.math.pow(10.0, ((log_psd - 96) / 10.0))


def tf_psd(sig):
    spec = tf.compat.v2.signal.rfft(sig)
    abs_spec = tf.sqrt(tf.square(tf.math.real(spec)) + tf.square(tf.math.imag(spec)) + 1e-7)
    P = (20 * tf_log10(abs_spec / 512.0))
    return P, (abs_spec / 512.0) ** 2


def tf_stft_real_log10(sig):
    spec = tf.compat.v2.signal.rfft(sig)
    real_part = (tf.square(tf.math.real(spec))) + 1e-7
    return 10 * tf_log10(real_part / 512.0)


def tf_stft_imag_log10(sig):
    spec = tf.compat.v2.signal.rfft(sig)
    imag_part = (tf.square(tf.math.imag(spec))) + 1e-7
    return 10 * tf_log10(imag_part / 512.0)


# The higher the SMR is, the less masking we have.
# We use SMR to prioritize the the loss minimization in frequency domain.
def smr_loss(decoded_sig, original_sig, GMS):
    dec_psd, dec_spectrograms = tf_psd(decoded_sig)
    ori_psd, ori_spectrograms = tf_psd(original_sig)
    priority_mat = tf_log10(((10 ** (0.1 * ori_psd)) / (10 ** (0.1 * GMS))) + 1)
    mse = tf.reduce_mean(input_tensor=tf.multiply(tf.square(tf.subtract(ori_psd, dec_psd)), priority_mat), axis=-1)
    return mse


# This is to maximize the lower bound of MNR
def nmr_max_mean_loss(decoded_sig, original_sig, GMS, the_frame_length=frame_length):
    GMS_psd = log_psd_to_psd(GMS)  # converted back to just spectral density, not in log scale.
    diff_psd, diff_spectrograms = tf_psd(decoded_sig - original_sig)
    mse = tf.reduce_max(input_tensor=tf.nn.relu(log_psd_to_psd(diff_psd) / GMS_psd - 1), axis=-1)
    return mse


def quan_loss(softmax_assignment):
    """
    Calculate the quantization loss from the Softmax quantization.
    A more one-hot vectorized softmax assignment is favored, as it has a lower perplexity.
    """
    return tf.reduce_mean(input_tensor=(tf.reduce_sum(input_tensor=tf.sqrt(softmax_assignment + 1e-20), axis=-1) - 1.0),
                          axis=-1)


def entropy_coding_loss(soft_assignment):
    """
    Calculate the entropy value from the Softmax quantization. 
    """
    soft_assignment = tf.reshape(soft_assignment, (-1, soft_assignment.shape[2]))
    onehot_hist = tf.reduce_sum(input_tensor=soft_assignment, axis=0)
    onehot_hist /= tf.reduce_sum(input_tensor=onehot_hist)
    ent_loss = -tf.reduce_sum(input_tensor=onehot_hist * tf.math.log(onehot_hist + 1e-7) / tf.math.log(2.0))
    return ent_loss


def snr(ori_sig, dec_sig):
    min_len = min(len(ori_sig), len(dec_sig))
    ori_sig, dec_sig = ori_sig[:min_len], dec_sig[:min_len]
    nom = np.sum(np.power(ori_sig, 2))
    denom = np.sum(np.power(np.subtract(ori_sig, dec_sig), 2))
    eps = 1e-20
    snr = 10 * np.log10(nom / (denom + eps) + eps)
    return min_len, snr, ori_sig, dec_sig


def pesq(reference, degraded, sample_rate=None, program='pesq'):
    """ 
    Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
    on reference and degraded speech samples comparison.
    Sample rate must be 8000 or 16000 (or can be defined reading reference file
    header).
    PESQ utility must be installed.
    """
    if not os.path.isfile(reference) or not os.path.isfile(degraded):
        raise ValueError('reference or degraded file does not exist')
    if not sample_rate:
        import wave
        w = wave.open(reference, 'r')
        sample_rate = w.getframerate()
        w.close()
    if sample_rate not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')

    import subprocess
    args = [program, reference, degraded, '+%d' % sample_rate, '+wb']
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    out = out.decode("utf-8")
    last_line = out.split('\n')[-2]
    pesq_wb = float(last_line.split()[-1:][0])
    return pesq_wb
