import tensorflow as tf
# tf.python.util.deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import os
from constants import *


def mu_law_mapping(input_x):
	return tf.sign(input_x) * (tf.compat.v1.math.log(1.0 + mu * tf.abs(input_x)) / tf.compat.v1.math.log(1.0 + mu))

def inverse_mu_law_mapping(input_x):
    return tf.sign(input_x) * (1 / mu) * ((1 + mu) ** tf.abs(input_x) - 1)

def entropy_to_bitrate(total_entropy, the_strides):	
	bitrate = total_entropy / ((frame_length - overlap_each_side) / float(frame_length)) #h*w/(w-o)
	PRE_ENTROPY_RATE = (frame_length/the_strides) * (float(frame_length/the_strides) / frame_length) #256/r
	bitrate = bitrate * PRE_ENTROPY_RATE / sample_rate #h*w/(w-o) * (256/r) /16 = 16*h*w/(w-o)*r
	return bitrate

def entropy_to_bitrate(total_entropy, the_strides):	
	bitrate  = ((sample_rate/1024.0) / (frame_length - overlap_each_side)) * (frame_length/the_strides) * total_entropy
	return bitrate


def bitrate_to_entropy(bitrate, the_strides):
	PRE_ENTROPY_RATE = (frame_length/the_strides) * (float(frame_length/the_strides) / frame_length)
	entropy = (bitrate / PRE_ENTROPY_RATE * sample_rate)
	entropy *= (frame_length - overlap_each_side / float(frame_length))
	return entropy

def mse_loss(decoded_sig, original_sig, kai_re_mat=1):
	mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(decoded_sig, original_sig)), axis = -1)
	return tf.sqrt(mse + 1e-07)

def tflog10(x):
	numerator = tf.compat.v1.log(x)
	denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator


def mfcc_transform(the_stft, the_spectrum, is_finetuning=False):
	# Warp the linear scale spectrograms into the mel-scale.
	num_spectrogram_bins = the_stft.shape[-1]
	# sample_rate, lower_edge_hertz, upper_edge_hertz = 16000, 0.0, 8000.0
	lower_edge_hertz, upper_edge_hertz = 0.0, sample_rate/2.0
	selected_ind = [8, 16, 32, 128]
	MEL_FILTERBANKS = []
	for num_mel_bins in selected_ind:
		linear_to_mel_weight_matrix = tf.compat.v2.signal.linear_to_mel_weight_matrix(
			num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
		MEL_FILTERBANKS.append(linear_to_mel_weight_matrix)
	transform = []
	for i in range(0, int(len(MEL_FILTERBANKS))):
		mel_spectrograms = tf.matmul(the_spectrum, MEL_FILTERBANKS[i]) # axis = 1 means it's just mat mul.
		mel_spectrograms = mel_spectrograms[:, 3:int(mel_spectrograms.shape[1]*1)] #112kbps
		log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-7)
		transform.append(log_mel_spectrograms)
	# Compute a stabilized log to get log-magnitude mel-scale spectrograms.
	return transform

def mfcc_loss(decoded_sig, original_sig, is_finetuning=False):
	# calculate stft spectrum
	the_frame_length = frame_length
	dec_stfts, dec_spectrograms = tf_stft(decoded_sig)
	ori_stfts, ori_spectrograms = tf_stft(original_sig)
	# calculate stft psd
	ori_spectrograms = ori_spectrograms**2
	ori_spectrograms = 1.0 / frame_length * ori_spectrograms 
	dec_spectrograms = dec_spectrograms**2
	dec_spectrograms = 1.0 / frame_length * dec_spectrograms
	###########
	
	pvec_pred = mfcc_transform(dec_stfts, dec_spectrograms, is_finetuning)
	pvec_true = mfcc_transform(ori_stfts, ori_spectrograms, is_finetuning)
	
	distances = []
	for i in range(0, len(pvec_true)):
		error = tf.expand_dims(mse_loss(pvec_pred[i], pvec_true[i]), axis = -1)
		distances.append(error)
	distances = tf.concat(distances, axis = -1)
	mfcc_loss = tf.reduce_mean(input_tensor=distances, axis = -1)   
	
	return mfcc_loss

def tf_stft(sig, the_frame_length = frame_length):		
	dec_stfts = tf.compat.v2.signal.stft(tf.reshape(sig, [-1, frame_length]), frame_length = the_frame_length, frame_step=int(the_frame_length), fft_length=the_frame_length, window_fn=None)
	# dec_stfts = tf.compat.v2.signal.rfft(sig)
	dec_stfts = tf.reshape(dec_stfts, (-1, int(the_frame_length/2)+1))
	dec_spectrograms = tf.sqrt(tf.square(tf.math.real(dec_stfts)) + tf.square(tf.math.imag(dec_stfts)) + 1e-7)
	return dec_stfts, dec_spectrograms

def tf_log10(x):
	numerator = tf.math.log(x)
	denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator

def tf_psd(sig, the_frame_length = frame_length):
	# dec_stfts = tf.compat.v2.signal.stft(tf.reshape(sig, [-1, frame_length]), frame_length = the_frame_length, frame_step=int(the_frame_length), fft_length=the_frame_length, window_fn=None)
	# dec_stfts = tf.reshape(dec_stfts, (-1, int(the_frame_length/2)+1))
	spec = tf.compat.v2.signal.rfft(sig)
	# abs_spec = tf.abs(spec)
	abs_spec = tf.sqrt(tf.square(tf.math.real(spec)) + tf.square(tf.math.imag(spec)) + 1e-7)
	# P = tf.clip_by_value(20 * tf_log10(abs_spec / 512), -200, 10000)
	P = (20 * tf_log10(abs_spec / 512))
	Delta = 96 - np.max(P)
	P = P +  Delta
	return P, spec


# The higher the SMR is, the less masking we have.
# We use SMR to prioritize the the loss minimization in frequency domain.
def SMR(decoded_sig, original_sig, GMS, the_frame_length = frame_length):
	dec_psd, dec_spectrograms = tf_psd(decoded_sig)
	ori_psd, ori_spectrograms = tf_psd(original_sig)
	# Usually you wouldn't spend any bits on the parts of the signal where 
	# you have a negative SMR, because that would imply that that part of the signal is inaudible anyway.
	SMR = tf.nn.relu(ori_psd-GMS) 
	mse = tf.reduce_mean(input_tensor=tf.multiply(tf.square(tf.subtract(ori_psd, dec_psd)), SMR), axis = -1)
	return tf.sqrt(mse + 1e-07)

# This is to maximize the average MNR 
def MNR(decoded_sig, original_sig, GMS, the_frame_length = frame_length):
	diff_psd, diff_spectrograms = tf_psd(decoded_sig - original_sig)
	return tf.reduce_mean((diff_psd - GMS) + 50, axis = -1)

# This is to maximize the lower bound of MNR 
def MNR_reduce_min(decoded_sig, original_sig, GMS, the_frame_length = frame_length):
	diff_psd, diff_spectrograms = tf_psd(decoded_sig - original_sig)
	return -tf.reduce_min((GMS - diff_psd) + 50, axis = -1)
	
def psd_loss(decoded_sig, original_sig, mat):
	ori_psd, ori_spectrograms = tf_psd(decoded_sig)
	dec_psd, dec_spectrograms = tf_psd(original_sig)
	mse = tf.reduce_mean(input_tensor=tf.multiply(tf.square(tf.subtract(ori_psd, dec_psd)), mat), axis = -1)
	return tf.sqrt(mse + 1e-07)

def stft_loss(decoded_sig, original_sig, mat):
	ori_stft, dec_spectrograms = tf_stft(decoded_sig)
	dec_stft, ori_spectrograms = tf_stft(original_sig)
	mse = tf.reduce_mean(input_tensor=tf.multiply(tf.square(tf.subtract(ori_spectrograms, dec_spectrograms)), mat), axis = -1)
	return tf.sqrt(mse + 1e-07)


def quan_loss(softmax_assignment):
	return tf.reduce_mean(input_tensor=(tf.reduce_sum(input_tensor=tf.sqrt(softmax_assignment + 1e-20), axis=-1) - 1.0), axis=-1)

def entropy_coding_loss(soft_assignment, the_strides):
	code_segment_len = int(frame_length/the_strides) 
	soft_assignment = tf.reshape(soft_assignment, (-1, soft_assignment.shape[2]))
	onehot_hist = tf.reduce_sum(input_tensor=soft_assignment, axis = 0)
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
	""" Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
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
	args = [ program, reference, degraded, '+%d' % sample_rate, '+wb']
	pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
	out, _ = pipe.communicate()
	out = out.decode("utf-8") 
	last_line = out.split('\n')[-2]
	pesq_wb = float(last_line.split()[-1:][0])
	return pesq_wb
