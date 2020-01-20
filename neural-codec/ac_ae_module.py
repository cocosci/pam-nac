"""
File name: ac_ae_module.py
Description: Implementation of the single ``coding module" in CMRL.

Author: Kai Zhen 
Contact: zhenk@iu.edu
Date: 2018/11/7
"""
from os import listdir
import random
import sys
import scipy.io.wavfile
import librosa
import argparse
# from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import math
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from data_preparation import PyTimitProcessing
import os.path
import time
import collections
from pystoi.stoi import stoi
import soundfile as sf 
from loss_terms_and_measures import *
# from lpc_utilities import *
from differential_coding import *
from constants import *
#tf.enable_eager_execution()
#tf.executing_eagerly()
# tf.compat.v1.disable_eager_execution()


EPS = np.finfo("float").eps


class AudioCodingAE(object):
	"""
	AudioCodingAE serves as one module in CMRL.
	Each module contains 0.45 million parameters;
	"""
	def __init__(self, arg):

		self._root_path = '/N/u/zhenk/BNN/libsdae-autoencoder-tensorflow/'
		
		self._learning_rate_tanh = arg.learning_rate_tanh
		
		self._coeff_term = list(map(lambda x:float(x), arg.coeff_term.split()))
		# self._learning_rate_whole_greedy_system = arg.learning_rate_whole_greedy_system
		timit_processer = PyTimitProcessing('Just do it!')
		self._pretrain_step = arg.pretrain_step
		self._target_entropy= arg.target_entropy
		self._the_strides = list(map(lambda x: int(x), arg.the_strides.split()))
		self._res_scalar = arg.res_scalar

		self._save_unique_mark = arg.save_unique_mark
		self._num_bins_for_follower = list(map(lambda x: int(x), arg.num_bins_for_follower.split()))
		# self._start_ind = np.load('./ac_stft_data/450_stft_ind.npy')
		
		self._max_amp = max_amp_tr #86.491813659667969# 129.73772048950195# 33.46148014068603#129.0# 33.46148014068603#timit_processer.get_max_amp()#32.904055595397949 # This is ugly coding. Will update in the future.
		# tr_data = np.load('./tr_lpc_res_300000_frame_length.npy')[:10000, :]

		# tr_lpc_coeff = np.load('./end2end_lpc_coeff_segments_16.npy')[:5000, :]
		# tr_lpc_res = np.load('./residuals_with_quantized_lpc_coeff_32bins.npy')[:, :]
		
		self._tr_data_size = training_data_size
		
		if sample_rate == 44100:
			tr_data = np.load(self._root_path + 'audio_400_512_44kHz.npy')# [:int(self._tr_data_size), :]
			# weighting_mat = np.load(self._root_path + 'hard_weighting_mat.npy').T
			# weighting_mat = np.load(self._root_path + 'soft_weighting_mat.npy').T
			weighting_mat = np.load(self._root_path + 'global_threshold_data.npy')
			# weighting_mat = np.concatenate((np.ones((tr_data.shape[0], 157)), np.zeros((tr_data.shape[0], 100))), 1)
			# weighting_mat = np.ones((tr_data.shape[0], 257))
			# weighting_mat = np.concatenate((np.load(self._root_path + 'soft_weighting_mat.npy').T[:, :100], np.zeros((tr_data.shape[0], 157))), 1)
			# weighting_mat = np.concatenate((np.zeros((tr_data.shape[0], 5)), np.concatenate((np.load(self._root_path + 'soft_weighting_mat.npy').T[:, 5:100], np.zeros((tr_data.shape[0], 157))), 1)) , 1)
			print('44100 case!!!', weighting_mat.shape, tr_data.shape)
		else:
			tr_data = np.load(self._root_path + 'audio_400_512_32kHz.npy')# [:int(self._tr_data_size), :]
			# weighting_mat = np.load(self._root_path + 'soft_weighting_mat_32.npy').T
			weighting_mat = np.load(self._root_path + 'global_threshold_data_32k.npy')

			
			print('32000 case!!!', weighting_mat.shape)
		
		

		print(' tr_data: ', tr_data.shape)
		self._tr_data = tr_data# [:5000, :]
		self._weighting_mat = weighting_mat# [:5000, :]
			
		self._epoch_tanh = arg.epoch_tanh
		self._epoch_greedy_followers = list(map(lambda x:int(x), arg.epoch_greedy_followers.split()))
		self._batch_size = arg.batch_size
		self._training_mode = int(arg.training_mode)
		self._iter_per_batch = self._tr_data.shape[0]//self._batch_size # //300
		
		self._sep_val = [self._root_path+'etri-val/' + f for f in listdir(self._root_path+'etri-val/') if f.endswith('.wav')]
		self._sep_test_files_10 = [self._root_path+'etri-test/' + f for f in listdir(self._root_path+'etri-test/') if f.endswith('.wav')]


		self._rand_model_id = str(np.random.randint(1000000, 2000000))
		print('self._rand_model_id:', self._rand_model_id)
		
		self._base_model_id = arg.base_model_id
		self._suffix = arg.suffix

		self._bottleneck_kernel_and_dilation = list(map(lambda x:int(x), arg.bottleneck_kernel_and_dilation.split()))
		# 
		self._window_size = arg.window_size
		
		self.write_to_file_and_update_to_display(str(arg)+'\n\n')

	
	def write_to_file_and_update_to_display(self, the_string):
		self._file_handler = open('./doc/' + self._rand_model_id + '_final.txt','a')
		self._file_handler.write(the_string)
		self._file_handler.close()

	def write_to_file_and_update_to_display_final(self, the_string):
		self._file_handler = open('./final_score/' + self._base_model_id + '.txt','a')
		self._file_handler.write(the_string)
		self._file_handler.close()

	def _generate_one_epoch_end2end(self, x, y, batchsize):
		the_list = list(range(0, self._tr_data.shape[0] - self._batch_size, self._batch_size))
		# the_list = list(np.random.choice(self._tr_data.shape[0] - self._batch_size, self._batch_size, replace = False))
		random.shuffle(the_list)
		for i in the_list[:5000]:
			ret = np.reshape(self._tr_data[i:i+batchsize, :], (batchsize, frame_length, 1))
			weighting_mat = np.reshape(self._weighting_mat[i:i+batchsize, :], (batchsize, int(frame_length/2)+1))
			yield ret, ret, weighting_mat

		
	def _load_sig(self, the_wav_file):

		"""
		_load_sig_to_long_arr is called
		Read one test file, normalize it, divided it by the max.
		@param the_wave_file: one test wave file.
		@return per_stft: stft spectram for that file
		"""
		
		s, sr = librosa.load(the_wav_file, sr=None) # saving redundantly many speech sources (easier to handle)
		
		if sr == sample_rate:
			pass
		else:
			s = librosa.resample(s, sr, sample_rate)
			print(sample_rate, len(s))
		# the_scaler = (25)*np.std(s)*(1/1.0)
		the_scaler = np.std(s) * self._max_amp
		# per_signal = s/np.std(s)		
		# per_signal /= (self._max_amp)

		s /= the_scaler
		return s, the_scaler


	def end2end_final_eval(self, ori_sig, dec_sig):
		_min_len, _snr, _ori_sig, _dec_sig  = snr(ori_sig, dec_sig)

		the_stoi = stoi(_ori_sig, _dec_sig, sample_rate, extended=False)

		sf.write('ori_sig_that_one'+self._rand_model_id+'.wav', ori_sig, sample_rate, 'PCM_16')
		sf.write('dec_sig_that_one'+self._rand_model_id+'.wav', dec_sig, sample_rate, 'PCM_16')
		the_pesq = 0.0  # disabled for audio coding.
		
		
		return _min_len, _snr, float(the_stoi), float(the_pesq), np.corrcoef(_ori_sig, _dec_sig)[0][1]


	def init_placeholder_end_to_end(self):
		x = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None, frame_length, 1), name = 'x')
		x_ = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None, frame_length, 1), name = 'x_')
		lr = tf.compat.v1.placeholder(dtype = tf.float32, shape = None, name = 'lr')
		the_share = tf.compat.v1.placeholder(dtype = tf.float32, shape = None, name = 'the_share')
		return x, x_, lr, the_share


	######## FCN #######
	def conv1d(self, inputs, num_filters, filter_size, padding = 'SAME', dilation_rate = 1, strides = 1, activation = tf.nn.tanh):
		out = tf.compat.v1.layers.conv1d(
			inputs = inputs, filters = num_filters, padding = padding,
			kernel_size = filter_size,
			activation = activation, 
			dilation_rate = dilation_rate,
			strides = strides,
			data_format = 'channels_last')
		return out

	def the_bottle_neck(self, the_input, wide_layer = 30, narrow_layer = 10, non_dilated_neck_kernel_size = 9, dilated_neck_kernel_size = 9, dilation_rate = 1, is_last_flat = False):
		compressed_bit2 = self.conv1d(the_input, narrow_layer, filter_size = non_dilated_neck_kernel_size, padding = 'SAME', dilation_rate = 1, activation = tf.nn.leaky_relu)
		compressed_bit2 = self.conv1d(compressed_bit2, narrow_layer, filter_size = dilated_neck_kernel_size, padding = 'SAME', dilation_rate = dilation_rate, activation = tf.nn.leaky_relu)
		compressed_bit2 = self.conv1d(compressed_bit2, wide_layer, filter_size = non_dilated_neck_kernel_size, padding = 'SAME', dilation_rate = 1, activation = None)
		if not is_last_flat:
			return tf.nn.leaky_relu(compressed_bit2 + the_input)
		else:
			return compressed_bit2 + the_input

	def stack_four_bottle_neck(self, compressed_bit, strides = 1, is_post_up_samling = True):
		compressed_bit_input = compressed_bit
		assert self._bottleneck_kernel_and_dilation[2]%strides == 0

		if compressed_bit.shape[-1] == 1: # this basically is for the channel expansion.
			wide_layer = self._bottleneck_kernel_and_dilation[2]
		else:
			wide_layer = int(compressed_bit.shape[-1]/strides) if is_post_up_samling else compressed_bit.shape[-1]
		
		for i in range(len(self._bottleneck_kernel_and_dilation) - 4): # the first one is the kernel # second is the wide layer
			flag = i==(len(self._bottleneck_kernel_and_dilation) - 5)
			compressed_bit = self.the_bottle_neck(compressed_bit, 
				non_dilated_neck_kernel_size = self._bottleneck_kernel_and_dilation[1],
				dilated_neck_kernel_size = self._bottleneck_kernel_and_dilation[0], 
				wide_layer = wide_layer,
				narrow_layer = self._bottleneck_kernel_and_dilation[3],
				dilation_rate = self._bottleneck_kernel_and_dilation[i + 4], 
				is_last_flat = flag)			
		return compressed_bit
		

	def change_channel_my(self, the_input, wide_layer = 30, the_channel = 1, kernel_size = 5, dilation_rate = 1, strides=1, activation=None):
		compressed_bit = self.conv1d(the_input, the_channel, filter_size = kernel_size, padding = 'SAME', dilation_rate = 1, strides = strides, activation = activation)
		return compressed_bit


	def down_sampling_mod(self, the_input, the_stride = 2):
		r = self.parametric_relu(self.change_channel(the_input, the_channel = self._bottleneck_kernel_and_dilation[2], kernel_size = 9, strides = the_stride, activation=None)) 
		return r

	def down_sampling_mod_my(self, the_input, the_stride = 2):
		r = self.conv1d(the_input, self._bottleneck_kernel_and_dilation[2], filter_size = 9, padding = 'SAME', dilation_rate = 1, strides = the_stride, activation = tf.nn.leaky_relu)
		return r # + r3


	def up_sampling_mod(self, the_input, the_stride = 2):		
		r = tf.reshape(the_input, (-1, the_input.shape[1], the_input.shape[2] // the_stride, the_stride))
		r = tf.keras.backend.permute_dimensions(r, (0, 1, 3, 2))
		r = tf.reshape(r, (-1, the_input.shape[1] * the_stride, the_input.shape[2] // the_stride))
		return r


	def up_sampling_mod_my(self, the_input, the_stride = 2):
		the_input_1 = self.conv1d(the_input, the_input.shape[-1], filter_size = 9, padding = 'SAME', dilation_rate = 1, strides = 1, activation = tf.nn.leaky_relu)
		r1 = self.up_sampling_mod(the_input_1, the_stride = the_stride)
		return r1


	def the_encoder_in_each_module(self, the_input, the_stride):
		compressed_bit = self.change_channel_my(the_input, the_channel = self._bottleneck_kernel_and_dilation[2], kernel_size = 55, activation = tf.nn.leaky_relu)
		compressed_bit = self.stack_four_bottle_neck(compressed_bit, is_post_up_samling = False)		
		compressed_bit = self.down_sampling_mod_my(compressed_bit, the_stride = the_stride)
		post_down_sampling_hidden = compressed_bit
		compressed_bit = self.stack_four_bottle_neck(compressed_bit, is_post_up_samling = False)
		compressed_bit = self.change_channel_my(compressed_bit, the_channel = 1, kernel_size = 9, activation = tf.nn.tanh)
		return post_down_sampling_hidden, compressed_bit

	def the_decoder_in_each_module(self, the_code, the_stride):
		compressed_bit = the_code
		compressed_bit = self.stack_four_bottle_neck(compressed_bit, is_post_up_samling = False)
		pre_up_sampling_hidden = compressed_bit
		compressed_bit = self.up_sampling_mod_my(compressed_bit, the_stride = the_stride)
		
		compressed_bit = self.stack_four_bottle_neck(compressed_bit, is_post_up_samling = False)
		expand_back = self.change_channel_my(compressed_bit, the_channel = 1, kernel_size = 9, activation=None)		
		return pre_up_sampling_hidden, expand_back


	def scalar_softmax_quantization(self, floating_code, alpha, bins, the_share, code_length, num_kmean_kernels):
		
		bins_expand = tf.expand_dims(bins, 1)
		bins_expand = tf.reshape(bins_expand, (1, 1, -1))			
		dist = tf.abs(floating_code - bins_expand)
		
		print(bins_expand.shape, floating_code.shape, dist.shape)
		soft_assignment = tf.nn.softmax(tf.multiply(alpha, dist)) # frame_length * 256
		soft_assignment_3d = soft_assignment
		# input()
		hard_assignment = tf.reshape(tf.one_hot(tf.nn.top_k(soft_assignment).indices, num_kmean_kernels), (-1, code_length, num_kmean_kernels))

		print('hard_assignment', hard_assignment.shape) #lpc ? 16 64
		print('soft_assignment', soft_assignment.shape) #lpc <unknown>
		soft_assignment = tf.cast((1 - the_share)*soft_assignment, tf.float32) + tf.cast(the_share* hard_assignment, tf.float32)
		# soft_assignment = hard_assignment

		soft_assignment = tf.reshape(soft_assignment, (-1, num_kmean_kernels))
		print(soft_assignment.shape, tf.expand_dims(bins, 1).shape)
		return soft_assignment_3d, soft_assignment, hard_assignment

	
	def computational_graph_end2end_quan_on(self, encoded, the_share, is_quan_on, number_bins, the_scope, the_strides):
		with tf.compat.v1.variable_scope(the_scope):		
			alpha = tf.Variable(init_alpha, dtype = tf.float32, name = 'alpha')
			bins = tf.Variable(np.linspace(-1, 1, number_bins), dtype = tf.float32, name = 'bins')
			# bins = tf.Variable(mu_law_bins, dtype = tf.float32, name = 'bins')
			hidden_1, floating_code = self.the_encoder_in_each_module(encoded, the_strides)			
			print('floating_code', floating_code.shape)

			# floating_code = differential_coding_subtract(floating_code)
			soft_assignment_3d, soft_assignment, hard_assignment = self.scalar_softmax_quantization(floating_code, alpha, bins, the_share, frame_length // the_strides, number_bins)
			bit_code = tf.reshape(tf.matmul(soft_assignment, tf.expand_dims(bins, 1)), (-1, int(frame_length/the_strides), 1))# 
			bit_code_hard = bit_code
			# bit_code = differential_coding_add(bit_code_hard)

			compressed_bit = bit_code
			print('compressed_bit shape: ', compressed_bit.shape)
			the_final_code = tf.cast((1 - is_quan_on) * floating_code, tf.float32) + tf.cast(is_quan_on* compressed_bit, tf.float32)
			the_final_code = tf.reshape(the_final_code, (-1, frame_length//the_strides, 1))
			hidden_2, expand_back = self.the_decoder_in_each_module(the_final_code, the_strides) # noquan
			print('model parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
			print('end2end output shape:', expand_back.shape)
			return soft_assignment_3d, -1, -1, bit_code_hard[0, : ,0], expand_back[:, :, 0], alpha, bins, hard_assignment#, hidden_1, hidden_2

	def model_training(self, sess, x, x_, mat, lr,  tau, the_share, is_quan_on, encoded, loss1, mfcc_loss, quan_loss, ent_loss, trainop2_list, decoded, alpha, bins, saver, the_learning_rate, epoch,  flag, interested_var = None, save_id = '', the_tau_val = 1.0):
		print('model_training is called.', flag)
		ave_snr, ave_stoi, ave_pesq = 0, 0, 0
		init_tau = the_tau_val

		for i in range(epoch):
			print('-----------------------')
			if flag == 'pretrain':
				is_quan_on_val = 0.0 if i < self._pretrain_step else 1.0
				if i < self._pretrain_step:
					trainop2 = trainop2_list[0]
					print('no quan op is used')
					
				else:
					trainop2 = trainop2_list[1] # normal ent
					print('quan op', init_tau)
					
			else:
				is_quan_on_val = 1.0
				trainop2 = trainop2_list[1]
				print('greedy_follower is working.', init_tau)
			
			print('-----------------------', i)

			start = time.perf_counter()

	
			for b_x, b_x_, weighting_mat in self._generate_one_epoch_end2end(self._tr_data, self._tr_data, self._batch_size):
				sess.run(trainop2, feed_dict={x: b_x, x_: b_x_, mat: weighting_mat, lr: the_learning_rate, the_share: 0.0, tau:init_tau, is_quan_on: is_quan_on_val})

			elapsed = time.perf_counter() - start
			np.random.shuffle(self._tr_data)

			if i % 1 == 0:								
				if type(decoded) == list:
					decoded = np.sum(decoded, axis=0)
				ave_snr, ave_stoi, ave_pesq, ave_linearity, _quan_loss, fully_snr, fully_pesq, fully_entropy = self.end2end_eval(sess, x, x_, mat, lr, the_share, is_quan_on, flag, loss1, encoded, decoded, alpha, bins, 100, i, interested_var)
			
			print('Epoch %3d: SNR: %7.5f dB    STOI: %6.5f    PESQ: %6.5f   linearity: %6.5f  modelid: %s  _quan_loss: %6.5f,  fully_entropy: %6.5f , time: %.3f, tau: %.3f' %(i, ave_snr, ave_stoi, ave_pesq, ave_linearity, self._rand_model_id, _quan_loss, fully_entropy, elapsed, init_tau))
			
			
			if flag != 'pretrain' and flag != 'the_follower':
				self._target_entropy *= 1.2
			
			ent_change = 0.015
			
			if fully_entropy > self._target_entropy + 0.05:
				init_tau += ent_change
				# pass
			elif fully_entropy < self._target_entropy:
				init_tau -= ent_change #* 3
				
			print('tau:', init_tau)

			self.write_to_file_and_update_to_display('Epoch %3d: SNR: %7.5f dB    STOI: %6.5f   PESQ: %6.5f   _quan_loss: %6.5f  tau: %6.5f   fully_pesq: %6.5f  fully_entropy: %6.5f \n' %(i, ave_snr, ave_stoi, ave_pesq, _quan_loss, init_tau, fully_pesq, fully_entropy))

			
		if flag == 'quan' and epoch==0:
			pass
			
		elif epoch !=0:
			save_path = saver.save(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + save_id + ".ckpt")
			print('Model saved!')
		else:
			print('Model not saved.')

	

	def utterance_to_segment(self, utterance, post_window = False):
		ret = np.empty((len(range(0, len(utterance) - frame_length, frame_length - overlap_each_side)), frame_length))
		ind = 0
		the_window = np.append(np.append(np.hanning(overlap_each_side*2)[:(overlap_each_side)],  np.array([1] * (frame_length - overlap_each_side*2))), np.hanning(overlap_each_side*2)[(overlap_each_side):])
		if post_window:
			for i in range(0, len(utterance) - frame_length, frame_length - overlap_each_side):
				# ret[ind, :] = utterance[i:i + frame_length] * the_window
				ret[ind, :] = utterance[i:i + frame_length] * 1
				ind += 1
		else:
			for i in range(0, len(utterance) - frame_length, frame_length - overlap_each_side):
				ret[ind, :] = utterance[i:i + frame_length] * the_window
				ind += 1
		return ret


	def hann_process(self, utterance_seg, seg_ind, seg_amount):
		hop_size = frame_length - overlap_each_side # followed bloomberg paper
		the_window = np.append(np.append(np.hanning(overlap_each_side*2)[:(overlap_each_side)],  np.array([1] * (frame_length - overlap_each_side*2))), np.hanning(overlap_each_side*2)[(overlap_each_side):])
		first_window= np.append(np.append(np.array([1]*(overlap_each_side)),  np.array([1] * (frame_length - overlap_each_side*2))), np.hanning(overlap_each_side*2)[(overlap_each_side):])
		last_window = first_window= np.append(np.append(np.hanning(overlap_each_side*2)[:(overlap_each_side)],  np.array([1] * (frame_length - overlap_each_side*2))), np.array([1]*(overlap_each_side)))
		if seg_ind == 0:
			utterance_seg = utterance_seg*first_window
		elif seg_ind == seg_amount - 1:
			utterance_seg = utterance_seg*last_window
		else:
			utterance_seg = utterance_seg*the_window
		return utterance_seg

	def end2end_eval(self, sess, x, x_, mat, lr, the_share, is_quan_on, flag, loss1, encoded, decoded, alpha, bins, how_many, the_epoch, interested_var = None):
		# if the_epoch < 95:
		# 	how_many = 20
		how_many = len(self._sep_val)
		min_len, snr_list = np.array([0] * how_many), np.array([0.0] * how_many)
		the_stoi, the_pesqs = np.array([0.0] * how_many), np.array([0.0] * how_many)

		fully_snr_list, fully_the_pesqs, fully_the_entropy = np.array([0.0] * how_many), np.array([0.0] * how_many), np.array([0.0] * how_many)

		_quan_loss_arr = np.array([0.0] * how_many)
		the_linearitys = np.array([0.0] * how_many)
		all_loss = [0.0] * how_many
		
		_alpha, _bins = 0, 0
		
		for i in range(how_many):
			per_sig, the_std = self._load_sig(self._sep_val[i])
			
			segments_per_utterance = self.utterance_to_segment(per_sig, True)
			
			_decoded_sig = np.array([0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))
			code_segment_len = int(frame_length/self._the_strides[0])
			_encoded_sig = np.array([0.0] * (code_segment_len + (code_segment_len) * (segments_per_utterance.shape[0] - 1))).astype(np.float32)
			_quan_loss_arr_each = np.array([0.0]*segments_per_utterance.shape[0])
			entropy_per_segment = np.array([0.0]*segments_per_utterance.shape[0])
			lpc_entropy = np.array([0.0]*segments_per_utterance.shape[0])

			for j in range(segments_per_utterance.shape[0]):
				feed_x = np.reshape(segments_per_utterance[j], (1, frame_length, 1))
				_interested_var, the_loss, _encoded, _decoded, _alpha, _bins = sess.run([interested_var, loss1, encoded, decoded, alpha, bins], feed_dict = 
					{x: feed_x, 
					x_: feed_x, 
					mat: np.ones((1, int(frame_length/2)+1)),
					lr: 0.0,
					the_share: 1.0,
					is_quan_on: 1.0}) # share is 0 means it's soft, 1 means it's hard 
				
				
				_decoded_sig[j * (frame_length - overlap_each_side): j * (frame_length - overlap_each_side) + frame_length] += self.hann_process(_decoded.flatten(), j, segments_per_utterance.shape[0])				
				_encoded_sig[j * (code_segment_len): j * (code_segment_len) + code_segment_len] += _encoded#[0,:]
				
				_quan_loss_arr_each[j] = _interested_var[2]
				if isinstance(_interested_var[5], np.ndarray):
					entropy_per_segment[j] = np.mean(_interested_var[5])
				else:
					entropy_per_segment[j] = _interested_var[5]
			
			per_sig *= the_std
			_decoded_sig  *= the_std
			np.save('the_code_' + self._rand_model_id+'.npy', _encoded_sig)

			fully_the_entropy[i] = np.mean(entropy_per_segment)
			lpc_entropy[i] = _interested_var[-2]


			min_len[i], snr_list[i], the_stoi[i], the_pesqs[i], the_linearitys[i] = self.end2end_final_eval(per_sig, _decoded_sig)
			print('signal ', i, 'snr: ', snr_list[i])
			_quan_loss_arr[i] = np.mean(_quan_loss_arr_each)
			

		sdr_return_it = np.sum(min_len*snr_list/np.sum(min_len))
		stoi_return_it = np.sum(min_len*the_stoi/np.sum(min_len))
		pesq_return_it = np.sum(min_len*the_pesqs/np.sum(min_len))
		linearity_return_it = np.sum(min_len*the_linearitys/np.sum(min_len))
		quan_return_it = np.sum(min_len*_quan_loss_arr/np.sum(min_len))

		fully_sdr_return_it = np.sum(min_len*fully_snr_list/np.sum(min_len))
		fully_pesq_return_it = np.sum(min_len*fully_the_pesqs/np.sum(min_len))
		fully_entropy_return_it = np.sum(min_len*fully_the_entropy/np.sum(min_len))		
		return sdr_return_it, stoi_return_it, pesq_return_it, linearity_return_it, quan_return_it, fully_sdr_return_it, fully_pesq_return_it, fully_entropy_return_it
	

	def one_model(self, arg):		
		with tf.Graph().as_default():
			x, x_, lr, the_share = self.init_placeholder_end_to_end()
			mat = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None, int(frame_length/2)+1), name = 'mat')
			
			tau = tf.compat.v1.placeholder(dtype = tf.float32, shape = None, name = 'tau')
			is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')

		
			_softmax_assignment, weight, decoded_fully, encoded, decoded, alpha, bins, _hard_assignment = self.computational_graph_end2end_quan_on(
				x, 
				the_share, 
				is_quan_on, 
				self._num_bins_for_follower[0], 
				'scope_1', 
				self._the_strides[0])
			time_loss = mse_loss(decoded, x_[:, :, 0])
			freq_loss = mfcc_loss(decoded, x_[:, :, 0])
			

			quantization_loss = quan_loss(_softmax_assignment)
			ent_loss = entropy_coding_loss(_softmax_assignment, self._the_strides[0])
			
			loss_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
			+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat))
			loss_quan_init = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
			+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat)) + self._coeff_term[4] * quantization_loss + tau * ent_loss
			

			trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).minimize(loss_no_quan,
						var_list=tf.compat.v1.trainable_variables())
			trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).minimize(loss_quan_init,
						var_list=tf.compat.v1.trainable_variables())
			trainop2_list = [trainop2_no_quan, trainop2_quan_init]
			saver = tf.compat.v1.train.Saver()
			interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss, encoded]
			with tf.compat.v1.Session() as sess:					
				sess.run(tf.compat.v1.global_variables_initializer())			
				self.model_training(sess, x=x, x_=x_, lr=lr, mat=mat, the_share=the_share,  tau=tau, 
					is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
					quan_loss=quan_loss, ent_loss = ent_loss, trainop2_list=trainop2_list, decoded=decoded, alpha=alpha,
					bins=bins, saver=saver,
					the_learning_rate=self._learning_rate_tanh, epoch=self._epoch_tanh,
					flag='pretrain', interested_var=interested_var, save_id='',
					the_tau_val=self._coeff_term[5])
