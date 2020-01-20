"""
File name: cmrl.py
Description: Implementation of cross-module residual learning.
			 Current version supports 2 cascaded coding modules.
"""
from ac_ae_module import *

class CMRL(AudioCodingAE):
	"""docstring for CMRL"""
	def __init__(self, arg):
		super(CMRL, self).__init__(arg)
		self._num_resnets = arg.num_resnets
		self._from_where_step = int(arg.from_where_step)
		self._learning_rate_greedy_followers = list(map(lambda x: float(x), arg.learning_rate_greedy_followers.split()))


	def _greedy_followers(self, num_res):
		#_greedy_followers is called
		# tf.compat.v1.reset_default_graph()
		with tf.Graph().as_default():
			x, x_, lr, the_share = self.init_placeholder_end_to_end()
			mat = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None, int(frame_length/2)+1), name = 'mat')
			tau = tf.compat.v1.placeholder(dtype = tf.float32, shape = None, name = 'tau')
			is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')

			residual_coding_x = [None] * num_res
			all_var_list = []

			for i in range(0, num_res - 1):
				if i == 0:
					_res_scalar_first = 1
				
					_softmax_assignment_1, weight, dist, encoded, residual_coding_x[0], alpha, bins, soft_assignment_fully_1  = self.computational_graph_end2end_quan_on( 
						x * _res_scalar_first, 
						the_share, 
						is_quan_on, 
						self._num_bins_for_follower[0], 
						'scope_1', 
						self._the_strides[0])
					residual_coding_x[0]  = residual_coding_x[0]/_res_scalar_first
					all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_1')					
				else:
					pass
					
			saver = tf.compat.v1.train.Saver(var_list=all_var_list)
			with tf.compat.v1.Session() as sess:
				# 1136741: 4.2 after pretrain at 15, with 3.8 ent. 0.89m param.
				if num_res==2:				
					saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' +".ckpt")
				else:
					saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + '_follower_' + str(num_res - 2) + self._suffix + ".ckpt")
					print('model' + self._rand_model_id + '_follower_' + str(num_res - 2) + ' is restored!')
				
				_softmax_assignment, weight, dist, encoded, residual_coding_x[-1], alpha, bins, soft_assignment_fully = self.computational_graph_end2end_quan_on(
					(x - tf.expand_dims(tf.reduce_sum(input_tensor=residual_coding_x[:-1], axis=0), axis=2)) * self._res_scalar, 
					the_share, 
					is_quan_on, 
					self._num_bins_for_follower[i], 
					'scope_' + str(num_res), 
					self._the_strides[i])
# # 
				residual_coding_x[-1] = residual_coding_x[-1]/self._res_scalar
				all_var_list = []
				for i in range(num_res):
					all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_' + str(i + 1))
				# all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='lpc_quan')
				saver = tf.compat.v1.train.Saver(var_list=all_var_list)
				
				decoded =  np.sum(residual_coding_x, axis=0)
				time_loss = mse_loss(decoded, x_[:, :, 0]) #- si_snr_loss((decoded), (res_x[:, :, 0])) * 0.2 # - si_snr_loss(_hidden_1, _hidden_2) 

				freq_loss = mfcc_loss(decoded, x_[:, :, 0])
				# freq_loss = 5*stft_loss(decoded, x_[:, :, 0], mat)

				quantization_loss = quan_loss(_softmax_assignment)
				
				ent_loss =  entropy_coding_loss(_softmax_assignment, self._the_strides[1])
				interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss, ent_loss, encoded]
				
				
				loss2_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
				+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat))
				loss2_quan_init = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
				+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat)) + self._coeff_term[4] * quantization_loss + tau * ent_loss
			

				trainop2_no_quan         = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_no_quan        , var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_2'))
				trainop2_quan_init       = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_quan_init      , var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_2'))				
				
				trainop2_list = [trainop2_no_quan, trainop2_quan_init]

				adam_vars = [var for var in tf.compat.v1.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]

				sess.run(tf.compat.v1.variables_initializer(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_' + str(num_res)) + adam_vars))				

				self.model_training(sess, x=x, x_=x_, lr=lr, mat=mat, the_share=the_share, tau=tau, 
						is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
						quan_loss=quan_loss, ent_loss = ent_loss, trainop2_list=trainop2_list, decoded=decoded, alpha=alpha,
						bins=bins, saver=saver,
						the_learning_rate=self._learning_rate_greedy_followers[-2], epoch=self._epoch_greedy_followers[-2],
						flag='the_follower', interested_var=interested_var, save_id='follower_' + str(num_res - 1) + self._suffix,
						the_tau_val=self._coeff_term[5]*0.01)

	def all_modules_feedforward(self):
		x, x_, lr, the_share = self.init_placeholder_end_to_end()
		mat = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None, int(frame_length/2)+1), name = 'mat')
		tau = tf.compat.v1.placeholder(dtype = tf.float32, shape = None, name = 'tau')
		is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')

		
		residual_coding_x = [None] * self._num_resnets

		for i in range(self._num_resnets):
			if i == 0:
				_res_scalar_first = 1
				_softmax_assignment_1, weight, dist, encoded_1, residual_coding_x[0], alpha, bins_1, soft_assignment_fully_1 = self.computational_graph_end2end_quan_on(
					_res_scalar_first*x, 
					# mu_law_mapping(res_x)* _res_scalar_first,  
					the_share, 
					is_quan_on, 
					self._num_bins_for_follower[i], 
					'scope_1', 
					self._the_strides[i])			

				residual_coding_x[0] = residual_coding_x[0]/_res_scalar_first
			else:				
				_softmax_assignment_2, weight, dist, encoded_2, residual_coding_x[i], alpha, bins_2, soft_assignment_fully_2 = self.computational_graph_end2end_quan_on(
					self._res_scalar*(x- tf.expand_dims(tf.reduce_sum(input_tensor=residual_coding_x[:-1], axis=0), axis=2)), 				
					the_share, 
					is_quan_on, 
					self._num_bins_for_follower[i], 
					'scope_' + str(i + 1), 
					self._the_strides[i])			

				residual_coding_x[i] = residual_coding_x[i]/self._res_scalar		
		if self._num_resnets==1:
			_softmax_assignment_2, encoded_2, bins_2, soft_assignment_fully_2 = _softmax_assignment_1, encoded_1, bins_1, soft_assignment_fully_1
		return x, x_, mat, lr, tau, the_share, is_quan_on, _softmax_assignment_1, _softmax_assignment_2, encoded_1, encoded_2, soft_assignment_fully_1, soft_assignment_fully_2, residual_coding_x, alpha, bins_1, bins_2


	def cmrl_eval(self, sess, x, x_, mat, lr, tau, the_share, is_quan_on, loss1, encoded_1, encoded_2, decoded, alpha, bins, how_many, the_epoch, interested_var = None):		
		num_of_test_files = len(self._sep_test_files_10)
		min_len, snr_list = np.array([0] * num_of_test_files), np.array([0.0] * num_of_test_files)
		the_stoi, the_pesqs = np.array([0.0] * num_of_test_files), np.array([0.0] * num_of_test_files)
		_quan_loss_arr = np.array([0.0] * num_of_test_files)
		the_linearitys = np.array([0.0] * num_of_test_files)
		all_loss = [0.0] * num_of_test_files
		_alpha, _bins = 0, 0
		
		if os.path.exists('./end2end_performance/'+str(self._base_model_id)):
			pass
		else:
			os.mkdir('./end2end_performance/'+str(self._base_model_id))

		for i in range(num_of_test_files):
			per_sig, the_std = self._load_sig(self._sep_test_files_10[i])
			segments_per_utterance = self.utterance_to_segment(per_sig, True)

			_decoded_sig = np.array([0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))
			code_segment_len = int(frame_length/self._the_strides[0])
			_encoded_sig_1 = np.array([0.0] * (code_segment_len + (code_segment_len) * (segments_per_utterance.shape[0] - 1))).astype(np.float32)
			_encoded_sig_2 = np.array([0.0] * (code_segment_len + (code_segment_len) * (segments_per_utterance.shape[0] - 1))).astype(np.float32)

			all_entropy = np.array([0.0]*segments_per_utterance.shape[0])
			for j in range(segments_per_utterance.shape[0]):
				feed_x = np.reshape(segments_per_utterance[j], (1, frame_length, 1))
				_interested_var, the_loss, _encoded_1, _encoded_2, _decoded, _alpha, _bins = sess.run([interested_var, loss1, encoded_1, encoded_2, decoded, alpha, bins], feed_dict = 
					{x: feed_x, 
					x_: feed_x, 
					lr: 0.0,
					the_share: 1.0,
					is_quan_on: 1.0}) # share is 0 means it's soft, 1 means it's hard 
				
				_decoded_sig[j * (frame_length - overlap_each_side): j * (frame_length - overlap_each_side) + 512] += self.hann_process(_decoded.flatten(), j, segments_per_utterance.shape[0])			
				_encoded_sig_1[j * (code_segment_len): j * (code_segment_len) + code_segment_len] = _encoded_1#[0,:]
				_encoded_sig_2[j * (code_segment_len): j * (code_segment_len) + code_segment_len] = _encoded_2#[0,:]
				all_entropy[j] = _interested_var[-1]
			per_sig *= the_std
			_decoded_sig  *= the_std
			
			min_len[i], snr_list[i], the_stoi[i], the_pesqs[i], the_linearitys[i] = self.end2end_final_eval(per_sig, _decoded_sig)			
			
			print('Test Utterance %1d: SNR: %7.5f dB  PESQ-WB: %6.5f  Entropy: %6.5f  Bit rate: %6.5f  ID: %s' %(i, snr_list[i], the_pesqs[i], np.mean(all_entropy), entropy_to_bitrate(np.mean(all_entropy), self._the_strides[0]), self._rand_model_id))
			self.write_to_file_and_update_to_display('Test Utterance %1d: SNR: %7.5f dB  PESQ-WB: %6.5f   Entropy: %6.5f \n' %(i, snr_list[i], the_pesqs[i], np.mean(all_entropy)))

			np.save('./end2end_performance/'+str(self._base_model_id)+'/ori_sig'+'_'+str(the_epoch)+'_'+str(i)+'.npy', per_sig)
			np.save('./end2end_performance/'+str(self._base_model_id)+'/'+self._sep_test_files_10[i].split('/')[-1][:-4]+'.npy', _decoded_sig)

			np.save('./end2end_performance/'+str(self._base_model_id)+'/enc_sig' + '_' + '_' + str(i) + '_1.npy', _encoded_sig_1)
			np.save('./end2end_performance/'+str(self._base_model_id)+'/enc_sig' + '_' + '_' + str(i) + '_2.npy', _encoded_sig_2)

		sdr_return_it = np.sum(min_len*snr_list/np.sum(min_len))
		stoi_return_it = np.sum(min_len*the_stoi/np.sum(min_len))
		pesq_return_it = np.sum(min_len*the_pesqs/np.sum(min_len))
		linearity_return_it = np.sum(min_len*the_linearitys/np.sum(min_len))
		np.save('./end2end_performance/min_len_'+self._base_model_id+'.npy', min_len)
		np.save('./end2end_performance/snr_'+self._base_model_id+'.npy', snr_list)
		np.save('./end2end_performance/pesq_'+self._base_model_id+'.npy', the_pesqs)


	def model(self, training_mode, arg):		
		if training_mode=='one_module':
			self.one_model(arg)

		elif training_mode=='cascaded':
			self.one_model(arg)
			# self._rand_model_id = self._base_model_id
			for i in range(2, self._num_resnets + 1):
				self._greedy_followers(i)

			with tf.Graph().as_default():				
				x, x_, mat, lr, tau, the_share, is_quan_on, _softmax_assignment_1, _softmax_assignment_2, encoded_1, encoded_2, _soft_assignment_fully_1, _soft_assignment_fully_2, residual_coding_x, alpha, bins_1, bins_2 = self.all_modules_feedforward()
				
				saver = tf.compat.v1.train.Saver()			
				with tf.compat.v1.Session() as sess:
					saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix + ".ckpt")
					print("./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix, ' restored!')				
					decoded = np.sum(residual_coding_x, axis = 0)
					time_loss = mse_loss(decoded, x_[:, :, 0]) 					
					freq_loss = mfcc_loss(decoded, x_[:, :, 0])
					quantization_loss = quan_loss(_softmax_assignment_1) + quan_loss(_softmax_assignment_2)
					
					ent_loss_1 =  entropy_coding_loss(_softmax_assignment_1, self._the_strides[1])
					ent_loss_2 =  entropy_coding_loss(_softmax_assignment_2, self._the_strides[1])
					ent_loss = ent_loss_1 + ent_loss_2
					ent_loss_1_fully =  entropy_coding_loss(_soft_assignment_fully_1, self._the_strides[1])
					ent_loss_2_fully =  entropy_coding_loss(_soft_assignment_fully_2, self._the_strides[1])
					ent_loss_fully = ent_loss_1_fully + ent_loss_2_fully
					
					interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss_fully, ent_loss, ent_loss_fully]
					# FOR TWO RESNETS!
					if self._num_resnets == 2:
						loss2_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
						+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat))
						loss2_quan_init = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
						+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat)) + self._coeff_term[4] * quantization_loss + tau * ent_loss
						
						trainop2_no_quan         = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_no_quan        , var_list = tf.compat.v1.trainable_variables())
						trainop2_quan_init       = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_quan_init      , var_list = tf.compat.v1.trainable_variables())				
						trainop2_list = [trainop2_no_quan, trainop2_quan_init]
						
						adam_vars = [var for var in tf.compat.v1.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
						sess.run(tf.compat.v1.variables_initializer(adam_vars))
						self.model_training(sess, x=x, x_=x_, lr=lr, mat=mat, the_share=the_share, tau=tau, 
								is_quan_on=is_quan_on, encoded=encoded_1, loss1=time_loss, mfcc_loss=freq_loss,
								quan_loss=quan_loss, ent_loss = ent_loss, trainop2_list=trainop2_list, decoded=decoded, alpha=alpha,
								bins=bins_1, saver=saver,
								the_learning_rate=self._learning_rate_greedy_followers[-1], epoch=self._epoch_greedy_followers[-1],
								flag='finetuning', interested_var=interested_var, save_id='follower_all'+ self._suffix,
								the_tau_val=self._coeff_term[5])
		
		elif training_mode=='feedforward':	
			model_id = arg.base_model_id
			with tf.Graph().as_default():
				x, x_, mat, lr, tau, the_share, is_quan_on, _, _, encoded_1, encoded_2, _soft_assignment_fully_1, _soft_assignment_fully_2, residual_coding_x, alpha, bins_1, bins_2 = self.all_modules_feedforward()
				decoded = np.sum(residual_coding_x, axis = 0)	

				time_loss = mse_loss(decoded, x_[:, :, 0])
				freq_loss = mfcc_loss(decoded, x_[:, :, 0])
				ent_loss_1 = entropy_coding_loss(_soft_assignment_fully_1, self._the_strides[0]) 
				ent_loss_2 = entropy_coding_loss(_soft_assignment_fully_2, self._the_strides[1])				
				ent_loss = ent_loss_1 + ent_loss_2
				if self._num_resnets == 1:
					ent_loss = ent_loss_1
				
				interested_var = [time_loss, freq_loss, ent_loss_1, ent_loss_2, ent_loss]
				saver = tf.compat.v1.train.Saver()	
				with tf.compat.v1.Session() as sess:
					if self._num_resnets== 1:
						saver.restore(sess, "./check/model_bnn_ac_" + model_id +'_'+ ".ckpt")			
						print("./check/model_bnn_ac_" + model_id+'_' + ".ckpt")
				
					elif self._num_resnets== 2:
						saver.restore(sess, "./check/model_bnn_ac_" + model_id + '_' + self._save_unique_mark + self._suffix + ".ckpt")
						print("./check/model_bnn_ac_" + model_id + '_finetuning_' + self._save_unique_mark + self._suffix + ".ckpt") 						
					else:	
						print('Wrong setting.')
					self.cmrl_eval(sess, x, x_, mat, lr, tau, the_share, is_quan_on, time_loss, encoded_1, encoded_2, decoded, alpha, bins_1, bins_2 , 100, 30, interested_var)
		
		elif training_mode == 'retrain_from_somewhere':
			if len(self._the_strides) == 1:
				pass
			else:
				print('retrain_from_somewhere!')
				if self._from_where_step == -1:
					self._rand_model_id = arg.base_model_id
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
						# freq_loss = 5*stft_loss(decoded, x_[:, :, 0], mat)

						quantization_loss = quan_loss(_softmax_assignment)
						
						ent_loss =  entropy_coding_loss(_softmax_assignment, self._the_strides[1])
						
						# interested_var = [time_loss, freq_loss, quantization_loss, _softmax_assignment, ent_loss, ent_loss, encoded]
						interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss, encoded]
						loss2_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
						+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat))
						loss2_quan_init = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
						+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat)) + self._coeff_term[4] * quantization_loss + tau * ent_loss
						
						trainop2_no_quan         = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_no_quan        , var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_1'))
						trainop2_quan_init       = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_quan_init      , var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_1'))				
						
						trainop2_list = [trainop2_no_quan, trainop2_quan_init]

						adam_vars = [var for var in tf.compat.v1.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name] 
						saver = tf.compat.v1.train.Saver()
						# saver = tf.train.Saver()
						with tf.compat.v1.Session() as sess:
							saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' +".ckpt")
							print('./check/model_bnn_ac_'+ self._rand_model_id + ' loaded!')
							self.model_training(sess, x=x, x_=x_, lr=lr, mat=mat, the_share=the_share, tau=tau, 
									is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
									quan_loss=quan_loss, ent_loss = ent_loss, trainop2_list=trainop2_list, decoded=decoded, alpha=alpha,
									bins=bins, saver=saver,
									the_learning_rate=self._learning_rate_greedy_followers[-2], epoch=self._epoch_greedy_followers[-2],
									flag='the_follower', interested_var=interested_var, save_id='follower_0_finetuning'  + self._suffix,
									the_tau_val=self._coeff_term[5]*1)
						
				elif self._from_where_step == 0:
					self._rand_model_id = arg.base_model_id
					self._greedy_followers(2)
					with tf.Graph().as_default():				
						x, x_, mat, lr, tau, the_share, is_quan_on, _softmax_assignment_1, _softmax_assignment_2, encoded_1, encoded_2, _soft_assignment_fully_1, _soft_assignment_fully_2, residual_coding_x, alpha, bins_1, bins_2 = self.all_modules_feedforward()
						saver = tf.compat.v1.train.Saver()			
						with tf.compat.v1.Session() as sess:
							print("./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix, ' restored!')
							saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix + ".ckpt")
							print("./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix, ' restored!')				
							decoded = np.sum(residual_coding_x, axis = 0)
							time_loss = mse_loss(decoded, x_[:, :, 0]) 
							
							freq_loss = mfcc_loss(decoded, x_[:, :, 0]) 
							
							quantization_loss = quan_loss(_softmax_assignment_1) + quan_loss(_softmax_assignment_2)
							
							ent_loss_1 =  entropy_coding_loss(_softmax_assignment_1, self._the_strides[1])
							ent_loss_2 =  entropy_coding_loss(_softmax_assignment_2, self._the_strides[1])
							ent_loss = ent_loss_1 + ent_loss_2
							ent_loss_1_fully =  entropy_coding_loss(_soft_assignment_fully_1, self._the_strides[1])
							ent_loss_2_fully =  entropy_coding_loss(_soft_assignment_fully_2, self._the_strides[1])
							ent_loss_fully = ent_loss_1_fully + ent_loss_2_fully
							
							# interested_var = [time_loss, freq_loss, quantization_loss, ent_loss, ent_loss_fully]
							interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss_fully, ent_loss, ent_loss_fully]
							# interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss_fully, ent_loss_fully, ent_loss_fully, encoded]
							# FOR TWO RESNETS!
							if self._num_resnets == 2:
								loss2_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
								+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat))
								loss2_quan_init = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
								+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat)) + self._coeff_term[4] * quantization_loss + tau * ent_loss
								
								trainop2_no_quan         = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_no_quan        , var_list = tf.compat.v1.trainable_variables())
								trainop2_quan_init       = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_quan_init      , var_list = tf.compat.v1.trainable_variables())				
								trainop2_list = [trainop2_no_quan, trainop2_quan_init]
								
								adam_vars = [var for var in tf.compat.v1.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
								sess.run(tf.compat.v1.variables_initializer(adam_vars))
								self.model_training(sess, x=x, x_=x_, lr=lr, mat=mat, the_share=the_share, tau=tau, 
										is_quan_on=is_quan_on, encoded=encoded_1, loss1=time_loss, mfcc_loss=freq_loss,
										quan_loss=quan_loss, ent_loss = ent_loss, trainop2_list=trainop2_list, decoded=decoded, alpha=alpha,
										bins=bins_1, saver=saver,
										the_learning_rate=self._learning_rate_greedy_followers[-1], epoch=self._epoch_greedy_followers[-1],
										flag='finetuning', interested_var=interested_var, save_id='follower_all'+ self._suffix,
										the_tau_val=self._coeff_term[5])
				elif self._from_where_step == 1:
					self._rand_model_id = arg.base_model_id
					# self._greedy_followers(2)
					with tf.Graph().as_default():				
						x, x_, mat, lr, tau, the_share, is_quan_on, _softmax_assignment_1, _softmax_assignment_2, encoded_1, encoded_2, _soft_assignment_fully_1, _soft_assignment_fully_2, residual_coding_x, alpha, bins_1, bins_2 = self.all_modules_feedforward()
						saver = tf.compat.v1.train.Saver()			
						with tf.compat.v1.Session() as sess:
							print("./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix, ' restored!')
							saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix + ".ckpt")
							print("./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_' + str(self._num_resnets-1) + self._suffix, ' restored!')				
							decoded = np.sum(residual_coding_x, axis = 0)
							time_loss = mse_loss(decoded, x_[:, :, 0]) 
							
							freq_loss = mfcc_loss(decoded, x_[:, :, 0])
							# freq_loss = 5*stft_loss(decoded, x_[:, :, 0], mat)
							quantization_loss = quan_loss(_softmax_assignment_1) + quan_loss(_softmax_assignment_2)
							
							ent_loss_1 =  entropy_coding_loss(_softmax_assignment_1, self._the_strides[1])
							ent_loss_2 =  entropy_coding_loss(_softmax_assignment_2, self._the_strides[1])
							ent_loss = ent_loss_1 + ent_loss_2
							ent_loss_1_fully =  entropy_coding_loss(_soft_assignment_fully_1, self._the_strides[1])
							ent_loss_2_fully =  entropy_coding_loss(_soft_assignment_fully_2, self._the_strides[1])
							ent_loss_fully = ent_loss_1_fully + ent_loss_2_fully
							
							# interested_var = [time_loss, freq_loss, quantization_loss, ent_loss, ent_loss_fully]
							interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss_fully, ent_loss, ent_loss_fully]
							# interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss_fully, ent_loss_fully, ent_loss_fully, encoded]
							# FOR TWO RESNETS!
							if self._num_resnets == 2:
								mfcc_scalar = self._coeff_term[1]
								quan_scalar = self._coeff_term[2]
								loss2_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
								+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat))
								loss2_quan_init = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
								+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat)) + self._coeff_term[4] * quantization_loss + tau * ent_loss
								
								trainop2_no_quan         = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_no_quan        , var_list = tf.compat.v1.trainable_variables())
								trainop2_quan_init       = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_quan_init      , var_list = tf.compat.v1.trainable_variables())				
								trainop2_list = [trainop2_no_quan, trainop2_quan_init]
								
								adam_vars = [var for var in tf.compat.v1.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
								sess.run(tf.compat.v1.variables_initializer(adam_vars))
								self.model_training(sess, x=x, x_=x_, lr=lr, mat=mat, the_share=the_share, tau=tau, 
										is_quan_on=is_quan_on, encoded=encoded_1, loss1=time_loss, mfcc_loss=freq_loss,
										quan_loss=quan_loss, ent_loss = ent_loss, trainop2_list=trainop2_list, decoded=decoded, alpha=alpha,
										bins=bins_1, saver=saver,
										the_learning_rate=self._learning_rate_greedy_followers[-1], epoch=self._epoch_greedy_followers[-1],
										flag='finetuning', interested_var=interested_var, save_id='follower_all'+ self._suffix,
										the_tau_val=self._coeff_term[5])
				else:
					self._rand_model_id = arg.base_model_id
					with tf.Graph().as_default():				
						x, x_, mat, lr, tau, the_share, is_quan_on, _softmax_assignment_1, _softmax_assignment_2, encoded_1, encoded_2, _soft_assignment_fully_1, _soft_assignment_fully_2, residual_coding_x, alpha, bins_1, bins_2 = self.all_modules_feedforward()
						saver = tf.compat.v1.train.Saver()			
						with tf.compat.v1.Session() as sess:
							saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_all'  + self._suffix + ".ckpt")
							print("./check/model_bnn_ac_" + self._rand_model_id + '_' + 'follower_all'  + self._suffix, ' restored!')				
							decoded = np.sum(residual_coding_x, axis = 0)
							time_loss = mse_loss(decoded, x_[:, :, 0]) 
							
							freq_loss = mfcc_loss(decoded, x_[:, :, 0]) 
							quantization_loss = quan_loss(_softmax_assignment_1) + quan_loss(_softmax_assignment_2)
							
							ent_loss_1 =  entropy_coding_loss(_softmax_assignment_1, self._the_strides[1])
							ent_loss_2 =  entropy_coding_loss(_softmax_assignment_2, self._the_strides[1])
							ent_loss = ent_loss_1 + ent_loss_2
							ent_loss_1_fully =  entropy_coding_loss(_soft_assignment_fully_1, self._the_strides[1])
							ent_loss_2_fully =  entropy_coding_loss(_soft_assignment_fully_2, self._the_strides[1])
							ent_loss_fully = ent_loss_1_fully + ent_loss_2_fully
							interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss_fully, ent_loss, ent_loss_fully]
							# FOR TWO RESNETS!
						
							loss2_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
							+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat))
							loss2_quan_init = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss + self._coeff_term[2] * SMR(decoded, x_[:, :, 0], mat) 
							+ self._coeff_term[3] * (MNR(decoded, x_[:, :, 0], mat) + MNR_reduce_min(decoded, x_[:, :, 0], mat)) + self._coeff_term[4] * quantization_loss + tau * ent_loss
							
							trainop2_no_quan         = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_no_quan        , var_list = tf.compat.v1.trainable_variables())
							trainop2_quan_init       = tf.compat.v1.train.AdamOptimizer(lr, beta2 = 0.999).minimize(loss2_quan_init      , var_list = tf.compat.v1.trainable_variables())				
							trainop2_list = [trainop2_no_quan, trainop2_quan_init]
							
							adam_vars = [var for var in tf.compat.v1.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
							sess.run(tf.compat.v1.variables_initializer(adam_vars))
							self.model_training(sess, x=x, x_=x_, lr=lr, mat=mat, the_share=the_share, tau=tau, 
									is_quan_on=is_quan_on, encoded=encoded_1, loss1=time_loss, mfcc_loss=freq_loss,
									quan_loss=quan_loss, ent_loss = ent_loss, trainop2_list=trainop2_list, decoded=decoded, alpha=alpha,
									bins=bins_1, saver=saver,
									the_learning_rate=self._learning_rate_greedy_followers[-1], epoch=self._epoch_greedy_followers[-1],
									flag='finetuning', interested_var=interested_var, save_id='follower_all'+ self._suffix,
									the_tau_val=self._coeff_term[5])

