from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from celluloid import Camera

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt 

import params as yamnet_params
import yamnet as yamnet_model



def main(argv):

	params = yamnet_params.Params()
	yamnet = yamnet_model.yamnet_frames_model(params)
	yamnet.load_weights('yamnet.h5')
	yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

	for file_name in argv:
		# Decode the WAV file.
		wav_data, sr = sf.read(file_name,always_2d=False, dtype=np.int16)
		assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

		print('waveform original dtaa',wav_data.shape)


		waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
		waveform = waveform.astype('float32')
		print('waveform normal dtaa',waveform.shape)


		print('sampling rate',sr)
		print('sampling rate model params',params.sample_rate)

		# Convert to mono and the sample rate expected by YAMNet.
		if len(waveform.shape) > 1:
			print('entered')
			waveform = np.mean(waveform, axis=1)
		if sr != params.sample_rate:
			waveform = resampy.resample(waveform, sr, params.sample_rate)
		print('waveform normal dtaa',waveform.shape)



		scale=2.5

		# fig = plt.figure(figsize=(int(scale*4), int(scale*3)))
		# camera = Camera(fig)

		# for i in range(0,len(waveform),int(0.96*params.sample_rate/int(8))):
		# 	plt.plot(waveform[:i],color='b')
		# 	plt.xlabel('Samples')
		# 	plt.ylabel('Amplitude')
		# 	camera.snap()
		# animation = camera.animate()
		# animation.save(file_name+'_filename_'+str(scale)+'.mp4')
		# plt.close()

		# Predict YAMNet classes.
		scores, embeddings, spectrogram = yamnet(waveform)
		print('scores',scores)
		# Scores is a matrix of (time_frames, num_classes) classifier scores.
		# Average them along time to get an overall classifier output for the clip.
		prediction = np.mean(scores, axis=0)
		# Report the highest-scoring classes and their scores.
		top5_i = np.argsort(prediction)[::-1][:5]
		print(file_name, ':\n' +
		'\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
		for i in top5_i))

		# colors=['b','g','r']
		# fig=plt.figure()
		# camera = Camera(fig)
		# plt.xlabel('Time(0.5s)')
		# plt.ylabel('Probability')
		# for j in range(1,len(scores)):
		# 	k=0
		# 	for i in top5_i[1:-1]:

		# 		x=np.convolve(scores[:j,i].numpy(), np.ones((4,))/4, mode='valid')
		# 		# x=scores[:j,i].numpy()
		# 		plt.plot(x,color=colors[k])
		# 		k+=1
		# 	for i in range(1):

		# 		camera.snap()
		# plt.legend([yamnet_classes[i] for i in top5_i[1:-1]],loc='upper right')
		# animation = camera.animate(interval=int(1000))

		# # plt.show()
		# # plt.close()
		# animation.save(file_name+'_class_'+str(scale)+'.mp4')



		colors=['b','g','r']
		fig=plt.figure()
		camera = Camera(fig)
		plt.xlabel('Time(0.5s)')
		plt.ylabel('volume')
		vol_store=[]
		total_vol=0
		for j in range(len(scores)):

			vol=[]
			for i in top5_i[1:-1]:

				# x=np.convolve(scores[j,i].numpy(), np.ones((4,))/4, mode='valid')
				x=scores[j,i].numpy()
				if x>0.1:
					vol.append(float(1/24))
			# print(vol)
			if vol:
				total_vol+=np.mean(vol)
			print(total_vol)
			vol_store.append(total_vol)
			# print(vol_store)
			plt.plot(vol_store,color='b')
			camera.snap()
		# plt.legend(,loc='upper right')
		animation = camera.animate(interval=int(1000))

		# plt.show()
		# plt.close()
		animation.save(file_name+'_volume_'+str(scale)+'.mp4')





if __name__ == '__main__':
  main(sys.argv[1:])