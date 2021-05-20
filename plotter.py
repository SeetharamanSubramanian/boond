from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt 

import params as yamnet_params
import yamnet as yamnet_model

# plt.style.use('ggplot')


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

		print(waveform.shape,min(waveform))
		# plt.figure(figsize=(20, 8))
		# plt.plot(waveform)
		# plt.xlabel('Samples')
		# plt.ylabel('Amplitude')
		# # plt.savefig('waveform.png')
		# plt.show()
		# plt.close()

		# fig, ax = plt.subplots(figsize=(20, 8))
		fig = plt.figure()
		ax = plt.axes(xlim=(0, len(waveform)), ylim=(-0.16, 0.17))

		line, = ax.plot([], [], lw=1)

		def init():
			line.set_data([], [])
			return line,

		def animate(i):
		    x = np.linspace(0,len(waveform), len(waveform))
		    y = waveform[i]
		    line.set_data(x, y)
		    return line,

		anim = FuncAnimation(fig, animate, init_func=init,
		                               frames=200, interval=20, blit=True)

		plt.draw()
		plt.show()


# fig, ax = plt.subplots(figsize=(5, 3))
# ax.set(xlim=(-3, 3), ylim=(-1, 1))

# x = np.linspace(-3, 3, 91)
# print(x.shape)
# t = np.linspace(1, 25, 30)
# print(t.shape)

# X2, T2 = np.meshgrid(x, t)
 
# sinT2 = np.sin(2*np.pi*T2/T2.max())
# F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))

# line = ax.plot(x, F[0, :], color='k', lw=2)[0]

# def animate(i):
#     line.set_ydata(F[i, :])


# anim = FuncAnimation(
#     fig, animate, interval=100, frames=len(t)-1)
 
# plt.draw()
# plt.show()

# anim.save('filename.mp4')


# fig = plt.figure()
# ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
# line, = ax.plot([], [], lw=3)

# def init():
#     line.set_data([], [])
#     return line,
# def animate(i):
#     x = np.linspace(0, 4, 1000)
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
#     return line,

# anim = FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)

# plt.draw()
# plt.show()


if __name__ == '__main__':
  main(sys.argv[1:])

