# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt 

import params as yamnet_params
import yamnet as yamnet_model


def main(argv):
  assert argv, 'Usage: inference.py <wav file> <wav file> ...'

  params = yamnet_params.Params()
  yamnet = yamnet_model.yamnet_frames_model(params)
  yamnet.load_weights('yamnet.h5')
  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

  for file_name in argv:
    # Decode the WAV file.
    wav_data, sr = sf.read(file_name, dtype=np.int16)
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

    # plt.figure(figsize=(20, 8))
    # plt.plot(waveform)
    # plt.xlabel('Samples')
    # plt.ylabel('Amplitude')
    # # plt.savefig('waveform.png')
    # plt.show()
    # plt.close()


    print('waveform sample dtaa',waveform.shape)
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

    truth_labels=[yamnet_classes[i] for i in top5_i]
    print('ground labels',truth_labels)
    total_time=0

    # plt.figure(figsize=(20, 8))
    # plt.plot(scores[:,282].numpy(),label='water')
    # plt.plot(scores[:,364].numpy(),label='faucet')
    # plt.plot(scores[:,365].numpy(),label='sink')
    # plt.legend()
    # plt.show()
    # plt.close()



    for i in range(len(scores)):
      pred=scores[i]

      water_prob=pred[282].numpy()
      print('water_prob',water_prob)
      top5_i = np.argsort(pred)[::-1][:5]
      print(file_name, ':\n' +
      '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], pred[i])
      for i in top5_i))

      pred_class=yamnet_classes[top5_i[0]]
      print(pred_class)
      if pred_class in truth_labels:
        total_time+=0.96

    print('total time',total_time/2)







if __name__ == '__main__':
  main(sys.argv[1:])
