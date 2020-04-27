# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 00:53:17 2017

@author: luka
"""

import pysptk
import scipy.io.wavfile
import os
import numpy as np
import librosa

frame_length=320
hop_length=160
for filename in os.listdir('Voices'):
    f=open('LPCCoeff/'+filename+'.lpc', 'w')
    output=np.empty((0, 13))
    for i in os.listdir('Voices/'+filename):
        FSample, samples=scipy.io.wavfile.read('Voices/'+filename+'/'+i)
        #samplesFloat=samples.astype(float)
        if len(samples)>=320:
            frames=librosa.util.frame(samples, frame_length, hop_length).astype(np.float64).T
            frames*=pysptk.blackman(frame_length)
            assert frames.shape[1]==frame_length
            try:
                lpcCoeff=pysptk.sptk.lpc(frames[0], 12)
                lpcCoeff=pysptk.sptk.lpc2c(lpcCoeff, 12)
                output=np.append(output, np.array([lpcCoeff]), axis=0)
            except Exception:
                pass
    np.savetxt(f, output, fmt='%.7f')
    del output
    f.close()
