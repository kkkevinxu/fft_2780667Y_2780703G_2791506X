# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:08:11 2022

@author: Peifeng Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal

# used the wave file as string
file_a = '../ENG5027/a.wav'
file_e = '../ENG5027/e.wav'
file_i = '../ENG5027/i.wav'
file_o = '../ENG5027/o.wav'


# get the normalized signal
def normalize(origin):
    new = origin - np.mean(origin) # Eliminate DC component
    new = new / np.max(np.abs(new)) # Amplitude normalization
    return new

# get the maximum value of a specific array
def get_maximum(array, begin, end):
    maximum = -100
    for i in range(begin, end):
        if maximum < array[i]:
            maximum = array[i]
            order = i
    return maximum, order

# get the accurate position of the next peak of the signal
def find_next_peak(array, begin, end, gap):
    for i in range(begin, end, gap):
        maximum, result = get_maximum(array,i,i+gap)
        if maximum != array[i]:
            return maximum, result
            break

# function to identify vowels
def voweldetector(wavefile):
    #<code here>

    #define a maximum value for peak find
    maximum = -100

    #read the wave file.
    Fs, samples = wavfile.read(wavefile)
    sample_point = len(samples)

    # the Nyquist rate of the signal.
    nyq_rate = sample_point / 2.0
    width = 200.0/nyq_rate

    # the desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # compute the order and Kaiser parameter for the FIR filter.
    N, beta = scipy.signal.kaiserord(ripple_db, width)

    # the cutoff frequency of the filter.
    cutoff_hz = 500.0

    # cut off some of the signal
    cut_begin = 0
    cut_end = 20000

    # used in finding peak, should not be too big
    gap = 100

    # plot the original signal
    T = sample_point/Fs
    time = np.linspace(0,T,sample_point)
    plt.subplot(3,2,1)
    plt.plot(time, normalize(samples))
    plt.xlabel('time(ms)');
    plt.ylabel('Orignal Audio')
    plt.grid(1)
    plt.title('Original Audio')

    # normalized the sample
    Sample_fft = np.fft.fft(normalize(samples))
    Fre = np.linspace(cut_begin ,cut_end, cut_end-cut_begin)


    # plot the fast Fourier Transform of Sample
    plt.subplot(3,2,2)
    plt.plot(Fre,20*np.log10(np.abs(Sample_fft[cut_begin :cut_end])))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Orignal Audio(dB)')
    plt.grid(1)

    # get the ifft(by cut the radio into a half)
    Inverse = np.fft.ifft(Sample_fft[cut_begin :cut_end])
    Inverse_FFT = np.fft.fft(Inverse)
    FFT = np.fft.fft(Inverse_FFT)

    # get the fft of the fft of inverse
    plt.subplot(3,2,3)
    plt.plot(Fre,20*np.log10(np.abs(FFT)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Fft of fft(dB)')
    plt.grid(1)


    # low pass filter of fft
    # use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = scipy.signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    # use lfilter to filter the inverse of fft with the FIR filter.
    filtered = scipy.signal.lfilter(taps, 2.0, Inverse_FFT)
    # get the result array
    Result = 20*np.log10(np.abs(filtered))

    # plot  the signals
    plt.subplot(3,2,4)
    plt.plot(Fre,Result)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Vowel fft after filter(dB)')
    plt.grid(1)

    # find the frequency of the peaks of harmonics
    maximum, first = get_maximum(Result,0, cut_end-cut_begin)
    maximum2, second = find_next_peak(Result, first, cut_end-cut_begin, gap)
    maximum3, third = find_next_peak(Result, second, cut_end-cut_begin, gap)

    # judge what vowel should it be
    if first > 700:
        if second > 1000:
            return 'The vowel is a'
        else:
            return 'The vowel is e'
    else:
        if second < 1000:
            return 'The vowel is i'
        else:
            return 'The vowel is o'




def main():
    print(voweldetector(file_a))
    print(voweldetector(file_e))




if __name__ == "__main__":
    main()