# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:08:11 2022

@author: Peifeng Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal

# Using the relative path of wavefiles as strings
file_a = '../fft_2780667Y_2780703G_2791506X/a.wav'
file_e = '../fft_2780667Y_2780703G_2791506X/e.wav'
file_i = '../fft_2780667Y_2780703G_2791506X/i.wav'
file_o = '../fft_2780667Y_2780703G_2791506X/o.wav'

# Function to get the normalized signal
def normalize(origin):
    """
    normalize gets the normalized signal from the original signal.

    :param original: an array of the original signal
    :return new: a new array that is the normalized array of original
    """
    new = origin - np.mean(origin) # Eliminate DC component
    new = new / np.max(np.abs(new)) # Amplitude normalization
    return new

# Function to get the maximum value of a specific array
def get_maximum(array, begin, end):
    """
    get_maximum gets the maixmum value from a given interval of the given array.

    :param array: given array which is finding the maximum value
    :param begin: begin point of the interval, should be a number
    :param end: end point of the interval, should be a number
    :return maixmum: the maximum value, should be a number
    :return order: the index that the maximum appears in the array, should be a number
    """
    maximum = -100
    for i in range(begin, end):
        if maximum < array[i]:
            maximum = array[i]
            order = i
    return maximum, order

# Function to get the accurate position of the next peak of the signal
def find_next_peak(array, begin, end, gap):
    """
    find_next_peak gets the accurate position of the next peak of the signal by
                divided a long interval into small intervals and identify if they
                are rising or falling.

    :param array: an array of the signal
    :param begin: the begin point of the array to find the next peak, should be a number
    :param end: the end point of the array to find the next peak, should be a number
    :param gap: the step array is divided, should be a number
    :return maximum: the maximum value of the peak, should be a number
    :return result: the index that the peak appears in the array, should be a number
    """
    for i in range(begin, end, gap):
        maximum, result = get_maximum(array,i,i+gap)
        if maximum != array[i]:
            return maximum, result
            break

# Function to identify vowels
def voweldetector(wavefile):
    """
    voweldetector gets the accurate position of the first three formant of given
                wavefile and check which vowel they are.

    :param wavefile: the file loaded into the function, should be a string
    :return: a string about the name of the vowel
    """
    #<code here>

    # Define a maximum value for peak find
    maximum = -100

    # Load the wave file.
    Fs, samples = wavfile.read(wavefile)
    sample_point = len(samples)

    # The Nyquist rate of the signal.
    nyq_rate = sample_point / 2.0
    width = 200.0/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = scipy.signal.kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 500.0

    # The points to cut the signal
    cut_begin = 0
    cut_end = 20000

    # Used in finding peak, should not be too large
    gap = 100

    # Plot the original signal
    T = sample_point/Fs
    time = np.linspace(0,T,sample_point)
    plt.subplot(3,2,1)
    plt.plot(time, normalize(samples))
    plt.xlabel('time(ms)');
    plt.ylabel('Orignal Audio')
    plt.grid(1)
    plt.title('Original Audio')

    # Normalized the sample
    Sample_fft = np.fft.fft(normalize(samples))
    Fre = np.linspace(cut_begin ,cut_end, cut_end-cut_begin)


    # Plot the fast Fourier Transform of Sample
    plt.subplot(3,2,2)
    plt.plot(Fre,20*np.log10(np.abs(Sample_fft[cut_begin :cut_end])))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Orignal Audio(dB)')
    plt.grid(1)

    # Get the ifft after cut the radio into a half
    Inverse = np.fft.ifft(Sample_fft[cut_begin :cut_end])
    Inverse_FFT = np.fft.fft(Inverse)
    FFT = np.fft.fft(Inverse_FFT)

    # Get the fft of the fft of inverse
    plt.subplot(3,2,3)
    plt.plot(Fre,20*np.log10(np.abs(FFT)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Fft of fft(dB)')
    plt.grid(1)


    # low pass filter of fft
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = scipy.signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    # Use lfilter to filter the inverse of fft with the FIR filter.
    filtered = scipy.signal.lfilter(taps, 2.0, Inverse_FFT)
    # Get the result array
    Result = 20*np.log10(np.abs(filtered))

    # Plot  the signals
    plt.subplot(3,2,4)
    plt.plot(Fre,Result)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Vowel fft after filter(dB)')
    plt.grid(1)

    # Find the frequency of the peaks of harmonics
    maximum, first = get_maximum(Result,0, cut_end-cut_begin)
    maximum2, second = find_next_peak(Result, first, cut_end-cut_begin, gap)
    maximum3, third = find_next_peak(Result, second, cut_end-cut_begin, gap)

    # Judge what vowel should it be
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