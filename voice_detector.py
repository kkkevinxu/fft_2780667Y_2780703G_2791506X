# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:08:11 2022

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal

file_a = 'C:/Users/DELL/Desktop/DSP/a.wav'
file_e = 'C:/Users/DELL/Desktop/DSP/e.wav'
file_i = 'C:/Users/DELL/Desktop/DSP/i.wav'
file_o = 'C:/Users/DELL/Desktop/DSP/o.wav'


#Get the normalized signal
def normalize(origin):
    new = origin - np.mean(origin) # Eliminate DC component 
    new = new / np.max(np.abs(new)) # Amplitude normalization 
    return new

#Get the maximum value of a specific array
def get_maximum(array, begin, end):
    maximum = -100;
    for i in range(begin, end):
        if maximum < array[i]:
            maximum = array[i]
            order = i
    return maximum, order

#Get the accurate position of the next peak of the signal
def find_next_peak(array, begin, end, gap):
    for i in range(begin, end, gap):
        maximum, result = get_maximum(array,i,i+gap)
        if maximum != array[i]:
            return maximum, result
            break

#vowel detector
def voweldetector(wavefile):
    maximum = -100
    first = 0
    
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
    
    cut_begin = 0
    cut_end = 20000
    
    gap = 100
    
    T = sample_point/Fs
    time = np.linspace(0,T,sample_point)
    plt.subplot(3,2,1)
    plt.plot(time, normalize(samples))
    plt.xlabel('time(ms)');
    plt.ylabel('Orignal Audio')
    plt.grid(1)
    plt.title('Original Audio')
    
    #Normalized the sample
    Sample_fft = np.fft.fft(normalize(samples))
    Fre = np.linspace(cut_begin ,cut_end, cut_end-cut_begin)
    
    
    #Plot the fast Fourier Transform of Sample
    plt.subplot(3,2,2)
    plt.plot(Fre,20*np.log10(np.abs(Sample_fft[cut_begin :cut_end])))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Orignal Audio(dB)')
    plt.grid(1)
    
    #Get the ifft(by cut the radio into a half)
    Inverse = np.fft.ifft(Sample_fft[cut_begin :cut_end])
    Inverse_FFT = np.fft.fft(Inverse)
    FFT = np.fft.fft(Inverse_FFT)
    
    #Get the fft of the fft of inverse
    plt.subplot(3,2,3)
    plt.plot(Fre,20*np.log10(np.abs(FFT)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Fft of fft(dB)')
    plt.grid(1)
  
    
    #Low pass filter of fft 
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = scipy.signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    # Use lfilter to filter x with the FIR filter.
    filtered = scipy.signal.lfilter(taps, 2.0, Inverse_FFT)
    #Get the result array
    Result = 20*np.log10(np.abs(filtered))
   
    #Plot  the signals
    plt.subplot(3,2,4)
    plt.plot(Fre,Result)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Vowel fft after filter(dB)')
    plt.grid(1)

    
    
    maximum, first = get_maximum(Result,0, cut_end-cut_begin)
    maximum2, second = find_next_peak(Result, first, cut_end-cut_begin, gap)        
    maximum3, third = find_next_peak(Result, second, cut_end-cut_begin, gap)        
            
    return  first, second, third
    
   
    
    

def main():
    print(voweldetector(file_a))
    print(voweldetector(file_e))
    print(voweldetector(file_i))
    print(voweldetector(file_o))
    
   
    
if __name__ == "__main__":
    main()