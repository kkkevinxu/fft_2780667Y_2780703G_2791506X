# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:11:40 2022

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from firfilter import FIRfilter

"""
Finite impulse response filter coefficients calculation. 
Two functions are used to calculate the coefficients of high-pass filter
 and band-stop filter
"""
class filtercoefficients:
    """
    highpassDesign used to calculate high-pass filter coefficients.

    :param Fs: an integer of the sample frenquency
    :param Fc: an integer of the cut-off frequency
    :return result: an array that is the coefficients of the filter
    """
    def highpassDesign(Fs, Fc):
        k = int(Fc)
        x = np.ones(Fs)
        x[0:k+1] = 0
        x[Fs-k: Fs+1] = 0
        result = np.fft.ifft(x)
        result = np.real(result)
        return result
    
    """
    bandstopDesign used to calculate band-stop filter coefficients.

    :param Fs: an integer of the sample frenquency
    :param Fc1: an integer of the cut-off frequency of the low frequency
    :param Fc2: an integer of the cut-off frequency of the high frequency
    :return result: an array that is the coefficients of the filter
    """
    def bandstopDesign(Fs, Fc1, Fc2):
        k1 = int(Fc1)
        k2 = int(Fc2)
        x = np.ones(Fs)
        x[k1:k2+1] = 0
        x[Fs-k2: Fs-k1+1] = 0
        result = np.fft.ifft(x)
        result = np.real(result)
        return result
    
"""
fftshift used to shift the positive time and negative time.

:param fft: an array of the sample that need to shift, should be in frequency domain
:return result: an array that is already shifted by the function
"""    
def fftshift(fft):
    n = len(fft)
    result = np.zeros(n)
    result[0:int(n/2)] = fft[int(n/2):n]
    result[int(n/2):n] = fft[0:int(n/2)]
    return result

    
def main():
    # Load data file
    data = np.loadtxt('../fft_2780667Y_2780703G_2791506X/ECG_1000Hz_43.dat')
    fs = 1000
    ts = 1/fs
    
    # Plot a part of the original signal
    time = np.linspace(0,len(data)/fs,len(data))
    plt.plot(time[0:1500],data[0:1500])
    plt.xlabel('time(s)')
    plt.show()
    plt.ylabel('amplitude')
    
    # Get the fft of the original signal and plot it
    data_fft = np.fft.fft(data)
    fre = np.linspace(0,fs,len(data_fft))
    shiftfft = fftshift(20*np.log10(np.abs(data_fft)))
    plt.plot(fre,shiftfft)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()
    
    # Get the filter coefficient of the baseline high-pass filter
    h1 = fftshift(filtercoefficients.highpassDesign(fs,1)) * np.hamming(fs)
    fir1 = FIRfilter(h1)
    y1 = np.zeros(len(data))
    # Input the original signal one by one
    for i in range(0,len(data)):
        y1[i] = fir1.dofilter(data[i])
    fft_y1 = np.fft.fft(y1)
    shifty1 = fftshift(20*np.log10(np.abs(fft_y1)))
    plt.plot(fre,shifty1)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()
    
    # Get the filter coefficient of the 50Hz band-stop filter
    h2 = fftshift(filtercoefficients.bandstopDesign(fs,40,60)) * np.hamming(fs)
    plt.plot(np.linspace(0,fs,1000),h2)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()
    fir2 = FIRfilter(h2)
    y2 = np.zeros(len(y1))
    # Input the original signal one by one
    for i in range(0,len(y1)):
        y2[i] = fir2.dofilter(y1[i])
    fft_y2 = np.fft.fft(y2)
    shifty2 = fftshift(20*np.log10(np.abs(fft_y2)))
    plt.plot(fre,shifty2)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()
    
    # Plot a part of the signal after high-pass filter
    plt.plot(time[0:1500],y1[0:1500])
    plt.xlabel('time(s)')
    plt.ylabel('amplitude')
    plt.show()
    
    # Plot a part of the signal after band-stop filter
    plt.plot(time[1000:3000],y2[1000:3000])
    plt.xlabel('time(s)')
    plt.ylabel('amplitude')
    plt.show()
        
    
    
    

if __name__ == "__main__":
    main()