# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:11:40 2022

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def highpassDesign(Fs, Fc):
    k = int(Fc)
    x = np.ones(Fs)
    x[0:k+1] = 0
    x[Fs-k: Fs+1] = 0
    result = np.fft.ifft(x)
    result = np.real(result)
    return result

def bandstopDesign(Fs, Fc1, Fc2):
    k1 = int(Fc1)
    k2 = int(Fc2)
    x = np.ones(Fs)
    x[k1:k2+1] = 0
    x[Fs-k2: Fs-k1+1] = 0
    result = np.fft.ifft(x)
    result = np.real(result)
    return result
    
def fftshift(fft,n):
    result = np.zeros(n)
    result[0:int(n/2)] = fft[int(n/2):n]
    result[int(n/2):n] = fft[0:int(n/2)]
    return result

class FIRfilter: 
    def __init__(self, _coefficients):
        # your code here 
        self.co = _coefficients
        self.save = []
        self.h = []
        for i in range(0,len(self.co)):
            for m in range(0,30):
                self.h.append(self.co[i])
    def dofilter(self, v): 
        result = 0
        index = len(self.save)
        # your code here 
        index = 0
        self.save.append(v)
        for m in range(0,index+1):
            result = result + self.h[index-m]*self.save[m]
        return result
    
def main():
    data = np.loadtxt('../fft_2780667Y_2780703G_2791506X/ECG_1000Hz_43.dat')
    fs = 1000
    ts = 1/fs
    time = np.linspace(0,len(data[0:1000])/fs,len(data))
    plt.plot(time,data)
    plt.xlabel('time(s)')
    plt.show()
    plt.ylabel('amplitude')
    
    data_fft = np.fft.fft(data[0:1000])
    fre = np.linspace(0,fs,len(data_fft))
    shiftfft = fftshift(20*np.log10(np.abs(data_fft)),len(data_fft))
    plt.plot(fre,shiftfft)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()
    
    h1 = fftshift(highpassDesign(fs,1),fs) * np.hamming(fs)
    y1 = signal.lfilter(h1,1,data[0:1000])
    #fir1 = FIRfilter(h1)
    #y1 = np.zeros(len(data))
    #for i in range(0,len(data)):
    #    y1[i] = fir1.dofilter(data[i])
    fft_y1 = np.fft.fft(y1)
    shifty1 = fftshift(20*np.log10(np.abs(fft_y1)),len(fft_y1))
    plt.plot(fre,shifty1)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()
    
    h2 = fftshift(bandstopDesign(fs,45,55),fs) * np.hamming(fs)
    y2 = signal.lfilter(h2,1,y1[0:1000])
    #fir2 = FIRfilter(h2)
    #y2 = np.zeros(len(data))
    #for i in range(0,len(data)):
    #    y2[i] = fir2.dofilter(y1[i])
    fft_y2 = np.fft.fft(y2)
    shifty2 = fftshift(20*np.log10(np.abs(fft_y2)),len(fft_y2))
    plt.plot(fre,shifty2)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()
    
    
    

if __name__ == "__main__":
    main()