# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:48:50 2022

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal

sample_point = 472281;
combined = np.zeros(sample_point)
new = np.zeros(sample_point//2)

def cut_half(origin):
    for m in range(0,sample_point//2):
        new[m] = origin[m]
    return new
    
def normalize(origin):
    new = origin - np.mean(origin) # Eliminate DC component 
    new = new / np.max(np.abs(new)) # Amplitude normalization 
    return new

def main():
    Fs, samples = wavfile.read('/Users/DELL/Desktop/DSP/original.wav')
    #Fs, samples = wavfile.read('/Users/96335/Desktop/original.wav')
    for i in range(0,sample_point):
        combined[i] = (samples[i, 0] + samples[i, 1])/2
        
    Ts = 1/Fs
    T = sample_point/47100
    time = np.linspace(0,T,sample_point)
    plt.subplot(3,2,1)
    normalized = normalize(combined)
    plt.plot(time, normalized)
    plt.xlabel('time(ms)');
    plt.ylabel('Orignal Audio')
    plt.grid(1)
    plt.title('Original Audio')
    
    Origin_fft = np.fft.fft(normalized)
    Fre = np.linspace(0, 1/2*Fs,sample_point//2)
    Origin_fft_cut = cut_half(Origin_fft)
    
    plt.subplot(3,2,2)
    plt.plot(Fre,20*np.log10(np.abs(Origin_fft_cut)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Orignal Audio(dB)')
    plt.grid(1)
    plt.title('Original Audio')
    
    sos_highpass = scipy.signal.butter(4, Wn=150, fs = Fs, btype="highpass",analog = False, output='sos')
    Highpass_result = scipy.signal.sosfilt(sos_highpass, normalized)
    Highpass_fft = np.fft.fft(Highpass_result)
    Highpass_fft_cut = cut_half(Highpass_fft)
    plt.subplot(3,2,3)
    plt.plot(Fre, 20*np.log10(np.abs(Highpass_fft_cut)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    
    sos_bandstop1 = scipy.signal.butter(4, Wn = [325, 350], fs = Fs, btype = "bandstop",analog = False, output='sos')
    Bandstop1_result = scipy.signal.sosfilt(sos_bandstop1, Highpass_result)
    Bandstop1_fft = np.fft.fft(Bandstop1_result)
    Bandstop1_fft_cut = cut_half(Bandstop1_fft)
    plt.subplot(3,2,4)
    plt.plot(Fre, 20*np.log10(np.abs(Bandstop1_fft_cut)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    
    sos_bandstop2 = scipy.signal.butter(4, Wn = [5000, 7000], fs = Fs,btype = "bandstop",analog = False, output='sos')
    Bandstop2_result = scipy.signal.sosfilt(sos_bandstop2, Bandstop1_result)
    Bandstop2_fft = np.fft.fft(Bandstop2_result)
    Bandstop2_fft_cut = cut_half(Bandstop2_fft)
    plt.subplot(3,2,5)
    plt.plot(Fre, 20*np.log10(np.abs(Bandstop2_fft_cut)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    wavfile.write('improved.wav',Fs,Bandstop2_result.astype(np.int16))
    
    
    
if __name__ == "__main__":
    main()