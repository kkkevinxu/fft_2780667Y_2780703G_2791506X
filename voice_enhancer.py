# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:07:35 2022

@author: 77127
"""

#import
import numpy as np #to use array
import matplotlib.pyplot as plt #to use Matlab to plot
from scipy.io import wavfile
import scipy.signal

'''
plot the audio signal
1.Plot1: normalised amplitudes vs time using a linear axis in the time domain
2.Plot2: amplitude(dB) vs frequency using logarithmic axis in the frequency domain
'''

sample_point = 472281;
cut_point = int(214186.3945578231); #cut_point is (Sample_point/Sample_rate)*20000, 20kHz
combined = np.zeros(sample_point)
new = np.zeros(cut_point)

def normalize(origin):
    new = origin - np.mean(origin) # Eliminate DC component 
    new = new / np.max(np.abs(new)) # Amplitude normalization 
    return new

def cut_half(origin):
    for m in range(0,cut_point):
        new[m] = origin[m]
    return new
   
def main():
    #plot the audio signal
    #1.Plot1: normalised amplitudes vs time using a linear axis in the time domain
    Sample_rate, data = wavfile.read('../ENG5027/original.wav')
    length = data.shape[0] / Sample_rate
    Sample_point = int(Sample_rate*length)
    combined = np.zeros(Sample_point)
    #Combine the two channels into one and take the average
    for i in range(0,Sample_point):
        combined[i] = (data[i, 0] + data[i, 1])/2
    print(f"number of channels = {data.shape[1]}")  
    print(f"Sample_rate =  {Sample_rate}")
    print(f"length = {length}s")
    print(f'Sample_point = {Sample_point}')
    print('--------')
    print(combined)
    print('--------')
    
    normalized = normalize(combined)
    time = np.linspace(0., length, data.shape[0])
    
    #plot Normalized Amplitude vs Time(s)
    plt.plot(time, normalized)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(1)
    plt.title('Original Audio')
    plt.show()

    #2.Plot2: amplitude(dB) vs frequency using logarithmic axis in the frequency domain
    Fs = Sample_rate
    Origin_fft = np.fft.fft(normalized)
    Fre = np.linspace(0, 20000,cut_point)
    Fre_log = np.log10(Fre)
    Origin_fft_cut = cut_half(Origin_fft)
    
    #plot Orignal Audio(dB) vs frequency(Hz)
    plt.plot(Fre,20*np.log10(np.abs(Origin_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Orignal Audio(dB)')
    plt.grid(1)
    plt.title('Original Audio')
    plt.show()
    
    #plot Orignal Audio(dB) vs frequency(Hz)[log scale]
    plt.plot(Fre_log,20*np.log10(np.abs(Origin_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)[log scale]')
    plt.ylabel('amplitude(dB)')
    plt.grid(1)
    plt.title('Original Audio')
    plt.show()
    
    
    #Audio Analysis
    #1.Mark the peaks in the spectrum which correspond to the fundamental frequencies of any spoken vowels present in the sample.
    #The energy of the vowels primarily lies in the range 250 – 2,000 Hz
    
    
    #2.Mark the frequency range which mainly contains the consonants up to the highest frequencies containing them.
    
    #3.Mark the whole speech spectrum containing the vowels, consonants harmonics.


    #Fourier Transform
    
    #highpass filter 150~
    sos_highpass = scipy.signal.butter(4, Wn=150, fs = Fs, btype="highpass",analog = False, output='sos')
    Highpass_result = scipy.signal.sosfilt(sos_highpass, combined)
    Highpass_fft = np.fft.fft(Highpass_result)
    Highpass_fft_cut = cut_half(Highpass_fft)
    
    plt.plot(Fre_log, 20*np.log10(np.abs(Highpass_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    plt.show()

    #bandstop filter 325,350hz,to cut the signal from 325Hz to 350Hz
    sos_bandstop1 = scipy.signal.butter(4, Wn = [325, 350], fs = Fs, btype = "bandstop",analog = False, output='sos')
    Bandstop1_result = scipy.signal.sosfilt(sos_bandstop1, Highpass_result)
    Bandstop1_fft = np.fft.fft(Bandstop1_result)
    Bandstop1_fft_cut = cut_half(Bandstop1_fft)

    plt.plot(Fre_log, 20*np.log10(np.abs(Bandstop1_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    plt.show()
    
    #bandstop filter 5000,7000hz, to cut the signal around 6kHz
    sos_bandstop2 = scipy.signal.butter(4, Wn = [5000, 7000], fs = Fs,btype = "bandstop",analog = False, output='sos')
    Bandstop2_result = scipy.signal.sosfilt(sos_bandstop2, Bandstop1_result)
    Bandstop2_fft = np.fft.fft(Bandstop2_result)
    Bandstop2_fft_cut = cut_half(Bandstop2_fft)
 
    plt.plot(Fre_log, 20*np.log10(np.abs(Bandstop2_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    plt.show()
    wavfile.write('improved.wav',Fs,Bandstop2_result.astype(np.int16))
    '''  
    #Bandpass filter, boosts in the 5500hz to 6500hz range
    sos_bandpass1 = scipy.signal.butter(4, Wn = [5500,6500 ], fs = Fs,btype = "bandpass",analog = False, output='sos')
    Bandpass1_result = scipy.signal.sosfilt(sos_bandpass1, Bandstop2_result)
    Bandpass1_fft = np.fft.fft(Bandpass1_result)
    Bandpass1_fft_cut = cut_half(Bandpass1_fft)

    plt.plot(Fre_log, 20*np.log10(np.abs(Bandpass1_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    plt.show()  
    
    #Bandpass filter, narrow boosts in the 200hz to 600hz range
    sos_bandpass2 = scipy.signal.butter(4, Wn = [200, 600], fs = Fs,btype = "bandpass",analog = False, output='sos')
    Bandpass2_result = scipy.signal.sosfilt(sos_bandpass2, Bandstop2_result)
    Bandpass2_fft = np.fft.fft(Bandpass2_result)
    Bandpass2_fft_cut = cut_half(Bandpass2_fft)
 
    plt.plot(Fre_log, 20*np.log10(np.abs(Bandpass2_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    plt.show()
    ''' 
    
    #if the the region of the highest harmonic voice frequencies in the spectrum is 8k-10kHz
    
    #bulid a empty array to store the increasing data
    Result_fft = np.zeros(Sample_point)
    #increase the amplitudes of the highest harmonic voice frequencies
    for i in range(0,Sample_point):
        if  i>=8000 and i<=10000:
            Result_fft[i] = 1.1*Bandstop2_fft[i]
        else:
            Result_fft[i] = Bandstop2_fft[i]
    
     
    #rebulid voice    
    Result_wav = np.fft.ifft(Result_fft)
    
    #plot the result diagram
    Result_fft_cut = cut_half(Result_fft)   
    plt.plot(Fre_log, 20*np.log10(np.abs(Result_fft_cut/Sample_point)))
    plt.xlabel('frequency(Hz)')
    plt.ylabel('Result Audio(dB)')
    plt.grid(1)
    plt.title('Result Audio')
    plt.show()
   
    
    
    
    #output wav
    wavfile.write('improved.wav',Fs,Result_wav.astype(np.int16))

if __name__ == "__main__":
    main()
