# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 04:33:28 2022

@author: Weibo Gao
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Add an adaptive LMS filter command to FIR filter class
"""

class FIRfilter:
    def __init__(self, _coefficients):
        # your code here
        self.co = _coefficients
        self.save = []
        self.taps = 1000
        self.buffer = np.zeros(self.taps)

    def dofilter(self, v):
        result = 0
        # your code here
        self.save.append(v)
        index = len(self.save)
        if index > self.taps:
            for m in range(0,self.taps):
                result = result + self.co[self.taps-1-m]*self.save[m + index - self.taps]
        return result

    def filter(self, v):
        for j in range(self.taps-1):
            self.buffer[self.taps-j-1] = self.buffer[self.taps-j-2]
        self.buffer[0] = v
        return np.inner(self.buffer, self.co)

    def lms (self, error, mu = 0.01):
        for j in range (0,self.taps):
            self.co[j] = self.co[j] + error * mu * self.buffer[j]

    def doFilterAdaptive(self, signal, noise, learningRate):
        fs = 1000
        ntaps = 1000
        ecg = signal
        fnoise = noise
        f = FIRfilter(np.zeros(ntaps))
        y = np.empty(len(ecg))
        for i in range(len(ecg)):
            ref_noise = np.sin(2.0 * np.pi * fnoise/fs * i);
            canceller = f.filter(ref_noise)
            output_signal = ecg[i] - canceller
            f.lms(output_signal, learningRate)
            y[i] = output_signal
        return y

        result = 1
        return result


def fftshift(fft,n):
    result = np.zeros(n)
    result[0:int(n/2)] = fft[int(n/2):n]
    result[int(n/2):n] = fft[0:int(n/2)]
    return result

class filtercoefficients:

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


def main():
    data = np.loadtxt('ecg.dat')
    fs = 1000
    ts = 1/fs
    time = np.linspace(0,len(data)/fs,len(data))
    plt.plot(time[0:1500],data[0:1500])
    plt.xlabel('time(s)')
    plt.show()
    plt.ylabel('amplitude')

    data_fft = np.fft.fft(data)
    fre = np.linspace(0,fs,len(data_fft))
    shiftfft = fftshift(20*np.log10(np.abs(data_fft)),len(data_fft))
    plt.plot(fre,shiftfft)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()

    #highpass filter
    h1 = fftshift(filtercoefficients.highpassDesign(fs,1),fs) * np.hamming(fs)
    fir1 = FIRfilter(h1)
    y1 = np.zeros(len(data))
    for i in range(0,len(data)):
        y1[i] = fir1.dofilter(data[i])
    fft_y1 = np.fft.fft(y1)
    shifty1 = fftshift(20*np.log10(np.abs(fft_y1)),len(fft_y1))
    plt.plot(fre,shifty1)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()

    #bandstop filter
    h2 = fftshift(filtercoefficients.bandstopDesign(fs,40,60),fs) * np.hamming(fs)
    plt.plot(np.linspace(0,fs,1000),h2)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()

    fir2 = FIRfilter(h2)
    y2 = np.zeros(len(y1))
    for i in range(0,len(y1)):
        y2[i] = fir2.dofilter(y1[i])
    fft_y2 = np.fft.fft(y2)
    shifty2 = fftshift(20*np.log10(np.abs(fft_y2)),len(fft_y2))
    plt.plot(fre,shifty2)
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
    plt.show()

    plt.plot(time[0:1500],y1[0:1500])
    plt.title('highpass filter signal')
    plt.xlabel('time(s)')
    plt.ylabel('amplitude')
    plt.show()

    plt.plot(time[1000:7000],y2[1000:7000])
    plt.title('bandstop filter signal')
    plt.xlabel('time(s)')
    plt.ylabel('amplitude')
    plt.show()


    # signal =y2, noise = 50Hz learningRate = 0.001
    y3 = fir1.doFilterAdaptive(y2, 50, 0.001)
    plt.plot(time[1000:7000],y3[1000:7000])
    plt.title('LMS filter signal')
    plt.xlabel('time(s)')
    plt.ylabel('amplitude')
    plt.show()

    plt.plot(time[1200:2000],y3[1200:2000])
    plt.title('learning process')
    plt.xlabel('time(s)')
    plt.ylabel('amplitude')
    plt.show()

    plt.plot(time[1200:2000],y2[1200:2000])
    plt.title('compared signal')
    plt.xlabel('time(s)')
    plt.ylabel('amplitude')
    plt.show()




if __name__ == "__main__":
    main()