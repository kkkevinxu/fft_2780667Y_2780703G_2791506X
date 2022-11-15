# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:18:51 2022

@author: Chengjia Yu
"""

import numpy as np
import matplotlib.pyplot as plt
from firfilter import FIRfilter


"""
Finite impulse response filter coefficients calculation.
Two functions are used to calculate the coefficients of high-pass filter
 and band-stop filter
"""
class filtercoefficients:
    """
    highpassDesign used to calculate high-pass filter coefficients.
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
"""
def fftshift(fft):
    n = len(fft)
    result = np.zeros(n)
    result[0:int(n/2)] = fft[int(n/2):n]
    result[int(n/2):n] = fft[0:int(n/2)]
    return result

"""
Filter the ECG with the time reversed template
"""
def matched_filter(template,y):

    fir_coeffs = template[::-1]  #time reversing template
    fir = FIRfilter(fir_coeffs)
    det = np.zeros(len(y))
    for i in range(0,len(y)):
        det[i] = fir.dofilter(y[i])

    return det

"""
Detect r peaks with the matched filter
"""
def R_Detect(det,fs):
    distance_check = int(fs*0.3)  #set the minimum distance
    det_peaks = []
    peaks = [0]
    for i in range(len(det)):
        if i>0 and i<len(det)-1:
            if det[i-1]<det[i] and det[i+1]<det[i] and det[i]>40 : #find the R peaks
                peaks.append(i)
                if i-peaks[-2]>distance_check:  #removing bogus detections
                    det_peaks.append(i)

    return det_peaks



def main():
    # Load data file
    data = np.loadtxt('ecg.dat')
    fs = 1000

    # Get the filter coefficient of the baseline high-pass filter
    h1 = fftshift(filtercoefficients.highpassDesign(fs,1)) * np.hamming(fs)
    fir1 = FIRfilter(h1)
    y1 = np.zeros(len(data))
    # Input the original signal one by one
    for i in range(0,len(data)):
        y1[i] = fir1.dofilter(data[i])

    # Get the filter coefficient of the 50Hz band-stop filter
    h2 = fftshift(filtercoefficients.bandstopDesign(fs,40,60)) * np.hamming(fs)
    fir2 = FIRfilter(h2)
    y2 = np.zeros(len(y1))
    # Input the original signal one by one
    n=0
    for i in range(0,len(y1)):
        y2[i] = fir2.dofilter(y1[i])
        # Output the real-time heartbeat which shows the R peaks
        if(i==n*1000+2250):
            template = y2[1250:2250]
            fir_coeff = template[::-1]
            det = matched_filter(fir_coeff,y2)
            det = det * det
            plt.plot(det)
            plt.xlabel('time(s)')
            plt.ylabel('amplitude')
            plt.show()
            # Detect R peaks and point out the locations
            det_peaks = R_Detect(det,fs)
            print(det_peaks)
            n=n+1


if __name__ == "__main__":
    main()