# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 04:33:28 2022

@author: Weibo Gao
"""
import numpy as np

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