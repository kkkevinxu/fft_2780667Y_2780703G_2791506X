# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:14:08 2022

@author: DELL
"""
"""
Finite impulse response filter design. 
One functions is used to calculate the convolution of the coeffieients with the 
original signal
"""
class FIRfilter: 
    """
    Class init function.

    :param _coefficients: an array of filter coeffficients 
    """
    def __init__(self, _coefficients):
        # your code here 
        self.co = _coefficients
        self.save = []
        self.taps = 1000
        
    """
    dofilter used to filter the original input one by one .

    :param v: a value of the original signal
    :return result: an array that is already filtered
    """ 
    def dofilter(self, v): 
        result = 0
        # your code here 
        self.save.append(v)
        index = len(self.save)
        if index > self.taps:
            for m in range(0,self.taps):
                result = result + self.co[self.taps-1-m]*self.save[m + index - self.taps]
        return result
