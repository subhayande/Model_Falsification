#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Please cite: 
1. De, Subhayan, et al. "Investigation of model falsification using error and likelihood bounds with application to a structural system." 
Journal of Engineering Mechanics 144.9 (2018): 04018078. 
https://doi.org/10.1061/(ASCE)EM.1943-7889.0001440  
2. De, Subhayan, et al. "A hybrid probabilistic framework for model validation with application to structural dynamics modeling." 
Mechanical Systems and Signal Processing 121 (2019): 961-980. 
https://doi.org/10.1016/j.ymssp.2018.10.014

Created on Fri Jul 20 17:49:43 2018

@author: Subhayan De, Ph.D. (email: Subhayan.De@colorado.edu) 
Website: www.subhayande.com
"""

import numpy as np

def main():
    """
    An example of how to implement Model Falsification 
    using the python module 'Falsification'
    """
    from Falsification import EBBonferroni
    from Falsification import EBSidak
    from Falsification import EBBH
    from Falsification import LBBonferroni
    from Falsification import LBSidak
    from Falsification import LBBH

    # select algorithm
    # available options: 
    #                   EBB = Error-bound Bonferroni
    #                   EBS = Error-bound Sidak
    #                   EBBH = Error-bound BH
    #                   LBB = Likelihood-bound Bonferroni
    #                   LBS = Likelihood-bound Sidak
    #                   LBBH = Likelihood-bound BH
    alg = 'LBBH'

    np.random.seed(0)
    nMeas = 200
    # parameters
    w1 = 3.0
    w2 = 4.5
    # noisy data
    X = 2.0*np.random.rand(nMeas,1)
    np.savetxt('fals1_data.txt',X)
    y = w1 + w2 * X + np.random.randn(nMeas,1)
    
    # Model definition
    thetaDist = np.array(['uniform','normal'])
    modelDistVar = np.array([[2,5],[4.0,0.5]])
    
    # Error-bound algorithms
    if alg == 'EBB':
        print(EBBonferroni.__doc__)
        fls = EBBonferroni(nModels =1000, nMeas=nMeas, alpha=0.05, paramDist=thetaDist, distVar=modelDistVar, meas = y, func = fun, residualSD = 1.0)
        fls.doFalsification()
    elif alg == 'EBS':
        print(EBSidak.__doc__)
        fls = EBSidak(nModels =1000, nMeas=nMeas, alpha=0.05, paramDist=thetaDist, distVar=modelDistVar, meas = y, func = fun, residualSD = 1.0)
        fls.doFalsification()
    elif alg == 'EBBH':
        print(EBBH.__doc__)
        fls = EBBH(nModels =1000, nMeas=nMeas, alpha=0.05, paramDist=thetaDist, distVar=modelDistVar, meas = y, func = fun, residualSD = 1.0)
        fls.doFalsification()
    # Likelihood-bound algorithms
    elif alg == 'LBB':    
        print(LBBonferroni.__doc__)
        fls = LBBonferroni(nModels =1000, nMeas=nMeas, alpha=0.05, paramDist=thetaDist, distVar=modelDistVar, meas = y, func = fun, residualSD = 0.4)
        fls.doFalsification()
    elif alg == 'LBS':
        print(LBSidak.__doc__)
        fls = LBSidak(nModels =1000, nMeas=nMeas, alpha=0.05, paramDist=thetaDist, distVar=modelDistVar, meas = y, func = fun, residualSD = 0.4)
        fls.doFalsification()
    elif alg == 'LBBH':
        print(LBBH.__doc__)
        fls = LBBH(nModels =1000, nMeas=nMeas, alpha=0.05, paramDist=thetaDist, distVar=modelDistVar, meas = y, func = fun, residualSD = 0.5)
        fls.doFalsification()
    
    return fls

def fun(param):
    """
    function to calculate model predictions
    """
    X = np.loadtxt('fals1_data.txt')
    pred = param[0] + param[1]*X
    return pred

# Next two lines run the main function
if __name__ == "__main__":
    fls = main()
