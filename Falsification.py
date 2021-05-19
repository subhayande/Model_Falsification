#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the class file for Error-bound and Likelihood-bound model falsification
Error criteria used:
    (i) Familywise error rate (FWER)
        (a) Bonferroni
        (b) Sidak
    (ii) False discovery rate (FDR)
        (a) BH procedure

NOTE: Currently, only Gaussian distributions for residual errors are allowed.
Two-sided hypothesis tests are implemented.

Required packages: numpy, scipy, abc, time

Please cite: 
1. De, Subhayan, et al. "Investigation of model falsification using error and likelihood bounds with application to a structural system." 
Journal of Engineering Mechanics 144.9 (2018): 04018078. 
https://doi.org/10.1061/(ASCE)EM.1943-7889.0001440  

2. De, Subhayan, et al. "A hybrid probabilistic framework for model validation with application to structural dynamics modeling." 
Mechanical Systems and Signal Processing 121 (2019): 961-980. 
https://doi.org/10.1016/j.ymssp.2018.10.014

Created on Fri Jul 20 16:17:33 2018

License: MIT

Copyright 2018 Subhayan De

Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in 
 the Software without restriction, including without limitation the rights to 
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

@author: Subhayan De, Ph.D. (email: Subhayan.De@colorado.edu) 
Website: www.subhayande.com 
"""

import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import time

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
        Call in a loop to create terminal progress bar
        parameters:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

class Fals(ABC):        
    """
    ===========================================================================
    |                        Falsification class                              |
    |                         (Abstract class)                                |
    ===========================================================================
    This class can not be initialized from outside
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        genModels:          generates all the model instances
        funEval:            predicts
        getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def genModels(self):
        """
        generates models
        """
        self.models = np.zeros((self.nModels,self.nParam))
        # print progress bar
        print('\n --------------------------- \n STEP I: Generating candidate models \n')
        printProgressBar(0, self.nParam-1, prefix = self.method, suffix = 'Complete', length = 25)
        for i in range(self.nParam):
            # print progress bar
            printProgressBar(i, self.nParam-1, prefix = self.method, suffix = ('Complete: Time Elapsed = '+str(np.around(time.clock()-self.t,decimals=2))+'s    '), length = 25)
            if self.paramDist[i] == 'uniform':
                self.models[:,i] = np.random.uniform(low=self.distVar[i,0], high=self.distVar[i,1],size=self.nModels)
            elif self.paramDist[i] == 'gaussian' or self.paramDist[i] == 'normal':
                self.models[:,i] = self.distVar[i,0]+self.distVar[i,1]*np.random.standard_normal(size=self.nModels)
            elif self.paramDist[i] == 'lognormal':
                self.models[:,i] = np.random.lognormal(self.distVar[i,0],self.distVar[i,1],self.nModels)
                
    def funEval(self):
        """
        function evaluation
        """
        self.predictions = np.zeros((self.nModels,self.nMeas))
        # print progress bar
        print('\n --------------------------- \n STEP II: Predicting using candidate models \n')
        printProgressBar(0, self.nModels-1, prefix = self.method, suffix = 'Complete', length = 25)
        for i in range(self.nModels):
            self.currentModel = i
            self.predictions[i,:] = self.func(self.models[i,:])
            # print progress bar
            Fals.printProgress(self)
    
    def getAlphapValues(self):
        """
        p and alpha value calculation
        """
        #self.funEval()
        self.pValues = np.zeros((self.nModels,self.nMeas))
        self.alphaValues = np.zeros((self.nModels,self.nMeas))
        # print progress bar
        print('\n ---------------------------\n STEP III: Calculating alpha and p values \n')
        printProgressBar(0, self.nModels-1, prefix = self.method, suffix = 'Complete', length = 25)
        for i in range(self.nModels):
            # print progress bar
            self.currentModel = i
            Fals.printProgress(self)
            
            for j in range(self.nMeas):
                err = np.absolute(self.predictions[i,j] - self.meas[j])
                self.pValues[i,j] = (stats.norm.cdf(err,0.0,self.residualSD)) - (stats.norm.cdf(-err,0.0,self.residualSD))
                self.alphaValues[i,j] = 2.0*(1.0 - stats.norm.cdf(err,0.0,self.residualSD))
                
    def printProgress(self):
        # Update Progress Bar
        printProgressBar(self.currentModel, self.nModels-1, prefix = self.method, suffix = ('Complete: Time Elapsed = '+str(np.around(time.clock()-self.t,decimals=2))+'s    '), length = 25)

       
class EBFals(Fals):
    """
    ===========================================================================
    |                    Error-bound Falsification class                      |
    |                  (derived from Falsification class)                     |
    ===========================================================================
    This class should not be initialized from outside
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initializes the class
        """
        self.t = time.clock()
        allowed_kwargs = {'nModels', 'nMeas', 'alpha' , 'paramDist', 'distVar', 'meas', 'residualSD', 'func'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed to optimizer at: ' + str(k))
        self.__dict__.update(kwargs)
        self.nParam = np.size(self.paramDist)
        print('\nTotal number of candidate models = ',self.nModels,'\n')
        Fals.genModels(self)
        Fals.funEval(self)
        Fals.getAlphapValues(self)
        self.version = '0.0.0'
        
    @abstractmethod
    def doFalsification(self):
        """
        performs falsification
        """
        # initialize progress bar
        print('\n --------------------------- \n STEP IV: Falsifying models \n')
        printProgressBar(0, self.nModels-1, prefix = self.method, suffix = 'Complete', length = 25)
        if hasattr(self,'alphan'):
            if np.size(self.alphan)==1:
                alphn = self.alphan*np.ones(self.nMeas)
            else:
                alphn = self.alphan
        else:
            raise NameError('No alphan values\n')    
        
        self.unfalsModels = np.array([[],[]])
        for i in range(self.nModels):
            self.currentModel = i
            # print progress bar
            Fals.printProgress(self)
            falsFlag = True
            for j in range(self.nMeas):
                if self.alphaValues[i,j]<alphn[j]:
                    falsFlag = False
            if falsFlag:
                self.unfalsModels = np.append(self.unfalsModels,self.models[i,:])
                    
        self.unfalsModels = np.reshape(self.unfalsModels,(-1,2))
        self.percentUnfals = np.size(self.unfalsModels,0)/self.nModels*100.0
        print('\nNumber of unfalsified models = ', np.size(self.unfalsModels,0),'\n')
        print('\nPercent unfalsified = {0:3.2f}'.format(self.percentUnfals), '%\n')

class EBBonferroni(EBFals):
    """
    ===========================================================================
    |        Error-bound Falsification class using Bonferroni criterion       |
    |              (derived from Error-bound Falsification class)             |
    ===========================================================================
    Initialization:
    Fals = EBBonferroni(nModels, nMeas, alpha, paramDist, distVar, meas, func, 
                        residualSD)
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification 
                                using Bonferroni correction
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initializes the class
        """
        self.method = 'Error-bound Bonferroni'
        print('\nUsing Error-bound model falsification with Bonferroni criterion\n')
        
        EBFals.__init__(self,**kwargs)
        
    def doFalsification(self):
        """
        performs falsification
        """
        
        self.alphan = self.alpha/self.nMeas
        super().doFalsification()
        #EBFals.doFalsification(self)

        
class EBSidak(EBFals):
    """
    ===========================================================================
    |           Error-bound Falsification class using Sidak criterion         |
    |              (derived from Error-bound Falsification class)             |
    ===========================================================================
    Initialization:
    Fals = EBSidak(nModels, nMeas, alpha, paramDist, distVar, meas, func, 
                   residualSD)
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification using Sidak correction
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initialize the class
        """
        self.method = 'Error-bound Sidak'
        EBFals.__init__(self,**kwargs)
        
        
    def doFalsification(self):
        """
        performs falsification
        """
        print('\nUsing Error-bound model falsification with Sidak criterion\n')
        self.alphan = 1.0 - (1.0 -self.alpha)**(1.0/self.nMeas)
        super().doFalsification()
        #EBFals.doFalsification(self)
        
class EBBH(EBFals):
    """
    ===========================================================================
    |            Error-bound Falsification class using BH procedure           |
    |              (derived from Error-bound Falsification class)             |
    ===========================================================================
    Initialization:
    Fals = EBBH(nModels, nMeas, alpha, paramDist, distVar, meas, func, 
                residualSD)
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification using BH procedure
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initializes the class
        """
        self.method = 'Error-bound BH procedure'
        EBFals.__init__(self,**kwargs)
        
        
    def doFalsification(self):
        """
        performs falsification
        """
        print('\nUsing Error-bound model falsification with BH procedure\n')
        self.pValues.sort(axis = 1)
        self.alphaValues = 1.0 - self.pValues
        self.alphan = np.zeros(self.nMeas)
        for i in range(self.nMeas):
            self.alphan[i] = (self.nMeas-i)*self.alpha/self.nMeas
        super().doFalsification()
        #EBFals.doFalsification(self)
        
class LBFals(Fals):
    """
    ===========================================================================
    |                 Likelihood-bound Falsification class                    |
    |                  (derived from Falsification class)                     |
    ===========================================================================
    This class should not be initialized from outside
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initializes the class
        """
        self.t = time.clock()
        allowed_kwargs = {'nModels', 'nMeas', 'alpha' , 'paramDist', 'distVar', 'meas', 'residualSD', 'func'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed to optimizer at: ' + str(k))
        self.__dict__.update(kwargs)
        self.nParam = np.size(self.paramDist)
        
        print('\nTotal number of models = ',self.nModels,'\n')
        Fals.genModels(self)
        Fals.funEval(self)
        Fals.getAlphapValues(self)  
        self.version = '0.0.0'
        
    def calcLikelihood(self):
        """
        calculates the likelihood using Gaussian distribution 
        for residual errors
        """
        self.likelihood = np.zeros(self.nModels)
        # initialize progress bar
        print('\n --------------------------- \n STEP IV: Calculating likelihood of the candidate models \n')
        printProgressBar(0, self.nModels-1, prefix = self.method, suffix = 'Complete', length = 25)
        for i in range(self.nModels):
            self.currentModel = i
            # print progress bar
            Fals.printProgress(self)
            for j in range(self.nMeas):
                err = np.absolute(self.predictions[i,j] - self.meas[j])
                self.likelihood[i] += np.log(stats.norm.pdf(err,0.0,self.residualSD))
    
    def calcLikelihoodLimit(self):
        """
        calculates the likelihood limits from alpha values
        """
        self.likLim = 0
        for i in range(self.nMeas):
            x = stats.norm.ppf(self.alphan[i],loc = 0, scale = self.residualSD)
            self.likLim += np.log(stats.norm.pdf(x,loc=0,scale=self.residualSD))
    
    @abstractmethod    
    def doFalsification(self):
        """
        performs falsification
        """
        # initialize progress bar
        print('\n --------------------------- \n STEP V: Falsifying models \n')
        printProgressBar(0, self.nModels-1, prefix = self.method, suffix = 'Complete', length = 25)
        self.unfalsModels = np.array([[],[]])
        for i in range(self.nModels):
            self.currentModel = i
            # print progress bar
            Fals.printProgress(self)
            if self.likelihood[i]>self.likLim:
                self.unfalsModels = np.append(self.unfalsModels,self.models[i,:])
                    
        self.unfalsModels = np.reshape(self.unfalsModels,(-1,2))
        self.percentUnfals = np.size(self.unfalsModels,0)/self.nModels*100.0
        print('\nNumber of unfalsified models = ', np.size(self.unfalsModels,0),'\n')
        print('\nPercent unfalsified = {0:3.2f}'.format(self.percentUnfals), '%\n')
        
class LBBonferroni(LBFals):
    """
    ===========================================================================
    |     Likelihood-bound Falsification class using Bonferroni criterion     |
    |           (derived from Likelihood-bound Falsification class)           |
    ===========================================================================
    Initialization:
    Fals = LBBonferroni(nModels, nMeas, alpha, paramDist, distVar, meas, func, 
                        residualSD)
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification using 
                                Bonferroni criterion
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initializes the class
        """
        self.method = 'Likelihood-bound Bonferroni criterion'
        print('\nUsing Likelihood-bound model falsification with Bonferroni criterion\n')
        LBFals.__init__(self,**kwargs)
        
        
    def doFalsification(self):
        """
        performs falsification
        """
        LBFals.calcLikelihood(self)
        self.alphan = self.alpha/self.nMeas*np.ones(self.nMeas)
        LBFals.calcLikelihoodLimit(self)
        super().doFalsification()
        
class LBSidak(LBFals):
    """
    ===========================================================================
    |        Likelihood-bound Falsification class using Sidak criterion       |
    |           (derived from Likelihood-bound Falsification class)           |
    ===========================================================================
    Initialization:
    Fals = LBSidak(nModels, nMeas, alpha, paramDist, distVar, meas, func, 
                   residualSD)
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification using 
                                Bonferroni criterion
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initializes the class
        """
        self.method = 'Likelihood-bound Sidak criterion'
        print('\nUsing Likelihood-bound model falsification with Sidak criterion\n')
        LBFals.__init__(self,**kwargs)
        
        
    def doFalsification(self):
        """
        performs falsification
        """
        LBFals.calcLikelihood(self)
        self.alphan = (1.0 - (1.0 -self.alpha)**(1.0/self.nMeas))*np.ones(self.nMeas)
        LBFals.calcLikelihoodLimit(self)
        super().doFalsification()        
        
        
class LBBH(LBFals):
    """
    ===========================================================================
    |         Likelihood-bound Falsification class using BH procedure         |
    |           (derived from Likelihood-bound Falsification class)           |
    ===========================================================================
    Initialization:
    Fals = LBBH(nModels, nMeas, alpha, paramDist, distVar, meas, func, 
                residualSD)
    ===========================================================================
    Attributes:
        models:         Model instances
        nModels:        Total number of models
        nParam:         Number of uncertain parameter in a model
        paramDist:      Names of the probability distributions of the uncertain
                        parameters. 
                        (Currently supports normal/gaussian, uniform, lognormal)
        distVar:        Variables of the uncertain parameter distribution
        nMeas:          Total number of measurements
        predictions:    Predictions by the models
        residualSD:     Standard deviation of the residual error
        pValues:        p values for predictions from all the models
        alphaValues:    alpha values for predictions from all the models
        func:           prediction function
        version:        version of the code
    ===========================================================================
    Methods:
        doFalsification:        performs falsification using 
                                Bonferroni criterion
        inherited:
            genModels:          generates all the model instances
            funEval:            predicts
            getAlphapValues:    calculates alpha and p values
    ===========================================================================
    Reference: De, S., Brewick, P.T., Johnson, E.A. and Wojtkiewicz, S.F., 2018 
    "Investigation of Model Falsification Using Error and Likelihood Bounds 
    with Application to a Structural System."
    Journal of Engineering Mechanics, 144(9), p.04018078.  
    ===========================================================================
    written by Subhayan De (email:Subhayan.De@usc.edu)
    ===========================================================================
    """
    def __init__(self,**kwargs):
        """
        initializes the class
        """
        self.method = 'Likelihood-bound BH procedure'
        print('\nUsing Likelihood-bound model falsification with BH procedure\n')
        LBFals.__init__(self,**kwargs)
        
        
    def doFalsification(self):
        """
        performs falsification
        """
        LBFals.calcLikelihood(self)
        self.pValues.sort(axis = 1)
        self.alphaValues = 1.0 - self.pValues
        self.alphan = np.zeros(self.nMeas)
        for i in range(self.nMeas):
            self.alphan[i] = (self.nMeas-i)*self.alpha/self.nMeas
        LBFals.calcLikelihoodLimit(self)
        super().doFalsification()         
