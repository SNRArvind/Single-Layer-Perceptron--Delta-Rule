# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:34:33 2018

Roll No: 18AT91R02
Name: Arvind Kumar Gupta
Assignment No.: 6

"""
#%% Import libraries
from __future__ import print_function
import pandas as pd
import numpy as np
#%%
trainingSet_Data = pd.read_csv('data6.csv', header=None)
trainingSet_Data = trainingSet_Data.values
testingSet_Data = pd.read_csv('test6.csv', header=None)
testingSet_Data = testingSet_Data.values
   
#%% definition of functions

def sigmaH(inputsMat, weightsMat):
    sumH = 0.0
    for i,wt in zip(inputsMat, weightsMat):
        sumH +=i*wt           
    return sumH

def predictionT(inputsMat, weightsMat):
    activationFn = 0.0
    for i,wt in zip(inputsMat, weightsMat):
        activationFn +=i*wt        
    expFn = np.exp(-activationFn)
    sigmoFn = 1/(1+expFn)   
    return 1 if sigmoFn>=0.5 else 0

def trainingWeightsFn(dataMat,weightsMat,givenEpoch,lrnRate):
    for epoch in range(givenEpoch):
        for i in range(len(dataMat)):
            sigH = sigmaH(dataMat[i][:-1],weightsMat) 
            errorVal      = dataMat[i][-1]-sigH		 
            for j in range(len(weightsMat)):
                  weightsMat[j] = weightsMat[j]+(lrnRate*errorVal*dataMat[i][j]) 

    return weightsMat 

def main():
    
    epochGiven		= 10
    learningRate  		= 0.01
    
    weightVector= [0, 0.2,	.5,  .3, .7,	.2,  0.13, 0.7,	0.22] # initial weights specified in problem
    trainingMat_Data = trainingSet_Data
    Row,Column = trainingMat_Data.shape
    BiasValue = np.ones((Row,1))   #Bias as 1
    newTrainingMat_Data = np.hstack(( BiasValue, trainingMat_Data))   #add bias in data set
    testingMat_Data = testingSet_Data
    roW,columN = testingMat_Data.shape
    biaS = np.ones((roW,1))   #Bias as 1
    newTestingMat_Data = np.hstack(( biaS, testingMat_Data))   #add bias in data set

    trainedWeight_Vect = trainingWeightsFn(newTrainingMat_Data,weightVector,epochGiven,learningRate)
#    print("\n Weight vector w0 to W8: \n")
#    print(trainedWeight_Vect)
#    print('\n')
    predscton       = []
    returnPrediction = []
    for i in range(len(newTestingMat_Data)):
        predscton   = predictionT(newTestingMat_Data[i],trainedWeight_Vect) # get predicted classification
        print("Predicted Class:",predscton)
        returnPrediction.append(predscton)
       
    return returnPrediction

#%%
if __name__ == '__main__':
    predictO = main()
    np.set_printoptions(threshold=np.nan)
    f2= open("18AT91R02_6.out","w+")
    f2.write(str(predictO))
    f2.close()
