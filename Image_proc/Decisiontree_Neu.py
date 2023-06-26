# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:19:49 2022

@author: ogrue
"""
from math import log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



class OwnDecisiontreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.aAdjazenz = None
        if max_depth == None: self.iMaxDepth = 13051998
        else: self.iMaxDepth = max_depth
        self.iMinSplit = min_samples_split
        
    def fit(self, dfInput, aLabels):
        '''
        Creates the desicion tree based on the training set. The desicion tree is saved in the attribute self.aAdjazenz.
        This is a kind of a 2D-Adjazenzmatrix. If there is 1 entry than the row node is connectet with the column node.
        If the 1 is in the diagonal than the node is a leaf node. Otherwise in the diagonale is a tuple saved with the
        featurename on the first index and the threshold for the split.

        Parameters
        ----------
        dfInput : pandas.DataFrame
            Features of the training data.
        aLabels : Series or Array
            Coresponding labels to the input dataframe.

        Returns
        -------
        None.

        '''
        self.aAdjazenz = np.array([[None]])
        iLevel = 0              # depth level in the tree
        iLastCurrentLevel = 0   # number of the last Node in the current level
        iLastNextLevel = 0      # number of the last Node in the next level    
        iCurrentNode = 0
        bCreateTree = True      # break condition for deeper trees
        
        lIndices = dfInput.index
        aNodeSet = [lIndices]
        
        while bCreateTree:
            
            # check if node is a leaf (1 in diag of Adjazenzmetrix)
            if self.aAdjazenz[iCurrentNode, iCurrentNode] == None:
                
                # add two nodes in Adjazenzmetrix
                for i in range(2):
                    tShape = self.aAdjazenz.shape
                    aAddition = np.array([[None for i in range(tShape[0])]])
                    self.aAdjazenz = np.append(self.aAdjazenz, aAddition, axis=0)
                    aAddition = np.array([[None for i in range(tShape[0]+1)]])
                    
                    self.aAdjazenz = np.append(self.aAdjazenz, np.transpose(aAddition), axis=1)
                    # connection from actual node to created child
                    self.aAdjazenz[iCurrentNode,tShape[0]] = 1
                    iLastNextLevel += 1
                
                # variables for best split
                sFeature = None
                fHighestGain = 0
                fThresholdBest = None
                lNodeIndices = aNodeSet[iCurrentNode]
                index = dfInput.iloc[lNodeIndices].index
                
                # Parent Node
                iCountParentNode = len(lNodeIndices)
                
                aNodeOutput = aLabels.iloc[lNodeIndices]
                iCountOne = aNodeOutput[aNodeOutput == 1].shape[0]
                iCountZero = aNodeOutput[aNodeOutput == 0].shape[0]
                
                if iCountOne/iCountParentNode == 0: fE1 = 0             # can't calculate log(0,2) -> math domain error
                else: fE1 = -1*iCountOne/iCountParentNode * log(iCountOne/iCountParentNode,2)
                if iCountZero/iCountParentNode == 0: fE2 = 0
                else: fE2 = iCountZero/iCountParentNode * log(iCountZero/iCountParentNode,2)
                fEntropy = fE1 - fE2
                #print('Count Parent Node: ' + str(iCountParentNode))
                
                for feature in dfInput.columns:
                    serValues = dfInput.iloc[lNodeIndices][feature]
                    
                    # average is threshold for split
                    fThreshold = serValues.mean()
                    serCondition = serValues > fThreshold
                    lIndices = index[serCondition].tolist()
                    
                    serOutputGreater = aLabels[lIndices]
                    serOutputSmaller = aLabels.drop(index=lIndices)
                    
                    # Node Greater
                    iCountGreater = len(serOutputGreater)
                    iCountOne = serOutputGreater[serOutputGreater == 1].shape[0]
                    iCountZero = serOutputGreater[serOutputGreater == 0].shape[0]
                    
                    if iCountOne/iCountGreater == 0: fE1 = 0
                    else: fE1 = -1*iCountOne/iCountGreater * log(iCountOne/iCountGreater,2)
                    if iCountZero/iCountGreater == 0: fE2 = 0
                    else: fE2 = iCountZero/iCountGreater * log(iCountZero/iCountGreater,2)
                    fEntropyGreater = fE1 - fE2 
        
                    # Node Smaller
                    iCountSmaller = len(serCondition) - iCountGreater
                    iCountOne = serOutputSmaller[serOutputSmaller == 1].shape[0]
                    iCountZero = serOutputSmaller[serOutputSmaller == 0].shape[0]
                    
                    if iCountOne/iCountSmaller == 0: fE1 = 0
                    else: fE1 = -1*iCountOne/iCountSmaller * log(iCountOne/iCountSmaller,2) 
                    if iCountZero/iCountGreater == 0: fE2 = 0
                    else: fE2 = iCountZero/iCountSmaller * log(iCountZero/iCountSmaller,2) 
                    fEntropySmaller = fE1 - fE2
                    
                    # calculate Gain
                    fGain = fEntropy - (iCountGreater/iCountParentNode * fEntropyGreater + iCountSmaller/iCountParentNode * fEntropySmaller)
                    
                    if fGain > fHighestGain:
                        fHighestGain = fGain
                        sFeature = feature
                        fThresholdBest = fThreshold
                        
        
                # write best split in desicion tree
                if sFeature != None:
                    self.aAdjazenz[iCurrentNode, iCurrentNode] = (sFeature, fThresholdBest)
                    serCondition = dfInput.iloc[lNodeIndices][sFeature] > fThresholdBest
                    lIndicesGreater = index[serCondition]
                    aNodeSet.append(lIndicesGreater)
                    lIndicesSmaller = aNodeSet[iCurrentNode].drop(lIndicesGreater)
                    aNodeSet.append(lIndicesSmaller)
                else: 
                    self.aAdjazenz[iCurrentNode, iCurrentNode] = 1
                    for i in range(2):
                        self.aAdjazenz = np.delete(self.aAdjazenz,-1,0)
                        self.aAdjazenz = np.delete(self.aAdjazenz,-1,1)
                
                #print('CountGreater: ' + str(len(lIndicesGreater)) + '\nCountSmaller: ' + str(len(lIndicesSmaller)) + '\n')
                
                
                ## check if child nodes with best split are leafs
                # check for Hyperparameters
                if iLevel+1 >= self.iMaxDepth or len(lIndicesGreater) < self.iMinSplit: 
                    bLeafGreater = True
                else: bLeafGreater = False 
                if iLevel+1 >= self.iMaxDepth or len(lIndicesSmaller) < self.iMinSplit: 
                    bLeafSmaller = True
                else: bLeafSmaller = False 
          
                
                serOutputChild = aLabels.iloc[lIndicesGreater]
                # if all labels in child node "greater" are equal or by Hyperparameter
                if (serOutputChild == serOutputChild[serOutputChild.index[0]]).all() == True or bLeafGreater:
                    iCountOne = serOutputChild[serOutputChild == 1].shape[0]
                    iCountZero = serOutputChild[serOutputChild == 0].shape[0]
                    if iCountOne > iCountZero: self.aAdjazenz[tShape[0]-1, tShape[0]-1] = 1
                    else: self.aAdjazenz[tShape[0]-1, tShape[0]-1] = 0
                serOutputChild = aLabels.iloc[lIndicesSmaller]  
                # if all labels in child node "smaller" are equal or by Hyperparameter
                if (serOutputChild == serOutputChild[serOutputChild.index[0]]).all() == True or bLeafSmaller:
                    iCountOne = serOutputChild[serOutputChild == 1].shape[0]
                    iCountZero = serOutputChild[serOutputChild == 0].shape[0]
                    if iCountOne > iCountZero: self.aAdjazenz[tShape[0], tShape[0]] = 1
                    else: self.aAdjazenz[tShape[0], tShape[0]] = 0
                             
            # check if all open nodes are leafs
            aDiag = self.aAdjazenz.diagonal()
            if len(aDiag[aDiag == None]) == 0: 
                bCreateTree = False
                    
            if iCurrentNode == iLastCurrentLevel:
                iLastCurrentLevel = iLastNextLevel
                iLevel += 1
            iCurrentNode += 1
        
    def predict(self, x):
        '''
        Uses the model to classify new data.

        Parameters
        ----------
        x : DataFrame
            DataFrame with the new data to classify.

        Returns
        -------
        aPrediction : Array
            Array with the predicted labels.

        '''
        aPrediction = np.empty([len(x)]) 
        
        for i, datapoint in x.iterrows():
            bRunTree = True
            iCurrentNode = 0
            
            while bRunTree: 
                # leaf node
                if self.aAdjazenz[iCurrentNode][iCurrentNode] == 1 or self.aAdjazenz[iCurrentNode][iCurrentNode] == 0:
                    bRunTree = False
                    aPrediction[i] = self.aAdjazenz[iCurrentNode][iCurrentNode]
                    
                # follow split
                else:
                    tSplit = self.aAdjazenz[iCurrentNode][iCurrentNode]
                    # get child nodes
                    aChildNodes = np.where(self.aAdjazenz[iCurrentNode] != None)[0]
                    if datapoint[tSplit[0]] > tSplit[1]:
                        iCurrentNode = aChildNodes[-2]
                    else:
                        iCurrentNode = aChildNodes[-1]
        
        return aPrediction
    
    def getMATLABAdjazenz(self):
        '''
        Converts the attribut self.aAdjazenz to a true Adjazenzmetrix to plot the tree in MATLAB.

        Returns
        -------
        aMATLAB : Array
            Adjazenzmetrix of the tree.

        '''
        aMATLAB = self.aAdjazenz.copy()
        for i, Node in enumerate(aMATLAB):
            aNotOne = np.where(Node != 1)[0]
            for j in aNotOne:
                aMATLAB[i][j] = 0
            aMATLAB[i][i] = 0
        return aMATLAB
                
            
                        
     
     
        
     
        

        
if __name__ == "__main__":

#    # load Data
 #   dfData = pd.read_excel(r'C:\Users\ogrue\OneDrive\VDKI\BilderDaten\fetaturedata\featuredata_train_09.xlsx')
  #  dfTest = pd.read_excel(r'C:\Users\ogrue\OneDrive\VDKI\BilderDaten\fetaturedata\featuredata_test_09.xlsx')

   # aOutput = dfData['Category']
    #dfInput = dfData.drop(['Category'], axis=1)
#    aTest = dfTest['Category']
 #   dfTest = dfTest.drop(['Category'], axis=1)
    
  #  OwnTree = OwnDecisiontreeClassifier(min_samples_split=6)
   # OwnTree.fit(dfInput, aOutput)
    #pred = OwnTree.predict(dfTest)
    
#    aTestScore = pred == aTest
 #   iCountRichtig = pred[pred == aTest].shape[0]
  #  iCountFalsch = pred[pred == aTest].shape[0]
   # print('Finish')
    None
    