# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:10:59 2022

@author: andib
"""

import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import feature_detection as featdec
import reportsplots as rp
from KNN_self import KNN
import Decisiontree as DT


if __name__ == "__main__":
    
    #init 
    #init datapaths with pictures
    #pathsteckdose=r'C:\Prüfungsbilder\Steckdose'
    #pathstecker=r'C:\Prüfungsbilder\Stecker'
    pathsteckdose=r'C:\Users\ogrue\Downloads\jpg2png'
    pathstecker=r'C:\Users\ogrue\Downloads\jpg2png (1)'
    
    #init datapaths for features
    #pathtrain=r'C:\Users\andib\OneDrive\Master-HKA\VDKI\Scripts\17_Grüßer_Hetzel\featuredata_train_10.xlsx'
    #pathtest=r'C:\Users\andib\OneDrive\Master-HKA\VDKI\BilderDaten\fetaturedata\featuredata_test_10.xlsx'
    #pathprüfung=r'C:\Users\andib\OneDrive\Master-HKA\VDKI\Scripts\17_Grüßer_Hetzel\featuredata_prüfung_10.xlsx'
    pathtrain=r'C:\Users\ogrue\OneDrive\VDKI\Scripts\17_Grüßer_Hetzel\featuredata_train_10.xlsx'
    pathtest=r'C:\Users\ogrue\OneDrive\VDKI\BilderDaten\fetaturedata\featuredata_test_10.xlsx'
    pathprüfung=r'C:\Users\ogrue\OneDrive\VDKI\Scripts\17_Grüßer_Hetzel\featuredata_prüfung_10.xlsx'
    
    
    #create a new dataframe with scatterplpt?
    i=1
    
    if i==1:
        #create a featrueframe an calc features from both paths

        dftest=featdec.featuredata()
        featdec.calccnt(dftest, pathsteckdose, 0)
        featdec.calccnt(dftest, pathstecker, 1)
        dftest.toexcel(pathprüfung)
        
        #scatterplot from calculatet features
        #rp.scatterplots(pathprüfung)

    
    #create dataframe with features using pandas
    feature= ['HuMoment0','HuMoment1','HuMoment2','HuMoment3','HuMoment4','HuMoment5','HuMoment6','Area','Circumference','AspectRatio','Category']
    dftrain = pd.read_excel(pathtrain, skiprows=1, header=None, names=feature)
    dftest = pd.read_excel(pathprüfung, skiprows=1, header=None, names=feature)
    
    #create test an train array
    X_train=dftrain.drop(columns=['Category','HuMoment4','HuMoment6'])
    X_test=dftest.drop(columns=['Category','HuMoment4','HuMoment6'])
    y_train=dftrain['Category'].T
    y_test=dftest['Category'].T
    
    #KNN self programmed 
    clf_self = KNN(k=10)
    Xn_train=dftrain.drop(columns=['HuMoment3','Category','HuMoment4','HuMoment5','HuMoment6','Area','Circumference','AspectRatio'])
    Xn_test=dftest.drop(columns=['HuMoment3','Category','HuMoment4','HuMoment5','HuMoment6','Area','Circumference','AspectRatio'])
    clf_self.fit(Xn_train, y_train)
    y_pred_knn_self = clf_self.predict(Xn_test)
    acc_self=rp.accuracy(y_pred_knn_self,y_test)
    print('Accuracy KNN Self: '+str(acc_self))
    rp.labelreport(y_pred_knn_self,y_test,'KNN_Self')
        
    #KNN from seaborn/sklearn
    clf = KNeighborsClassifier() 
    Xn_train=dftrain.drop(columns=['Category','HuMoment4','HuMoment5','HuMoment6','Area','Circumference','AspectRatio'])
    Xn_test=dftest.drop(columns=['Category','HuMoment4','HuMoment5','HuMoment6','Area','Circumference','AspectRatio'])
    clf = clf.fit(Xn_train, y_train)
    y_pred_knn = clf.predict(Xn_test)
    #print("Classification report: \n", classification_report(y_test, y_pred))
    acc=rp.accuracy(y_pred_knn,y_test)
    print('Accuracy KNN SKLearn: '+str(acc))
    rp.labelreport(y_pred_knn,y_test,'KNN SKLearn')
    
    #DecisionTree self programmed
    dt_self=DT.OwnDecisiontreeClassifier()
    dt_self.fit(X_train, y_train)
    y_pred_dt_self=dt_self.predict(X_test)
    acc_self=rp.accuracy(y_pred_dt_self,y_test)
    print('Accuracy Decisiontree self: '+str(acc_self))
    rp.labelreport(y_pred_dt_self,y_test,'Decisiontree self')
    
    #decisiontree from sklearn
    dt = DecisionTreeClassifier()
    dt = dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    #print("Classification report: \n", classification_report(y_test, y_pred))
    acc=rp.accuracy(y_pred_dt,y_test)
    print('Accuracy Decisiontree SKLearn: '+str(acc))
    rp.labelreport(y_pred_dt,y_test,'Decisiontree SKlearn')
    
    #randomforest from sklearn
    rf = RandomForestClassifier(n_estimators=250)
    rf = rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    #print("Classification report: \n", classification_report(y_test, y_pred))
    acc=rp.accuracy(y_pred_rf,y_test)
    print('Accuracy Decisiontree SKLearn: '+str(acc))   
    rp.labelreport(y_pred_rf,y_test,'Decisiontree self')
    
    
    #Create a Confusion Plot
    #knn self
    plt.figure()
    cm_self = rp.confusion_matrix(y_test, y_pred_knn_self)
    cm_plot_labels = ['Steckdose','Stecker']
    rp.plot_confusion_matrix(cm=cm_self, classes=cm_plot_labels, title='Confusion Matrix')
    all_sample_title = 'KNN self \n Accuracy: '+str(rp.accuracy(y_pred_knn_self,y_test))
    plt.title(all_sample_title, size=15)
    
    #knn sklearn
    plt.figure()
    cm = rp.confusion_matrix(y_test, y_pred_knn)
    cm_plot_labels = ['Steckdose','Stecker']
    rp.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    all_sample_title = 'KNN SKLearn \n Accuracy: '+str(rp.accuracy(y_pred_knn,y_test))
    plt.title(all_sample_title, size=15)
    
    #decision tree self
    plt.figure()
    cm_self = rp.confusion_matrix(y_test, y_pred_dt_self)
    cm_plot_labels = ['Steckdose','Stecker']
    rp.plot_confusion_matrix(cm=cm_self, classes=cm_plot_labels, title='Confusion Matrix')
    all_sample_title = 'Decisiontree self \n Accuracy: '+str(rp.accuracy(y_pred_dt_self,y_test))
    plt.title(all_sample_title, size=15)
    
    #decision tree sklearn
    plt.figure()
    cm = rp.confusion_matrix(y_test, y_pred_dt)
    cm_plot_labels = ['Steckdose','Stecker']
    rp.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    all_sample_title = 'Decisiontree SKLearn \n Accuracy: '+str(rp.accuracy(y_pred_dt,y_test))
    plt.title(all_sample_title, size=15)   

    #randomforest
    plt.figure()
    cm = rp.confusion_matrix(y_test, y_pred_rf)
    cm_plot_labels = ['Steckdose','Stecker']
    rp.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    all_sample_title = 'RandomForest SKLearn \n Accuracy: '+str(rp.accuracy(y_pred_rf,y_test))
    plt.title(all_sample_title, size=15)   
    
    #print classificationreports
    print('>>Classification Report KNNself<< \nUsed Features: HuMoment0, HuMoment1, HuMoment2, HuMoment3 \n' +classification_report(y_test,y_pred_knn_self))
    print('>>Classification Report KNN<< \nUsed Features: HuMoment0, HuMoment1, HuMoment2, HuMoment3 \n'+classification_report(y_test,y_pred_knn))
    print('>>Classification Report Decisiontree self<< \nUsed Features: HuMoment0, HuMoment1, HuMoment2, HuMoment3 \nHuMoment5, Area, Circumference, AspectRatio \n'+classification_report(y_test,y_pred_dt_self))
    print('>>Classification Report Decisiontree<<  \nUsed Features: HuMoment0, HuMoment1, HuMoment2, HuMoment3 \nHuMoment5, Area, Circumference, AspectRatio \n'+classification_report(y_test,y_pred_dt))
    print('>>Classification Report Randomforest<< \nUsed Features: HuMoment0, HuMoment1, HuMoment2, HuMoment3 \nHuMoment5, Area, Circumference, AspectRatio \n'+classification_report(y_test,y_pred_rf))
    