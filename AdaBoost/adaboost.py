#!/usr/bin/python


#Learn the model using decision tree of SampledTrainingSet.csv and use it to predict the probability of representative of test sequence.


import sys
import csv
import numpy as np
import sklearn.linear_model as sklm
from sklearn.tree import DecisionTreeClassifier 
import sklearn.ensemble as sklm
from sklearn.naive_bayes import GaussianNB
import cPickle
 

with open("questions.csv","rb") as csvQuestionsFile:
        csvQuestionReader = csv.reader(csvQuestionsFile,delimiter=',')
        
        train_vector = np.empty((2956227,3))#np.empty((387,3))#
        class_vector = np.empty( 2956227 )#np.empty(387)#

        print "Creating the training array"

        with open("SampledTrainingSet.csv",'rb') as csvTrainfile: #reading actual train.csv file..NOT YET
                csvTrainReader = csv.reader(csvTrainfile,delimiter=',')

                #creating training data
                for index,row in enumerate(csvTrainReader):
			train_vector[index]= [float(row[1]), float(row[2]), float(row[3])]
                        class_vector[index]= int(row[4])
                        
        print "Training the data set using decision tree"

        dtree = sklm.AdaBoostClassifier(base_estimator=GaussianNB())
        dtree.fit(train_vector, class_vector)
        
        print "Dumping the decision tree model"

        cPickle.dump(dtree, open("dtree_SampledTrainingSet.model","wb"))        
                         
        print "Creating Test vector"

        test_vector = np.empty((90024,3))

        with open("testMeans.txt", 'rb') as csvTestfile: #reading testMeans.csv file.. making prediction based on representative..
                csvTestReader = csv.reader(csvTestfile, delimiter=' ')
                
                for index,row in enumerate(csvTestReader):
                        #print row
			test_vector[index] = [float(row[0]),float(row[1]),float(row[2])]

        deviceDict = {} #dictionary mapping deviceId --> Index (0..387)
        
        with open("classes.txt", 'rb') as csvDevicefile: 
                csvDeviceReader = csv.reader(csvDevicefile, delimiter=',')
                
                for index,row in enumerate(csvDeviceReader):
                        deviceDict[int(row[0])] = index
                
        
        print "Calculating probability predictions for test data set"
                        
        predictions = dtree.predict_proba(test_vector)


        print "Be patient, your results are finally written to output file"

        with open("submission_dtree_SampledDataSet.csv",'wb') as csvSubmissionfile:
                csvSubmissionWriter = csv.writer(csvSubmissionfile,delimiter=',')
                
                csvSubmissionWriter.writerow(["QuestionId","IsTrue"])

                #write code from here....
                for index,row in enumerate(csvQuestionReader):
                        
                                
                        question_id = row[0]
                        seq = int(row[1])
                        device_id = int(row[2])
                        
                        prob = predictions[index][deviceDict[device_id]]

                        csvSubmissionWriter.writerow([question_id,prob])
