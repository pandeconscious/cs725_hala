#sg
from numpy import genfromtxt, savetxt
from sklearn.naive_bayes import *
from sklearn.externals import joblib
import numpy as np
import cPickle
def main():

    sgdclf = joblib.load('sgdclf.joblib')
    print 'loaded'
    testF = open('../../data/testmeans15perdev.txt', 'r')
    questionsF = open('../../data/questions.csv', 'r')
    classesF = open('../../data/classes.txt', 'r')
    i = 0
    testMeans = []
    questions = []
    
    classIndexMap = {}
    i = 0
    for line in classesF:
        classIndexMap[int(line.rstrip('\n'))] = i
        i = i + 1


    for line in testF:
        l = line.rstrip('\n').split(' ')
        vals = [float(ll) for ll in l]
        #x = [int(vals[4]), gnb.predict_proba(vals[1:4])]
        testMeans.append(vals)
    
    questionsF.readline()
    for line in questionsF:
        l = line.rstrip('\n').split(',')
        vals = [float(ll) for ll in l]
        questions.append(vals)

    for i in range(0, 90024):

        sumProb =  0
        for j in range(0,15):
            sumProb = sumProb + ((sgdclf.predict_proba(testMeans[i * 15 + j][1:4]))[0])[classIndexMap[int(questions[i][2])]]

        sumProb = float(sumProb) / 15 

        print "%d,%.10f" %(i + 1, sumProb)
        #print "%d,%.10f" %(i + 1,((sgdclf.predict_proba(testMeans[i][1:4]))[0])[classIndexMap[int(questions[i][2])]])

        #print classIndexMap[int(questions[i][2])]
                #print gnb.predict_proba(testMeans[i][1:4])[classIndexMap[int(questions[i][2])]]
    
    
    #cPickle.dump(res, open('result3.p', 'wb'))
if __name__=="__main__":
    main()
