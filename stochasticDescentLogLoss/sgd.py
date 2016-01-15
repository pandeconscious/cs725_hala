#sg
import numpy as np
from numpy import genfromtxt, savetxt
from sklearn import linear_model
from sklearn.externals import joblib

def main():

    print 'reading data'
    dataset = genfromtxt(open('../../data/strains.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[4] for x in dataset]
    train = [x[1:4] for x in dataset]
    sgdclf = linear_model.SGDClassifier(loss='log')
    print 'training'
    sgdclf.fit(train, target)
    print 'training over'

    print 'dumping model'
    fileName = 'sgdclf.joblib'
    joblib.dump(sgdclf, fileName)
    print 'success, exiting'

if __name__ == '__main__':
    main()
