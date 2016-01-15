#sg
from numpy import genfromtxt, savetxt
from sklearn.naive_bayes import *
from sklearn.externals import joblib
from sklearn import svm, grid_search

def trainsvm():
#create the training & test sets, skipping the header row with [1:]
    print 'Begin loading data'
    dataset = genfromtxt(open('data/scaledTrain.txt','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[4] for x in dataset]
    train = [x[1:4] for x in dataset]
    print 'Data loading over'
    print train[0], target[0]
    #test = genfromtxt(open('data/test.csv','r'), delimiter=',', dtype='f8')[1:]
    fileName = 'svmmean.joblib'
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    parameters = {'kernel':['rbf'], 'C':[2**i for i in range(-3,11)],  'gamma':[2**i for i in range(-3,11)]} 

    #enable predict_proba
    svr = svm.SVC(probability = True)
    
    #train with cross validation
    clf = grid_search.GridSearchCV(svr, parameters)
   
    clf.fit(train, target)
    
    #dump the trained model
    joblib.dump(clf, fileName)

if __name__ == '__main__':
    trainsvm()
