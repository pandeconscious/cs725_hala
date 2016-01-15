#sg
from numpy import genfromtxt, savetxt
from sklearn.naive_bayes import *
from sklearn.externals import joblib
def main():

    #create the training & test sets, skipping the header row with [1:]
    print 'Begin loading data'
    dataset = genfromtxt(open('../data/strains.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[4] for x in dataset]
    train = [x[1:4] for x in dataset]
    print 'Data loading over'
    print train[0], target[0]
    #test = genfromtxt(open('data/test.csv','r'), delimiter=',', dtype='f8')[1:]
    fileName = 'bigGaussianUpdated.joblib'
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    gnb = GaussianNB()
    model = gnb.fit(train, target)
    joblib.dump(gnb, fileName)
    #predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

    #savetxt('data/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f')

if __name__=="__main__":
    main()
