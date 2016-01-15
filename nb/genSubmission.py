#sg

import cPickle
def smallPred():

    ques = open('../data/questions.csv', 'r')
    i = 2
    res = cPickle.load(open('result.p', 'r'))
    ques.readline()
    ques.readline()
    
    MARGIN = 100
    for q in ques:
        ql = q.rstrip('\n').split(',')
        #print ql[1], ql[2], res[i], res[i+1], res[i + 2]

        #print res[i][1], ql[2],
        if(abs(int(res[i][1]) - int(ql[2])) <= MARGIN or abs(int(res[i + 1][1]) - int(ql[2])) <= MARGIN or  abs(int(res[i + 2][1]) - int(ql[2])) <= MARGIN):

            print '%s,%s' % (ql[0], '1')
        else:
            print '%s,%s' % (ql[0], '0')

        i = i + 3

if __name__ == '__main__':
    smallPred()
