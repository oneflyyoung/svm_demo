
from numpy import *
from object_json import *
import pdb


def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def simpleTest():
    dataMatT,labelMatT = loadDataSet('testSet.txt')
    datMat=mat(dataMatT); labelMat = mat(labelMatT).transpose()
    
    try:
        SVMClassifier_simple = objectLoadFromFile('SVMClassifier_simple.json')
        SVMClassifier_simple.jsonLoadTransfer()
        print 'load SVMClassifier successfully'
    except IOError, ValueError:
        from svm import *
        print 'SVM classifer file doesnt exist, Train first'
        
        testSvm = svmTrain(datMat, labelMat, 0.6, 0.001)
        b,alpha = testSvm.smoP()
        vecAlp, vec, vecClass = testSvm.svmSurpportVecsGet()
        SVMClassifier_simple = svmClassifer(b, vecAlp, vec, vecClass)
        SVMClassifier_simple.jsonDumps('SVMClassifier_simple.json')
        #pdb.set_trace()
        SVMClassifier_simple.jsonLoadTransfer()
        
    m,n = datMat.shape
    errorCount = 0.0
    for  i in range(100):
        dataIn = datMat[i,:]
        result = SVMClassifier_simple.svmClassify(dataIn)
        print 'predict result is: ',result, ' real result is: ',labelMatT[i]
        if result != labelMatT[i]: errorCount += 1
    print 'errorRate:', errorCount/100

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    filename = 'SVMClassifierRbf.json'

    try:
        SVMClassifier = objectLoadFromFile(filename)
        SVMClassifier.jsonLoadTransfer()
        print 'load SVMClassifier successfully'
    except IOError, ValueError:
        from svm import *
        print 'SVM classifer file doesnt exist, Train first'
        testSvm = svmTrain(datMat, labelMat, 200, 0.001, ('rbf', k1))
        b,alpha = testSvm.smoP(10000)
        vecAlp, vec, vecClass = testSvm.svmSurpportVecsGet()
        SVMClassifier = svmClassifer(b, vecAlp, vec, vecClass,('rbf',k1))
        SVMClassifier.jsonDumps(filename)# change to python default list
        #pdb.set_trace()
        SVMClassifier.jsonLoadTransfer()#change back to numpy matrix
    
    m,n = shape(datMat)
    errorCount = 0.0
    for i in range(m):
        result = SVMClassifier.svmClassify(datMat[i,:])
        if result!=sign(labelArr[i]):
            print 'training predict result is: ',result, ' training real result is: ',sign(labelArr[i])
            errorCount += 1
    print "the training error rate is: %2.2f%%" % ((float(errorCount)/m)*100)

    
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        result = SVMClassifier.svmClassify(datMat[i,:])
        if result!=sign(labelArr[i]):
            print 'test predict result is: ',result, ' test real result is: ',sign(labelArr[i])
            errorCount += 1    
    print "the test error rate is: %2.2f%%" % ((float(errorCount)/m)*100 )

if __name__ == '__main__':

    simpleTest()
    print 'k1=%f'%1.3
    testRbf()
