from svm import *
from testsvm import *

dataMatT, labelMatT = loadDataSet("testSet.txt")
dataMat = mat(dataMatT); labelMat = mat(labelMatT).transpose()
print dataMat
print labelMat
testSvm = svmTrain(dataMat, labelMat)
b,alpha = testSvm.smoP()
supVectorAlp,supVector,supVectorClass = testSvm.svmSurpportVecsGet()
SVMClassfier = svmClassifer(b, alpha, supVector, supVectorClass)
SVMClassfier.jsonDumps("mytest.json")
SVMClassfier.jsonLoadTransfer()
print SVMClassfier