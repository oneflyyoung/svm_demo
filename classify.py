
import xlrd
import numpy
from svm import *
import random

def getRows(path):
    book = xlrd.open_workbook(path)
    table = book.sheets()[0]
    t = type(table.row(1)[1].value)

    row = 0
    for x in range(1, table.nrows):
        count = 0
        for y in range(table.ncols):
            if(type(table.row(x)[y].value) == t):
                count += 1
        if(count == table.ncols):
            row += 1

    return row


def readData(path):
    book = xlrd.open_workbook(path)
    table = book.sheets()[0]
    nrows = table.nrows
    ncols = table.ncols
    row = getRows(path)
    dMatrix = numpy.zeros((row, ncols-1))

    t=type(table.row(1)[1].value)

    index = 0
    for x in range(1,nrows):
        count = 0
        for y in range(ncols):
            if(type(table.row(x)[y].value) == t):
                count += 1
        if(count == ncols):
            dMatrix[index] = table.row_values(x)[:-1]
            index += 1

    lMatrix = []
    for i in range(1,nrows):
        if(type(table.col_values(ncols-1)[i]) == float):
            lMatrix.append(table.col_values(ncols-1)[i])

    return dMatrix, lMatrix


def train():
    dMatrix, lMatrix = readData('RData_.xlsx')
    dMat = mat(dMatrix[:-10])
    lMat = mat(lMatrix[:-10]).transpose()
    #lMat = numpy.array(lMatrix,dtype=float).T
    #lMat.astype(numpy.float)

    _lMat = []
    for i in range(1, 10):
        _lMat = lMat
        for x in range(len(_lMat)):
            if(_lMat[x] == i):
                _lMat[i] = 1
            else:
                _lMat[i] = -1
        tsvm = svmTrain(dMat, lMat)
        b, alpha = tsvm.smoP()
        vecAlp, vec, vecClass = tsvm.svmSurpportVecsGet()
        csvm = svmClassifer(b, vecAlp, vec, vecClass)
        path = 'model' + str(i) + '.json'
        csvm.jsonDumps(path)

def test():
    dMatrix, lMatrix = readData('RData_.xlsx')
    dMat = mat(dMatrix[-100:])
    lMat = mat(lMatrix[-100:]).transpose()

    csvm = objectLoadFromFile('model.json')
    csvm.jsonLoadTransfer()

    errorCount = 0
    for i in range(100):
        #x = random.randint(1, 2336)
        dataIn = dMat[i]
        result = csvm.svmClassify(dataIn)
        print 'predict result is: ', result, ' real result is: ', lMat[i][0]
        if result != lMat[i][0]: errorCount += 1
    print 'errorCount:', errorCount
    print 'testCount: ', lMat.shape


if __name__ == '__main__':
    train()
    #test()

