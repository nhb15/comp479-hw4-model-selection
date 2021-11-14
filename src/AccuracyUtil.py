import sklearn

def getF1Sklearn(labels, predictions):
    return sklearn.metrics.f1_score(labels, predictions)

def getAccuracySklearn(labels, predictions):
    return sklearn.metrics.accuracy_score(labels, predictions)