import sklearn

def getAccuracySkleran(labels, predictions):
    print(f'labels: {labels}')
    print(f'preds: {predictions}')
    return sklearn.metrics.accuracy_score(labels, predictions)

def getAccuracyPercentageCorrect(features, predictions, labels):
    incorrect_rows = []
    correct_rows = []
    for index, row in enumerate(features):
        if predictions(index) == labels(index):
            correct_rows = correct_rows.append(row)
        else:
            incorrect_rows.append(row)

    return correct_rows, incorrect_rows

def getF1Score(features, predictions, labels):
    print(features, predictions, labels)
