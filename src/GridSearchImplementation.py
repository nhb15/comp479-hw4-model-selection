import csv
import random
import math
import AccuracyUtil
import numpy as np
import file_ingestion

from sklearn.linear_model import LogisticRegressionCV

def runAGridSearchPattern(c_value, tolerance_value):
    train_df, features, labels = file_ingestion.readFromCSV()

    model = LogisticRegressionCV(Cs=[c_value], tol=tolerance_value).fit(features.values, labels.values)

    predictions = []
    for index, row in features.iterrows():
        #print(f'features {features.iloc[index, :]} ')
        data_to_predict = np.array(features.iloc[index, :])
        #print(f'data to predict {data_to_predict}')
        predictions.append(model.predict(data_to_predict.reshape((1, 60))))
    accuracy_score_percentage = AccuracyUtil.getAccuracySkleran(labels, predictions) #check training data first then test data
    return accuracy_score_percentage
    # f1_score = AccuracyUtil.getF1Score()
def main():
    test_c_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
    test_tolerance_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
    results = []

    for c in test_c_values:
        for tolerance in test_tolerance_values:
            accuracy_score = runAGridSearchPattern(c, tolerance)
            results.append([accuracy_score, c, tolerance])
    print(results)


if __name__ == "__main__":
    main()