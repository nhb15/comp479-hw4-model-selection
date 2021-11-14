import AccuracyUtil
import numpy as np
import file_ingestion
import csv

from sklearn.linear_model import LogisticRegressionCV

def runAGridSearchPattern(c_value, tolerance_value):
    train_df, features, labels = file_ingestion.readFromCSV()

    model = LogisticRegressionCV(Cs=[c_value], tol=tolerance_value).fit(features.values, labels.values)

    predictions = []
    for index, row in features.iterrows():
        data_to_predict = np.array(features.iloc[index, :])
        predictions.append(model.predict(data_to_predict.reshape((1, 60))))
    accuracy_score_percentage = AccuracyUtil.getAccuracySklearn(labels, predictions) #check training data first then test data
    f1_score = AccuracyUtil.getF1Sklearn(labels, predictions)
    return accuracy_score_percentage, f1_score


def runOnFinalTestData():
    train_df, features, labels = file_ingestion.readFromCSV()

    model = LogisticRegressionCV(Cs=[100], tol=0.1).fit(features.values, labels.values)

    test_df, features, labels = file_ingestion.readFromCSV('test_data.csv')
    predictions = []

    for index, row in features.iterrows():
        data_to_predict = np.array(features.iloc[index, :])
        predictions.append(model.predict(data_to_predict.reshape((1, 60))))

    accuracy_score_percentage = AccuracyUtil.getAccuracySklearn(labels, predictions) #check training data first then test data
    f1_score = AccuracyUtil.getF1Sklearn(labels, predictions)
    print(f'final accuracy: {accuracy_score_percentage}')
    print(f'final f1: {f1_score}')
    return accuracy_score_percentage, f1_score

def main():
    test_c_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    test_tolerance_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []

    with open("gridresults.csv", 'w') as file:
        writer = csv.writer(file)
        for c in test_c_values:
            for tolerance in test_tolerance_values:
                accuracy_score, f1_score = runAGridSearchPattern(c, tolerance)
                writer.writerow([f1_score, accuracy_score, c, tolerance])
                results.append([f1_score, accuracy_score, c, tolerance])
    print(results)

    runOnFinalTestData()

if __name__ == "__main__":
    main()