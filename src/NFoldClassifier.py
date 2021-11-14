import math
import file_ingestion
from sklearn.linear_model import LogisticRegressionCV
import AccuracyUtil

def populateNFoldDataSets(n):
    data_working_list, features, labels = file_ingestion.readFromCSV()
    max_size = len(data_working_list)

    n_index_pool_size = max_size / n if max_size % n == 0 else math.floor(max_size / n)
    test_data = []

    for index in range(n):
        data_working_list = data_working_list.sample(frac=1, axis=0)
        #np.default_rng.shuffle(data_working_list)

        first_x_rows_in_shuffled_dataframe = data_working_list.iloc[:n_index_pool_size, :]
        data_working_list.drop(data_working_list.index[:n_index_pool_size], inplace=True)
        if len(first_x_rows_in_shuffled_dataframe) == n_index_pool_size:
            test_data.append(first_x_rows_in_shuffled_dataframe)
    return test_data

def main():
    train_df_initial, features_initial, labels_initial = file_ingestion.readFromCSV()
    n_test_sets = populateNFoldDataSets(10)
    n_prediction_accuracy_scores = []
    n_prediction_f1_scores = []

    for test_set in n_test_sets:
        train_df = train_df_initial.copy(deep=True)
        features = features_initial.copy(deep=True)
        labels = labels_initial.copy(deep=True)
        test_set_labels = []

        for row in test_set.iterrows():
            train_df.drop(index=row[0], inplace=True)
            features.drop(index=row[0], inplace=True)
            labels.drop(index=row[0], inplace=True)

        n_fold_model = LogisticRegressionCV(max_iter=50000).fit(features.values, labels.values)
        n_fold_predictions = []

        for row in test_set.iterrows():
            # row = np.asarray(row.iloc[:, :]).reshape(1, 60)
            feature_values = row[1]
            feature_values_without_label = feature_values.iloc[0:60]
            test_set_labels.append(feature_values.iloc[60])
            n_fold_predictions.append(n_fold_model.predict([feature_values_without_label]))

        n_prediction_accuracy_scores.append(AccuracyUtil.getAccuracySklearn(test_set_labels, n_fold_predictions))
        n_prediction_f1_scores.append(AccuracyUtil.getF1Sklearn(test_set_labels, n_fold_predictions))
    print(f'accuracies: {n_prediction_accuracy_scores}')
    print(f'f1s: {n_prediction_f1_scores}')

    average_accuracy = sum(n_prediction_accuracy_scores) / len(n_prediction_accuracy_scores)
    average_f1 = sum(n_prediction_f1_scores) / len(n_prediction_f1_scores)

    print(f'average accuracy: {average_accuracy}')
    print(f'average f1: {average_f1}')

if __name__ == "__main__":
    main()