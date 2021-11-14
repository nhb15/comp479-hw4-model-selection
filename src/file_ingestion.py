import pandas as pd

def readFromCSV(file = 'train_sonar_data.csv'):
    train_df = pd.read_csv(file, header=None)
    features = train_df.iloc[:, 0:60]
    labels = train_df.iloc[:, 60].apply(lambda x: 1 if (x == "M") else 0)
    train_df.iloc[:, 60] = labels
    return train_df, features, labels