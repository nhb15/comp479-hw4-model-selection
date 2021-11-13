import pytest
import NFoldClassifier
import file_ingestion

def test_nfold():
    train_df, features, labels = file_ingestion.readFromCSV()
    NFoldClassifier.populateNFoldDataSets(10)