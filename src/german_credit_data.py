import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split


def data_modification():
    """
    Downloaded the data (german.data) from https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data).
    This function modifies it in order to be used for the experiments. The modification follows Dutta, et al. 2022.
    Procedure:
        1. load the data
        2. Select 10 features according to Dutta, et al. 2022
        3. Make the target variable 0 and 1 instead of 1 and 2
        4. Label encode all the categorical features
        5. Normalise all the features
        6. Split train and test data
        7. Save the data
    """
    le = LabelEncoder()

    cat_cols = [
        "A11",
        "A34",
        "A65",
        "A75",
        "A101",
        "A121",
        "A152",
        "A173",
    ]

    # 1. load the data
    df = pd.read_csv("data/german.data", sep=" ")

    # 2. Select 10 features according to Dutta, et al. 2022
    df = df[
        ["A11", "A34", "1169", "A65", "A75", "A101", "A121", "A152", "2", "A173", "1.1"]
    ]

    #3. Make the target variable 0 and 1 instead of 1 and 2
    df["1.1"] = df["1.1"] - 1

    # 4. Label encode all the categorical features
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 5. Normalise all the features
    normalised_df = pd.DataFrame(Normalizer().fit_transform(X=df.drop(["1.1"], 1)))
    normalised_df["target"] = df["1.1"]

    normalised_df.columns = [
        "existing_checking",
        "credit_history",
        "credit_amount",
        "savings",
        "employment_since",
        "other_debtors",
        "property",
        "housing",
        "existing_credits",
        "job",
        "target",
    ]

    # 6. Split train and test data
    train_df, test_df = train_test_split(normalised_df, test_size=0.3, random_state=42)

    # 7. Save the data
    train_df.to_csv("data/cf_german_train.tsv", sep="\t")
    test_df.to_csv("data/cf_german_test.tsv", sep="\t")


if __name__ == "__main__":
    data_modification()
