import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
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
    ohe = OneHotEncoder(handle_unknown="ignore", drop="first")

    cat_cols = [
        "A11",
        "A34",
        "A43",
        "A65",
        "A75",
        "A93",
        "A101",
        "A121",
        "A143",
        "A152",
        "A173",
        "A192",
        "A201",
    ]

    # 1. load the data
    df = pd.read_csv("data/german.data", sep=" ")

    # 2. rename columns
    # normalised_train_df.columns = [
    #     "existing_checking",
    #     "credit_history",
    #     "credit_amount",
    #     "savings",
    #     "employment_since",
    #     "other_debtors",
    #     "property",
    #     "housing",
    #     "existing_credits",
    #     "job",
    #     "target",
    # ]

    # 3. Make the target variable 0 and 1 instead of 1 and 2
    df["1.1"] = df["1.1"] - 1

    # 6. Split train and test data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df, test_df = train_df.reset_index(), test_df.reset_index()

    ohe_train_df = pd.DataFrame(ohe.fit_transform(train_df[cat_cols]).toarray(),
                                columns=ohe.get_feature_names_out())
    ohe_test_df = pd.DataFrame(ohe.transform(test_df[cat_cols]).toarray(),
                               columns=ohe.get_feature_names_out())

    train_df, test_df = train_df.drop(cat_cols, 1), test_df.drop(cat_cols, 1)
    train_df, test_df = pd.concat([train_df, ohe_train_df], 1), pd.concat([test_df, ohe_test_df], 1)

    # 5. Normalise all the features
    normalised_train_df = pd.DataFrame(Normalizer().fit_transform(X=train_df.drop(["1.1"], 1)))
    normalised_train_df["target"] = train_df["1.1"]
    normalised_test_df = pd.DataFrame(Normalizer().fit_transform(X=test_df.drop(["1.1"], 1)))
    normalised_test_df["target"] = test_df["1.1"]

    # 7. Save the data
    train_df.to_csv("data/cf_german_train.tsv", sep="\t")
    test_df.to_csv("data/cf_german_test.tsv", sep="\t")


if __name__ == "__main__":
    data_modification()
