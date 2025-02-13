import pandas as pd
from sklearn.model_selection import train_test_split

en_train: pd.DataFrame = pd.read_csv("./en_train.csv.zip")
en_train = en_train.drop_duplicates(subset=["before", "after"], keep="first")
en_train_unique: pd.DataFrame = en_train[en_train["before"] != en_train["after"]]
# en_train.to_csv("./en_train_cleaned.csv")
# en_train_unique.to_csv("./en_train_unique.csv")
# en_test: pd.DataFrame = pd.read_csv("./en_test.csv.zip")
# en_sample_submission: pd.DataFrame = pd.read_csv("./en_sample_submission.csv.zip")

X = en_train["before"]
y = en_train["after"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1999
)

print(en_train.head(10))
