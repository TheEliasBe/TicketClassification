import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os


def train_stacked_model(gpt_run_log_file):
    gpt_prediction_df = pd.read_csv(gpt_run_log_file)
    translated_df = pd.read_csv("../data/02_intermediate/translated_2022.csv")

    gpt_prediction_df = gpt_prediction_df.rename(columns={"prompt": "Text"})
    gpt_prediction_df["Text_Short"] = gpt_prediction_df["Text"].str[:150]
    translated_df["Text_Short"] = translated_df["Text"].str[:150]

    # compute accuracy before stacking
    correct = len(gpt_prediction_df[gpt_prediction_df["completion"] == gpt_prediction_df["classification"]])
    all = len(gpt_prediction_df)
    gpt_accuracy = correct / all
    print(f"Accuracy {correct / all}")

    merged_df = pd.merge(gpt_prediction_df, translated_df, on="Text_Short", how="inner")
    tickets = pd.read_csv("../data/01_raw/2022/tickets.csv")
    merged_df = pd.merge(merged_df, tickets, on="ID")

    # this is the context data
    model_input = merged_df[["Kategorie ID", "Unterkategorie ID", "classification", "completion"]]
    model_input["Kategorie ID"] = model_input["Kategorie ID"].astype("category")
    model_input["Unterkategorie ID"] = model_input["Kategorie ID"].astype("category")
    model_input["classification"] = model_input["classification"].astype("category")
    model_input["completion"] = model_input["completion"].astype("category")

    # one hot encoding
    model_input_oh = pd.get_dummies(model_input, columns=["Kategorie ID", "Unterkategorie ID", "classification"])

    model_input_oh["completion"] = model_input_oh["completion"].cat.codes

    # split into train and test set
    train, test = train_test_split(model_input_oh, test_size=0.2)
    X_train = train.drop(["completion"], axis=1)
    y_train = train["completion"]
    X_test = test.drop(["completion"], axis=1)
    y_test = test["completion"]

    xgb_clf = xgb.XGBClassifier(
        colsample_bytree= 1,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        subsample = 0.85
    )

    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)

    stacked_accuracy = accuracy_score(y_test, y_pred)

    return {"before": gpt_accuracy, "after": stacked_accuracy}


files = [
    "olive-donkey-76.csv",
    "royal-sky-72.csv",
    "blooming-deluge-68.csv",
    "pretty-dew-66.csv"
         ]


results_df = pd.DataFrame(columns=["before", "after"])
for file in files:
    if file.endswith(".csv"):
        print(f"../data/07_model_output/{file}")
        result = train_stacked_model(f"../data/07_model_output/{file}")
        results_df = results_df.append(result, ignore_index=True)


results_df.to_csv("stacked_2.csv")


