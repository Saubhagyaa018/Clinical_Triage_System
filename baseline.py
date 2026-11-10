import pandas as pd
from sqlalchemy import text
from utils.database import engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


query = text("""
    SELECT
        stay_id,
        subject_id,
        chiefcomplaint,
        heartrate,
        resprate,
        o2sat,
        sbp,
        dbp,
        ground_truth_acuity
    FROM model_ready_triage
    WHERE ground_truth_acuity IN (1, 2, 3)
    ORDER BY stay_id
""")


with engine.connect() as conn:
    df = pd.read_sql(query, conn)


print("Number of records:", len(df))
print()
print("First 5 records:")
print(df.head())
print()
print("Class distribution:")
print(df["ground_truth_acuity"].value_counts().sort_index())

X = df[
    [
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp"
    ]
]

y = df["ground_truth_acuity"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print()
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


print()
print("Training class distribution:")
print(y_train.value_counts().sort_index())

print()
print("Testing class distribution:")
print(y_test.value_counts().sort_index())


#after traning 
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print()
print("Accuracy:", accuracy)

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print()
print("Predictions:")
print(y_pred)