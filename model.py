import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# LOAD DATASET

data = pd.read_csv("salary_prediction.csv")

# Rename column
data.rename(columns={"experience_years": "experience"}, inplace=True)

# Convert education_level to numeric
education_map = {
    "High School": 0,
    "Diploma": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4
}

data["education_level"] = data["education_level"].map(education_map)


# SELECT FEATURES

FEATURE_COLUMNS = ["experience", "education_level", "skills_count", "certifications"]

data = data[FEATURE_COLUMNS + ["salary"]]
data = data.fillna(0)

X = data[FEATURE_COLUMNS]
y = data["salary"]


# TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


# SAVE MODEL + COLUMNS

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(FEATURE_COLUMNS, open("columns.pkl", "wb"))

print("Model + columns saved!")