import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 🔹 Load dataset
df = pd.read_csv("cropdata_updated.csv")

# 🔹 Show basic info (optional debug)
print("Dataset loaded successfully")
print(df.head())

# 🔹 Handle categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_stage = LabelEncoder()

df['crop ID'] = le_crop.fit_transform(df['crop ID'])
df['soil_type'] = le_soil.fit_transform(df['soil_type'])
df['Seedling Stage'] = le_stage.fit_transform(df['Seedling Stage'])

# 🔹 Features & target
X = df[['crop ID', 'soil_type', 'Seedling Stage', 'MOI', 'temp', 'humidity']]
y = df['result']

# 🔹 Train-test split (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Accuracy check
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 🔹 Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# 🔹 Save encoders (IMPORTANT for app.py)
with open("le_crop.pkl", "wb") as f:
    pickle.dump(le_crop, f)

with open("le_soil.pkl", "wb") as f:
    pickle.dump(le_soil, f)

with open("le_stage.pkl", "wb") as f:
    pickle.dump(le_stage, f)

print("✅ Model and encoders saved successfully!")