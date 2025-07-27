import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

#Load Dataset
try:
    df = pd.read_csv("Downloads/crop_recommendation_dataset.csv")  # Update path if needed
except FileNotFoundError:
    raise FileNotFoundError("Dataset file not found. Please check the file path.")

print("\n Dataset Info:")
print(df.info())
print(df.head())

#Feature Preparation
target_col = 'Crop'
X = df.drop(columns=[target_col])
y = df[target_col]
original_features = X.columns.tolist()

# Soil Mapping (for categorical input)
soil_mapping = {
    1: 'Sandy',
    2: 'Clay',
    3: 'Loamy',
    4: 'Black',
    5: 'Red',
    6: 'Alluvial',
    7: 'Acidic Soil',
    8: 'Saline Soil'
}

#Identify categorical (soil) column
soil_col_name = None
for col in X.columns:
    if X[col].dtype == 'object':
        soil_col_name = col
        break

if soil_col_name is None:
    print("No categorical soil column found. Proceeding with numerical-only features.")

#Feature Ranges for Validation
feature_ranges = {}
for col in X.columns:
    if col != soil_col_name:
        min_val = df[col].min()
        max_val = df[col].max()
        allowed_min = min_val - 0.05 * abs(min_val)
        allowed_max = max_val + 0.05 * abs(max_val)
        feature_ranges[col] = (allowed_min, allowed_max)

#One-hot Encode Categorical Variables
X_encoded = pd.get_dummies(X)

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#Define Ensemble Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('gb', gb),
    ('knn', knn)
], voting='soft')

#Train Model
ensemble.fit(X_train, y_train)

#Evaluate Model
y_pred = ensemble.predict(X_test)
print("\n Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

#User Input for Prediction
print("\n Enter the following values to predict the crop:")

user_input_dict = {}
for feature in original_features:
    if feature == soil_col_name:
        print("\n Choose soil type:")
        for key, value in soil_mapping.items():
            print(f"{key}: {value}")
        while True:
            try:
                soil_choice = int(input("Enter soil type number: "))
                if soil_choice in soil_mapping:
                    user_input_dict[feature] = soil_mapping[soil_choice]
                    break
                else:
                    print(" Invalid choice. Please select a number from the list.")
            except ValueError:
                print(" Invalid input. Please enter a number.")
    else:
        min_allowed, max_allowed = feature_ranges[feature]
        while True:
            try:
                val = float(input(f"{feature} (range: {min_allowed:.2f} - {max_allowed:.2f}): "))
                if min_allowed <= val <= max_allowed:
                    user_input_dict[feature] = val
                    break
                else:
                    print(f" Value out of allowed range. Please enter between {min_allowed:.2f} and {max_allowed:.2f}")
            except ValueError:
                print(" Invalid input. Please enter a number.")

#Prepare and Encode User Input
user_df = pd.DataFrame([user_input_dict])
user_df_encoded = pd.get_dummies(user_df)
user_df_encoded = user_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

#Predict Crop
predicted_crop = ensemble.predict(user_df_encoded)[0]
print(f"\n Predicted Crop: {predicted_crop}")

#1. Feature Importance (Top 5)
rf.fit(X_train, y_train)  # Ensure model is trained
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_importances = importances.sort_values(ascending=False).head(5)

plt.figure(figsize=(8, 4))
top_importances.plot(kind='barh', color='teal')
plt.title("Top 5 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

#2. Correlation Heatmap
plt.figure(figsize=(10, 8))
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_features].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

#3. graphs on the frequency of instance of all columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#4. Model Accuracy Comparison
models = {
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'KNN': knn,
    'Ensemble': ensemble
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, preds)

plt.figure(figsize=(8, 4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='pastel')
plt.title("Model Accuracy (%)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontsize=10)
plt.tight_layout()
plt.show()