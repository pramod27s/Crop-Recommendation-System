
---

# 🌱 Crop Recommendation System

A **Machine Learning-based Crop Recommendation System** that predicts the most suitable crop for cultivation based on soil and environmental parameters. The project uses **Ensemble Models (Random Forest, Gradient Boosting, KNN, and Soft Voting Classifier)** and provides **visual insights** like feature importance, correlations, and data distributions.

---

## 🚀 Features

* Predicts the most suitable crop for given conditions
* User-friendly input system with **soil type selection**
* Handles categorical and numerical data
* Ensemble model combining Random Forest, Gradient Boosting, and KNN
* Model performance comparison across multiple algorithms
* Visualizations:

  * Top 5 important features
  * Correlation heatmap
  * Feature distributions
  * Model accuracy comparison

---

## 🛠️ Technologies Used

* **Python 3.8+**
* **Pandas, NumPy** (data handling)
* **Matplotlib, Seaborn** (visualization)
* **Scikit-learn** (ML models, evaluation)

---

## 📂 Project Structure

```
📦 CropRecommendationSystem
 ┣ 📜 crop_recommendation.py   # Main project file
 ┣ 📜 crop_recommendation_dataset.csv   # Dataset file
 ┗ 📜 README.md                # Documentation
```

---

## ▶️ How to Run

### Prerequisites

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/CropRecommendationSystem.git
   cd CropRecommendationSystem
   ```
2. Ensure `crop_recommendation_dataset.csv` is in the project folder.
3. Run the script:

   ```bash
   python crop_recommendation.py
   ```

---

## 🎯 Usage

* Enter environmental values (temperature, humidity, pH, rainfall, etc.) when prompted
* Choose soil type from a predefined list:

  * 1: Sandy
  * 2: Clay
  * 3: Loamy
  * 4: Black
  * 5: Red
  * 6: Alluvial
  * 7: Acidic Soil
  * 8: Saline Soil

The system will output the **predicted crop** 🌾.

---

## 📊 Visualizations

1. **Top 5 Important Features**
   Shows which environmental factors influence prediction most.

2. **Correlation Heatmap**
   Visualizes feature relationships.

3. **Feature Distributions**
   Frequency plots of all numerical features.

4. **Model Accuracy Comparison**
   Bar graph comparing accuracies of Random Forest, Gradient Boosting, KNN, and the Ensemble model.

---

## 📈 Model Performance

The ensemble model generally achieves **higher accuracy** than individual models.
(Exact values depend on dataset splits).

---





