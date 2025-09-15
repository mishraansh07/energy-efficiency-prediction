# Energy Efficiency Prediction

This project predicts the **Heating Load** of buildings using the [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency).  
It applies a **Random Forest Regressor** to model energy efficiency based on various building features.  

---

## 📌 Dataset
- **Source:** UCI Machine Learning Repository  
- **Features (X):** 8 building attributes (e.g., Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution).  
- **Target (y):** Heating Load (kWh/m²).  

---

## ⚙️ Tech Stack
- Python 🐍
- Pandas & NumPy (data handling)
- Scikit-learn (ML model & evaluation)
- Matplotlib (visualization)

---

## 🚀 Project Workflow
1. Load dataset from `ucimlrepo`.  
2. Explore data & check for missing values.  
3. Split into training (80%) and testing (20%).  
4. Train a **Random Forest Regressor**.  
5. Evaluate model with metrics:
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - R² Score  
6. Visualize **Predicted vs Actual Heating Load**.

---

## 📊 Results
- **Mean Absolute Error:** ~0.35  
- **Mean Squared Error:** ~0.24  
- **R² Score:** ~0.997 (very high accuracy ✅)

Scatter Plot (Predicted vs Actual):  
