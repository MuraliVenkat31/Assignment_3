## 🚗 Used Car Price Prediction – CarDekho Dataset

### 📌 Overview

This project aims to build a machine learning model to **predict the selling price of used cars** based on various features such as the car's age, kilometers driven, fuel type, transmission type, and more. The dataset used is sourced from **CarDekho**, one of India's leading car marketplaces.

---

### 🧾 Dataset Description

The dataset contains information on around 6,000 used cars. Key features include:

| Column          | Description                               |
| --------------- | ----------------------------------------- |
| `Car_Name`      | Name of the car                           |
| `Year`          | Year when the car was purchased           |
| `Selling_Price` | Price the owner wants to sell the car for |
| `Present_Price` | Current ex-showroom price of the car      |
| `Kms_Driven`    | Kilometers driven                         |
| `Fuel_Type`     | Type of fuel used (Petrol, Diesel, CNG)   |
| `Seller_Type`   | Individual or Dealer                      |
| `Transmission`  | Manual or Automatic                       |
| `Owner`         | Number of previous owners                 |

---

### 🛠️ Project Pipeline

#### 1. **Data Cleaning**

* Removed duplicates and unnecessary columns
* Handled missing values (if any)
* Standardized text data

#### 2. **Feature Engineering**

* Created `Car_Age` = Current Year – `Year`
* Dropped irrelevant columns like `Car_Name`

#### 3. **Encoding & Scaling**

* Label encoded `Fuel_Type`, `Seller_Type`, and `Transmission`
* Scaled numerical features where required

#### 4. **Model Building**

* Algorithms tried:

  * Linear Regression
  * Random Forest Regressor
  * XGBoost Regressor
* Best performance from: **XGBoost Regressor**

#### 5. **Evaluation**

* Metrics: R² Score, MAE, RMSE
* R-squared (R2) training :  9.629300607094947 %
* R-squared (R2) testing :  9.562799077477946 %

#### 6. **Deployment (Optional)**

* Developed a simple web app using **Streamlit**
* User inputs car details and gets the predicted price

---

### 📊 Results

* High accuracy in price predictions
* Model generalizes well on unseen data
* Useful for sellers to set a fair price & buyers to negotiate better
