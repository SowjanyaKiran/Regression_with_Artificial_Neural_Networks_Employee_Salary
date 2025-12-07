# 💼 Regression with Artificial Neural Networks (ANN) – Employee Salary Prediction

This project uses Artificial Neural Networks (ANNs) to predict employee salaries based on demographic, educational, and experience-related features. The aim is to understand how neural networks handle mixed data (categorical + numerical) in real-world salary prediction scenarios.

# 📂 Dataset

The dataset used in this project is sourced from Kaggle and includes employee attributes such as:

Age

Gender

Education Level

Job Role

Experience

City

Target: Salary (annual income)

# 🚀 Project Workflow

1️⃣ Data Understanding

1. Loaded and explored dataset structure

2. Identified input features & target variable (Salary)

3. Checked data types, missing values, and duplicates

4. Investigated categorical vs numerical variables

5. Analyzed unique values in categories (Gender, Job Role, City, etc.)

2️⃣ Exploratory Data Analysis (EDA)

1. Distribution plots for Salary and key numeric features

Salary comparison across:

1. Gender

2. Education Levels

3. Job Roles

4. Experience Groups

5. Boxplots to identify salary outliers

6. Trend analysis: Salary vs Experience

7. Correlation heatmap for numerical features

8. Feature relationship insights using scatterplots & countplots

3️⃣ Data Preprocessing

1. Encoded multiple categorical variables using Label Encoding / One-Hot Encoding

2. Scaled numerical columns for ANN input

3. Created Experience Groups (if needed)

4. Split data into training and testing sets

5. Ensured balanced representation of categorical features

4️⃣ ANN Model Building (Baseline Model)

Constructed an ANN with:

1. Input layer

2. Hidden layer(s) with ReLU activation

3. Output neuron for continuous salary prediction

4. Trained baseline model using:

5. Loss: Mean Squared Error (MSE)

6. Optimizer: Adam

# Evaluated with:

1. MAE (Mean Absolute Error)

2. MSE (Mean Squared Error)

3. RMSE (Root Mean Squared Error)

4. R² Score

5️⃣ Model Optimization & Hyperparameter Tuning

Experimented with tuning:

1. Number of hidden layers

2. Neurons per layer

3. Learning rate adjustments

4. Batch size & epoch counts

5. Dropout layers to reduce overfitting

6. EarlyStopping for stable convergence

6️⃣ Model Evaluation

1. Compared training vs validation performance

2. Loss curve visualization

3. Actual vs Predicted Salary plot

4. Residual analysis to understand prediction errors

5. Highlighted patterns that influenced high/low salaries

# 📊 Results & Insights

1. ANN accurately predicted employee salaries after optimization

2. Encoding & feature scaling were essential for stable ANN convergence

3. Experience, Education Level & Job Role showed highest predictive strength

4. Dropout layers improved generalization and prevented overfitting

5. Residual distribution showed consistent predictions across salary ranges

# 📝 Deliverables

1. Jupyter Notebook:
Includes EDA, preprocessing, ANN modeling, optimization, evaluation

2. Short Write-Up:
Summarizes dataset insights, model performance, and challenges solved

# 🔑 Key Learnings

1. Handling mixed datasets with categorical & numerical features

2. Building and tuning deep regression models

3. Understanding salary determinants through ANN visualization

4. Using MAE, RMSE & R² to evaluate model performance

5. Applying real-world machine learning workflows

# 🛠️ Tech Stack

1. Python

2. TensorFlow / Keras

3. Pandas, NumPy

4. Matplotlib, Seaborn

5. Scikit-learn (preprocessing & evaluation)

👤 Author

Sowjanya U
Data Science & Machine Learning Enthusiast

📧 Email: usowjanyakiran@gmail.com

🌐 GitHub: https://github.com/SowjanyaKiran/
