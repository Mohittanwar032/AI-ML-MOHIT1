# Titanic Data Cleaning & Preprocessing Script (No Arguments)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('titanic.csv')

# Step 1: Show basic information
print("First 5 rows:\n", df.head())
print("\nData Types and Null Values:\n")
print(df.info())
print("\nNull values count:\n", df.isnull().sum())

# Step 2: Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)              # Numerical: Median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)   # Categorical: Mode
df.drop('Cabin', axis=1, inplace=True)                          # Too many missing values

# Step 3: Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})             # Label Encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True) # One-Hot Encoding

# Step 4: Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 5: Remove outliers using IQR for 'Fare'
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# Optional: Visualize outliers
sns.boxplot(data=df[['Age', 'Fare']])
plt.title('Boxplots After Outlier Removal')
plt.show()

# Save cleaned data
df.to_csv('cleaned_titanic.csv', index=False)
print("\n✅ Preprocessing complete. Cleaned file saved as 'cleaned_titanic.csv'")
