import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
data = pd.read_csv('student-mat.csv', delimiter=';')
print("First few rows of the dataset:")
print(data.head())

print("\nDataset size:")
print(data.shape)

print("\nColumns in the dataset:")
print(data.columns)


# Data Exploration
print("\nChecking for missing values:")
print(data.isnull().sum())

print("\nColumn data types:")
print(data.dtypes)

print("\nDataset size:")
print(data.shape)

# Data Cleaning
# Handling missing values
if data.isnull().sum().sum() > 0:
    data.fillna(data.median(), inplace=True)

# Removing duplicates
data.drop_duplicates(inplace=True)

# Data Analysis Questions
# 1. Average score in math (G3)
average_g3 = data['G3'].mean()
print(f"\nAverage final grade (G3): {average_g3:.2f}")

# 2. Number of students scoring above 15 in final grade (G3)
students_above_15 = (data['G3'] > 15).sum()
print(f"Number of students scoring above 15 in final grade (G3): {students_above_15}")

# 3. Correlation between study time and final grade (G3)
correlation = data['studytime'].corr(data['G3'])
print(f"Correlation between study time and final grade (G3): {correlation:.2f}")

# 4. Gender with higher average final grade (G3)
average_g3_by_gender = data.groupby('sex')['G3'].mean()
print("\nAverage final grade (G3) by gender:")
print(average_g3_by_gender)

# Data Visualization
# 1. Histogram of final grades (G3)
plt.figure(figsize=(8, 5))
plt.hist(data['G3'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.show()

# 2. Scatter plot between study time and final grade (G3)
plt.figure(figsize=(8, 5))
plt.scatter(data['studytime'], data['G3'], alpha=0.7, color='purple')
plt.title('Study Time vs Final Grade')
plt.xlabel('Study Time (hours/week)')
plt.ylabel('Final Grade (G3)')
plt.show()

# 3. Bar chart comparing average scores of male and female students
plt.figure(figsize=(8, 5))
average_g3_by_gender.plot(kind='bar', color=['blue', 'pink'], edgecolor='black')
plt.title('Average Final Grades by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Final Grade (G3)')
plt.xticks(rotation=0)
plt.show()

# Add Markdown explanations and insights when running this in Jupyter Notebook.
