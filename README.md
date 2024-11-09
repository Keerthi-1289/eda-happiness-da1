# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
data_path = '/mnt/data/happiness.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
print("Dataset Overview:")
print(data.head())

# Basic information and summary of the dataset
print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Step 1: Univariate Analysis
print("\nUnivariate Analysis:")
for column in data.select_dtypes(include=[np.number]).columns:
    sns.histplot(data[column], kde=True)
    plt.title(f'Univariate Analysis of {column}')
    plt.show()

# Step 2: Bivariate Analysis
print("\nBivariate Analysis:")
numeric_columns = data.select_dtypes(include=[np.number]).columns
sns.pairplot(data[numeric_columns])
plt.suptitle('Bivariate Pair Plot', y=1.02)
plt.show()

# Step 3: Principal Component Analysis (PCA)
# Standardize the data before PCA
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_columns])

# Applying PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Adding PCA results to the original data for visualization
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

# Plotting the PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', data=data)
plt.title('2D PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Step 4: N-dimensional Analysis (using all available components)
# Choosing the number of components based on the variance explained
pca_full = PCA().fit(data_scaled)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.title('Explained Variance by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

print("\nExplained Variance by Components:")
for i, var in enumerate(explained_variance, start=1):
    print(f'Component {i}: {var:.2f}')
