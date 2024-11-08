import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


data = pd.read_csv('happiness.csv')

# 1. Data Cleaning (Handle missing values)
data_cleaned = data.copy()
numerical_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns

# Fill missing numerical values with median and categorical with 'unknown'
data_cleaned[numerical_cols] = data_cleaned[numerical_cols].fillna(data_cleaned[numerical_cols].median())
data_cleaned[categorical_cols] = data_cleaned[categorical_cols].fillna('unknown')

# 2. Univariate Analysis (Summary statistics)
univariate_numeric_summary = data_cleaned.describe()
univariate_categorical_summary = {col: data_cleaned[col].value_counts() for col in categorical_cols}

# 3. Bivariate Analysis (Correlation and Cross-tabulation)
correlation_matrix = data_cleaned.corr()
cross_tab_workstat_happy = pd.crosstab(data_cleaned['workstat'], data_cleaned['happy'])
cross_tab_income_happy = pd.crosstab(data_cleaned['income'], data_cleaned['happy'])

# 4. Multivariate Analysis: Dimensionality Reduction using PCA
scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(data_cleaned[numerical_cols])

# Perform PCA (reduce to 2 dimensions for visualization)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(numerical_data_scaled)
data_cleaned['PCA1'] = pca_components[:, 0]
data_cleaned['PCA2'] = pca_components[:, 1]
pca_explained_variance = pca.explained_variance_ratio_

# 5. Clustering using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(numerical_data_scaled)
data_cleaned['Cluster'] = clusters

# Output
print("Univariate Numeric Summary:", univariate_numeric_summary)
print("Correlation Matrix:", correlation_matrix)
print("Work Status vs Happiness Crosstab:\n", cross_tab_workstat_happy)
print("Income vs Happiness Crosstab:\n", cross_tab_income_happy)
print("PCA Explained Variance Ratio:", pca_explained_variance)
print("Head of Data with PCA and Clusters:\n", data_cleaned[['PCA1', 'PCA2', 'Cluster']].head())
