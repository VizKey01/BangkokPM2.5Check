import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Bangkok PM2.5 Clustering Analysis", layout="wide")

# Title and description
st.title("Bangkok PM2.5 Clustering Analysis")
st.markdown("""
This application analyzes PM2.5 data from various air quality monitoring stations in Bangkok and its vicinity.
The analysis includes data preprocessing, clustering using K-means, and visualization of results.
""")

@st.cache_data
def load_data(file=None):
    """Load data from CSV file or use the uploaded file"""
    if file is None:
        # Use the example data
        df = pd.read_csv("1pm25.csv")
    else:
        # If a file is uploaded
        df = pd.read_csv(file)
    return df

# @st.cache_data
# def load_data(file=None):
#     """Load data from CSV file or use the uploaded file"""
#     if file is not None:
#         # If a file is uploaded
#         df = pd.read_csv(file)
#     else:
#         # Use the example data
#         df = pd.read_csv("1pm25.csv")
#     return df

# File uploader for custom data
uploaded_file = st.file_uploader("Upload your PM2.5 CSV file (or use the example data)", type=["csv"])

# Load data
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Example file not found. Please upload a CSV file.")
        st.stop()

# Display the raw data
st.subheader("Raw Data")
st.dataframe(df.head())

# Data Preprocessing
st.header("1. Data Preprocessing")

# Convert date and time columns if needed
st.subheader("Date and Time Processing")
if 'วันที่' in df.columns and 'ช่วงเวลา' in df.columns:
    st.write("Converting date and time columns...")
    # Create a datetime column by combining the date and time columns
    df['datetime'] = pd.to_datetime(df['วันที่'] + ' ' + df['ช่วงเวลา'], format='%m/%d/%Y %H:%M')
    st.write("Datetime column created successfully")
    
    # Display the dataframe with the new datetime column
    st.dataframe(df[['datetime', 'วันที่', 'ช่วงเวลา']].head())
else:
    st.warning("Date and time columns not found in the expected format")

# Select only numeric columns for clustering
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove the index column if it exists
if 'No.' in numeric_cols:
    numeric_cols.remove('No.')

st.subheader("Feature Selection")
st.write(f"Selected {len(numeric_cols)} numeric columns for clustering")

# Handle missing values
st.subheader("Handling Missing Values")
missing_values = df[numeric_cols].isna().sum()
st.write("Missing values per column:")
st.write(missing_values)

# Fill missing values with column means
df_clean = df[numeric_cols].fillna(df[numeric_cols].mean())
st.write("Missing values filled with column means")

# Standardize the data
st.subheader("Data Standardization")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)
st.write("Data standardized using StandardScaler")

# Show a sample of the standardized data
scaled_df = pd.DataFrame(scaled_data, columns=df_clean.columns)
st.dataframe(scaled_df.head())

# Elbow Method
st.header("2. Elbow Method")
st.write("Finding the optimal number of clusters using the Elbow Method")

# Calculate WCSS for different values of k
wcss = []
k_range = range(1, 11)

progress_bar = st.progress(0)
status_text = st.empty()

for i, k in enumerate(k_range):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    progress = (i + 1) / len(k_range)
    progress_bar.progress(progress)
    status_text.text(f"Computing for k={k}...")

status_text.text("Computation completed!")

# Plot the Elbow Method
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(k_range, wcss, 'bx-')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
st.pyplot(fig)

# Let user select the number of clusters
optimal_k = st.slider("Select number of clusters (k) based on the Elbow Method:", 2, 10, 3)

# Perform K-means clustering with the selected k
st.header("3. K-means Clustering Analysis")
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Centroid Analysis
st.subheader("Cluster Centroids")
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df_clean.columns)
st.write("Cluster centroids (values correspond to original scale):")
st.dataframe(centroids)

# Visualize the centroids
st.subheader("Centroid Graph")
st.write("Comparing the average PM2.5 values for each cluster across stations")

# We'll use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a dataframe for PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Cluster'] = clusters

# Plot PCA results colored by cluster
fig, ax = plt.subplots(figsize=(10, 6))
scatter = plt.scatter(pca_df['Principal Component 1'], 
                     pca_df['Principal Component 2'], 
                     c=pca_df['Cluster'], 
                     cmap='viridis', 
                     alpha=0.7,
                     s=50)
plt.title('Clusters Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')

# Add centroids to the plot
pca_centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=200, marker='X', c='red', label='Centroids')
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Additional centroid analysis - heatmap
st.subheader("Centroid Heatmap")
st.write("Heatmap showing the relative values of each feature across different clusters")

# Standardize the centroids for better visualization
centroid_z = pd.DataFrame(kmeans.cluster_centers_, columns=df_clean.columns)

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(centroid_z, cmap="YlGnBu", linewidths=0.5, annot=False)
plt.title('Standardized Cluster Centroids Heatmap')
plt.xticks(rotation=90)
st.pyplot(fig)

# Cluster Distribution
st.subheader("Cluster Distribution")
cluster_counts = df['Cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)

# Time analysis if datetime is available
if 'datetime' in df.columns:
    st.subheader("Time Analysis of Clusters")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        if not cluster_data.empty and 'datetime' in cluster_data.columns:
            plt.plot(cluster_data['datetime'], cluster_data[numeric_cols[0]], label=f'Cluster {cluster}')
    
    plt.title(f'Time Series of {numeric_cols[0]} by Cluster')
    plt.xlabel('Date and Time')
    plt.ylabel(f'{numeric_cols[0]}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Summary and Insights
st.header("4. Summary and Insights")

# Calculate the average PM2.5 values for each cluster
cluster_means = df.groupby('Cluster')[numeric_cols].mean()
st.write("Average PM2.5 values for each cluster:")
st.dataframe(cluster_means)

# Get the range of values for interpretation
pm25_min = df[numeric_cols].min().min()
pm25_max = df[numeric_cols].max().max()
pm25_mean = df[numeric_cols].mean().mean()

st.write(f"Overall PM2.5 value range: {pm25_min:.2f} to {pm25_max:.2f}, with an average of {pm25_mean:.2f}")

# Provide interpretations based on the WHO guidelines
st.subheader("Interpretation based on WHO Guidelines")
st.write("""
The World Health Organization (WHO) guidelines for PM2.5 are:
- 0-12 μg/m³: Good
- 12.1-35.4 μg/m³: Moderate
- 35.5-55.4 μg/m³: Unhealthy for Sensitive Groups
- 55.5-150.4 μg/m³: Unhealthy
- 150.5-250.4 μg/m³: Very Unhealthy
- 250.5+ μg/m³: Hazardous
""")

# Automatically generate insights based on the cluster centroids
st.subheader("Automated Insights")

insights = []
for i in range(optimal_k):
    cluster_avg = cluster_means.loc[i].mean()
    if cluster_avg < 12:
        quality = "Good"
    elif cluster_avg < 35.5:
        quality = "Moderate"
    elif cluster_avg < 55.5:
        quality = "Unhealthy for Sensitive Groups"
    elif cluster_avg < 150.5:
        quality = "Unhealthy"
    elif cluster_avg < 250.5:
        quality = "Very Unhealthy"
    else:
        quality = "Hazardous"
        
    # Find highest stations in this cluster
    high_stations = centroids.loc[i].nlargest(3).index.tolist()
    high_values = centroids.loc[i].nlargest(3).values
    
    # Find lowest stations in this cluster
    low_stations = centroids.loc[i].nsmallest(3).index.tolist()
    low_values = centroids.loc[i].nsmallest(3).values
    
    # Count observations in this cluster
    count = (df['Cluster'] == i).sum()
    percentage = count / len(df) * 100
    
    insight = f"""
    **Cluster {i}** (containing {count} observations, {percentage:.1f}% of the data):
    - Average PM2.5 value: {cluster_avg:.2f} μg/m³ ({quality})
    - Highest PM2.5 stations: {', '.join([f"{station} ({value:.2f})" for station, value in zip(high_stations, high_values)])}
    - Lowest PM2.5 stations: {', '.join([f"{station} ({value:.2f})" for station, value in zip(low_stations, low_values)])}
    """
    insights.append(insight)

for insight in insights:
    st.markdown(insight)

# Overall conclusions
st.subheader("Conclusions")
st.markdown("""
Based on the clustering analysis, we can draw the following conclusions:

1. **Spatial Patterns**: The clustering reveals spatial patterns in PM2.5 concentration across Bangkok and its vicinity.
2. **Temporal Patterns**: Different clusters show different temporal patterns, suggesting varying sources of pollution.
3. **Critical Areas**: Some stations consistently show higher PM2.5 values, which may require targeted interventions.
4. **Air Quality Management**: The clustering can help in designing targeted air quality management strategies for different areas.
5. **Health Implications**: Areas with higher cluster averages may pose greater health risks to sensitive populations.

This analysis can be useful for:
- Environmental agencies to design targeted interventions
- Urban planners to consider air quality in development decisions
- Public health officials to issue localized advisories
- Researchers studying pollution patterns and their causes
""")

# Download section
st.header("Download Results")
    
# Create a CSV with the clusters
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df)
st.download_button(
    label="Download Data with Cluster Labels",
    data=csv,
    file_name='bangkok_pm25_with_clusters.csv',
    mime='text/csv',
)

# Create a buffer for the plots
buffer = io.BytesIO()
fig.savefig(buffer, format='png')
buffer.seek(0)

st.download_button(
    label="Download PCA Plot",
    data=buffer,
    file_name=f'bangkok_pm25_pca_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
    mime='image/png',
)

st.markdown("---")
st.markdown("Developed for Bangkok PM2.5 Analysis Project")