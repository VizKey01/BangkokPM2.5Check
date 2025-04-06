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
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Bangkok PM2.5 Analysis",
    page_icon="🌇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        padding: 0.1rem;
        border-radius: 0.5rem;
        background-color: #1E88E5;
        margin-bottom: 2rem;
        border-left: 1px solid #1E88E5;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .warning-box {
        # padding: 1rem;
        # border-radius: 0.5rem;
        # background-color: #fff3e0;
        # margin-bottom: 1rem;
        # border-left: 4px solid #FF9800;
    }
    .success-box {
        # padding: 1rem;
        # border-radius: 0.5rem;
        # background-color: #e8f5e9;
        # margin-bottom: 1rem;
        # border-left: 4px solid #4CAF50;
    }
    .stPlotlyChart {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">Bangkok PM2.5 Clustering Analysis</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://github.com/VizKey01/BangkokPM2.5Check/blob/main/app/data/istockphoto-1503181890-1024x1024.jpg?raw=true", width=150)
    # st.image("data/istockphoto-1503181890-1024x1024.jpg", width=150)
    st.markdown("## Navigation")
    st.markdown("- [Data Overview](#data-overview)")
    st.markdown("- [Data Preprocessing](#data-preprocessing)")
    st.markdown("- [Elbow Method](#elbow-method)")
    st.markdown("- [Clustering Analysis](#clustering-analysis)")
    st.markdown("- [Summary and Insights](#summary-and-insights)")
    
    st.markdown("---")
    st.markdown("## About")
    st.markdown("""
    This application analyzes PM2.5 data from various air quality monitoring stations in Bangkok and its vicinity,
    using K-means clustering to identify patterns in air quality data.
    """)
    
    st.markdown("---")
    st.markdown("### Data Source")
    st.markdown("Data from air quality monitoring stations in Bangkok, Thailand")
# Main content
st.markdown('<div class="info-box">'
'   <p>This application analyzes PM2.5 data from various air quality monitoring stations in Bangkok and its vicinity.<br>The analysis includes data preprocessing, clustering using K-means, and visualization of results to identify patterns in air quality across different locations.'
'   </p></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data from CSV file in data directory"""
    file_path = "https://raw.githubusercontent.com/VizKey01/BangkokPM2.5Check/main/app/data/pm25.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please make sure the file exists.")
        return None

# Load data
df = load_data()
if df is None:
    st.stop()

# Display the raw data
st.markdown('<h2 class="sub-header" id="data-overview">Data Overview</h2>', unsafe_allow_html=True)
# st.markdown('<div class="section">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Total Records:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

with col2:
    # Find datetime columns if they exist
    date_cols = [col for col in df.columns if 'วันที่' in col or 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        date_range = f"{df[date_col].min()} to {df[date_col].max()}"
        st.write(f"**Date Range:** {date_range}")
    st.write(f"**Monitoring Stations:** {len([col for col in df.columns if 't' in col])}")

tab1, tab2 = st.tabs(["Data Preview", "Data Stats"])
with tab1:
    st.dataframe(df.head(10), use_container_width=True)
with tab2:
    st.dataframe(df.describe(), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Data Preprocessing
st.markdown('<h2 class="sub-header" id="data-preprocessing">Data Preprocessing</h2>', unsafe_allow_html=True)
# st.markdown('<div class="section">', unsafe_allow_html=True)

# Convert date and time columns if needed
if 'วันที่' in df.columns and 'ช่วงเวลา' in df.columns:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.write("**Date and Time Processing**")
    # Create a datetime column by combining the date and time columns
    df['datetime'] = pd.to_datetime(df['วันที่'] + ' ' + df['ช่วงเวลา'], format='%m/%d/%Y %H:%M')
    st.write("✅ Datetime column created successfully")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display the dataframe with the new datetime column
    st.dataframe(df[['datetime', 'วันที่', 'ช่วงเวลา']].head(), use_container_width=True)
else:
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.write("⚠️ Date and time columns not found in the expected format")
    st.markdown('</div>', unsafe_allow_html=True)

# Select only numeric columns for clustering
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove the index column if it exists
if 'No.' in numeric_cols:
    numeric_cols.remove('No.')

# st.markdown('<div class="info-box">', fe_allow_htunsaml=True)
st.write("**Feature Selection**")
st.write(f"✅ Selected {len(numeric_cols)} numeric columns for clustering analysis")
# st.markdown('</div>', unsafe_allow_html=True)

# Handle missing values
missing_values = df[numeric_cols].isna().sum()
if missing_values.sum() > 0:
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.write("**Handling Missing Values**")
    st.write(f"Found {missing_values.sum()} missing values across {sum(missing_values > 0)} columns")
    
    # Show columns with missing values
    st.dataframe(missing_values[missing_values > 0], use_container_width=True)
    
    # Fill missing values with column means
    df_clean = df[numeric_cols].fillna(df[numeric_cols].mean())
    st.write("✅ Missing values filled with column means")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.write("**Checking for Missing Values**")
    st.write("✅ No missing values found in the numeric columns")
    st.markdown('</div>', unsafe_allow_html=True)
    df_clean = df[numeric_cols]

# Standardize the data
# st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.write("**Data Standardization**")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)
st.write("✅ Data standardized using StandardScaler")
# st.markdown('</div>', unsafe_allow_html=True)

# Show a sample of the standardized data
with st.expander("View Standardized Data Sample"):
    scaled_df = pd.DataFrame(scaled_data, columns=df_clean.columns)
    st.dataframe(scaled_df.head(), use_container_width=True)

# Distribution of a sample column
if len(numeric_cols) > 0:
    selected_col = st.selectbox("Select a column to view distribution", numeric_cols)
    
    fig = px.histogram(
        df, 
        x=selected_col, 
        title=f"Distribution of {selected_col}",
        nbins=30,
        color_discrete_sequence=['#1E88E5']
    )
    fig.update_layout(
        xaxis_title=selected_col,
        yaxis_title="Frequency",
        plot_bgcolor='white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Elbow Method
st.markdown('<h2 class="sub-header" id="elbow-method">Elbow Method</h2>', unsafe_allow_html=True)
# st.markdown('<div class="section">', unsafe_allow_html=True)

st.write("Finding the optimal number of clusters using the Elbow Method")

# Use a collapsible expander to show the WCSS calculation
with st.expander("Show WCSS Calculation Details"):
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

    status_text.text("✅ Computation completed!")
    
    # Show the WCSS values in a table
    wcss_df = pd.DataFrame({'Number of Clusters (k)': k_range, 'WCSS': wcss})
    st.dataframe(wcss_df, use_container_width=True)

# Plot the Elbow Method using Plotly for interactive visualization
fig = px.line(
    x=list(k_range), 
    y=wcss,
    markers=True,
    labels={'x': 'Number of clusters (k)', 'y': 'Within-Cluster Sum of Squares (WCSS)'},
    title='Elbow Method For Optimal k'
)
fig.update_traces(marker=dict(size=10))
fig.update_layout(
    xaxis=dict(tickmode='linear', dtick=1),
    plot_bgcolor='white',
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Let user select the number of clusters
col1, col2 = st.columns([3, 1])
with col1:
    optimal_k = st.slider("Select number of clusters (k) based on the Elbow Method:", 2, 10, 3)
with col2:
    st.markdown('<div class="info-box" style="margin-top: 30px;">', unsafe_allow_html=True)
    st.write(f"**Selected: {optimal_k} clusters**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Perform K-means clustering with the selected k
st.markdown('<h2 class="sub-header" id="clustering-analysis">K-means Clustering Analysis</h2>', unsafe_allow_html=True)
# st.markdown('<div class="section">', unsafe_allow_html=True)

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Centroid Analysis
st.markdown("### Cluster Centroids")
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df_clean.columns)
st.write("Cluster centroids (values correspond to original scale):")

# Display centroids in a more interactive way
with st.expander("View Complete Centroid Table"):
    st.dataframe(centroids, use_container_width=True)

# More informative centroid visualization
selected_features = st.multiselect(
    "Select features to compare across clusters:",
    options=df_clean.columns.tolist(),
    default=df_clean.columns.tolist()[:5] if len(df_clean.columns) > 5 else df_clean.columns.tolist()
)

if selected_features:
    # Create a radar chart for the selected features
    fig = go.Figure()
    
    for i in range(optimal_k):
        fig.add_trace(go.Scatterpolar(
            r=centroids.loc[i, selected_features].values,
            theta=selected_features,
            fill='toself',
            name=f'Cluster {i}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )),
        showlegend=True,
        title="Radar Chart of Cluster Centroids",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Visualize the centroids with PCA
st.markdown("### PCA Visualization")
st.write("Using Principal Component Analysis (PCA) to visualize the clusters in 2D space")

# We'll use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a dataframe for PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Cluster'] = clusters

# If we have datetime, add it for hover info
if 'datetime' in df.columns:
    pca_df['datetime'] = df['datetime']

# Plot PCA results with Plotly for better interactivity
fig = px.scatter(
    pca_df, 
    x='Principal Component 1', 
    y='Principal Component 2', 
    color='Cluster',
    color_continuous_scale='viridis',
    hover_data=['Cluster'] + (['datetime'] if 'datetime' in pca_df.columns else []),
    opacity=0.7,
    title='Clusters Visualization using PCA'
)

# Add centroids to the plot
pca_centroids = pca.transform(kmeans.cluster_centers_)
for i, centroid in enumerate(pca_centroids):
    fig.add_trace(
        go.Scatter(
            x=[centroid[0]],
            y=[centroid[1]],
            mode='markers',
            marker=dict(
                color='red',
                size=15,
                symbol='x'
            ),
            name=f'Centroid {i}'
        )
    )

fig.update_layout(
    height=600,
    plot_bgcolor='white'
)
st.plotly_chart(fig, use_container_width=True)

# Additional centroid analysis - heatmap
st.markdown("### Centroid Heatmap")
st.write("Heatmap showing the relative values of each feature across different clusters")

# Standardize the centroids for better visualization
centroid_z = pd.DataFrame(kmeans.cluster_centers_, columns=df_clean.columns)

# Create a heatmap with Plotly
fig = px.imshow(
    centroid_z,
    labels=dict(x="Features", y="Cluster", color="Value"),
    x=centroid_z.columns,
    y=[f"Cluster {i}" for i in range(optimal_k)],
    color_continuous_scale="YlGnBu",
    aspect="auto"
)
fig.update_layout(
    height=600,
    xaxis=dict(tickangle=90)
)
st.plotly_chart(fig, use_container_width=True)

# Cluster Distribution
st.markdown("### Cluster Distribution")
cluster_counts = df['Cluster'].value_counts().sort_index()
cluster_df = pd.DataFrame({
    'Cluster': cluster_counts.index,
    'Count': cluster_counts.values,
    'Percentage': (cluster_counts.values / len(df) * 100).round(1)
})

col1, col2 = st.columns([2, 1])
with col1:
    fig = px.bar(
        cluster_df,
        x='Cluster',
        y='Count',
        text='Percentage',
        labels={'Count': 'Number of Observations', 'Cluster': 'Cluster Label'},
        title='Distribution of Observations Across Clusters',
        color='Cluster',
        color_continuous_scale='viridis'
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.dataframe(cluster_df, use_container_width=True)

# Time analysis if datetime is available
if 'datetime' in df.columns:
    st.markdown("### Time Analysis of Clusters")
    
    # Select a feature to plot over time
    time_feature = st.selectbox(
        "Select a feature to visualize over time:", 
        options=numeric_cols,
        index=0
    )
    
    # Create a time series plot for each cluster
    fig = go.Figure()
    
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        if not cluster_data.empty and 'datetime' in cluster_data.columns:
            # Sort by datetime for proper line plotting
            cluster_data = cluster_data.sort_values('datetime')
            
            fig.add_trace(go.Scatter(
                x=cluster_data['datetime'],
                y=cluster_data[time_feature],
                mode='lines+markers',
                name=f'Cluster {cluster}',
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=f'Time Series of {time_feature} by Cluster',
        xaxis_title='Date and Time',
        yaxis_title=time_feature,
        legend_title='Cluster',
        height=500,
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Summary and Insights
st.markdown('<h2 class="sub-header" id="summary-and-insights">Summary and Insights</h2>', unsafe_allow_html=True)
# st.markdown('<div class="section">', unsafe_allow_html=True)

# Calculate the average PM2.5 values for each cluster
cluster_means = df.groupby('Cluster')[numeric_cols].mean()
st.write("**Average PM2.5 values for each cluster:**")
st.dataframe(cluster_means, use_container_width=True)

# Get the range of values for interpretation
pm25_min = df[numeric_cols].min().min()
pm25_max = df[numeric_cols].max().max()
pm25_mean = df[numeric_cols].mean().mean()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Minimum PM2.5", f"{pm25_min:.2f} μg/m³")
with col2:
    st.metric("Average PM2.5", f"{pm25_mean:.2f} μg/m³")
with col3:
    st.metric("Maximum PM2.5", f"{pm25_max:.2f} μg/m³")

# Provide interpretations based on the WHO guidelines
st.markdown("### Interpretation based on WHO Guidelines")

# Create a colored table for WHO guidelines
who_data = [
    {"Range": "0-12 μg/m³", "Quality": "Good", "Color": "#4CAF50"},
    {"Range": "12.1-35.4 μg/m³", "Quality": "Moderate", "Color": "#FFEB3B"},
    {"Range": "35.5-55.4 μg/m³", "Quality": "Unhealthy for Sensitive Groups", "Color": "#FF9800"},
    {"Range": "55.5-150.4 μg/m³", "Quality": "Unhealthy", "Color": "#F44336"},
    {"Range": "150.5-250.4 μg/m³", "Quality": "Very Unhealthy", "Color": "#9C27B0"},
    {"Range": "250.5+ μg/m³", "Quality": "Hazardous", "Color": "#795548"}
]

col1, col2 = st.columns([1, 2])
with col1:
    for item in who_data:
        st.markdown(
            f'<div style="padding:5px; background-color:{item["Color"]}; color:{"white" if item["Color"] not in ["#FFEB3B", "#4CAF50"] else "black"}; border-radius:5px; margin-bottom:5px;">'
            f'<strong>{item["Quality"]}</strong>: {item["Range"]}'
            f'</div>',
            unsafe_allow_html=True
        )
with col2:
    # Create a gauge chart for each cluster's average value
    fig = make_subplots(
        rows=optimal_k,
        cols=1,
        subplot_titles=[f"Cluster {i} Air Quality" for i in range(optimal_k)],
        specs=[[{"type": "indicator"}] for _ in range(optimal_k)],
        vertical_spacing=0.1
    )
    
    for i in range(optimal_k):
        cluster_avg = cluster_means.loc[i].mean()
        
        # Determine color based on value
        if cluster_avg < 12:
            color = "#4CAF50"  # Good
        elif cluster_avg < 35.5:
            color = "#FFEB3B"  # Moderate
        elif cluster_avg < 55.5:
            color = "#FF9800"  # Unhealthy for Sensitive Groups
        elif cluster_avg < 150.5:
            color = "#F44336"  # Unhealthy
        elif cluster_avg < 250.5:
            color = "#9C27B0"  # Very Unhealthy
        else:
            color = "#795548"  # Hazardous
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cluster_avg,
                gauge={
                    'axis': {'range': [None, 300], 'tickwidth': 1},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 12], 'color': "#E8F5E9"},
                        {'range': [12, 35.5], 'color': "#FFF9C4"},
                        {'range': [35.5, 55.5], 'color': "#FFE0B2"},
                        {'range': [55.5, 150.5], 'color': "#FFCDD2"},
                        {'range': [150.5, 250.5], 'color': "#E1BEE7"},
                        {'range': [250.5, 300], 'color': "#D7CCC8"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': cluster_avg
                    }
                },
                number={'suffix': " μg/m³", 'font': {'size': 20}},
                title={'text': f"Cluster {i}", 'font': {'size': 24}}
            ),
            row=i+1,
            col=1
        )
    
    height_per_gauge = 200
    fig.update_layout(height=height_per_gauge * optimal_k, margin=dict(t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Automatically generate insights based on the cluster centroids
st.markdown("### Automated Insights")

for i in range(optimal_k):
    cluster_avg = cluster_means.loc[i].mean()
    if cluster_avg < 12:
        quality = "Good"
        color = "#4CAF50"
    elif cluster_avg < 35.5:
        quality = "Moderate"
        color = "#FFEB3B"
    elif cluster_avg < 55.5:
        quality = "Unhealthy for Sensitive Groups"
        color = "#FF9800"
    elif cluster_avg < 150.5:
        quality = "Unhealthy"
        color = "#F44336"
    elif cluster_avg < 250.5:
        quality = "Very Unhealthy"
        color = "#9C27B0"
    else:
        quality = "Hazardous"
        color = "#795548"
        
    # Find highest stations in this cluster
    high_stations = centroids.loc[i].nlargest(3).index.tolist()
    high_values = centroids.loc[i].nlargest(3).values
    
    # Find lowest stations in this cluster
    low_stations = centroids.loc[i].nsmallest(3).index.tolist()
    low_values = centroids.loc[i].nsmallest(3).values
    
    # Count observations in this cluster
    count = (df['Cluster'] == i).sum()
    percentage = count / len(df) * 100
    
    st.markdown(
        f'<div style="padding:15px; border-radius:5px; margin-bottom:15px; border-left:5px solid {color}; background-color:white;">'
        f'<h3 style="color:{color};">Cluster {i}</h3>'
        f'<p><strong>Observations:</strong> {count} ({percentage:.1f}% of the data)</p>'
        f'<p><strong>Average PM2.5 value:</strong> {cluster_avg:.2f} μg/m³ <span style="background-color:{color}; padding:3px 8px; border-radius:3px; color:{"white" if color not in ["#FFEB3B", "#4CAF50"] else "black"}">{quality}</span></p>'
        f'<p><strong>Highest PM2.5 stations:</strong> {", ".join([f"{station} ({value:.2f})" for station, value in zip(high_stations, high_values)])}</p>'
        f'<p><strong>Lowest PM2.5 stations:</strong> {", ".join([f"{station} ({value:.2f})" for station, value in zip(low_stations, low_values)])}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

# Overall conclusions
st.markdown("### Conclusions")
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

st.markdown('</div>', unsafe_allow_html=True)

# Download section
st.markdown('<h2 class="sub-header">Download Results</h2>', unsafe_allow_html=True)
# st.markdown('<div class="section">', unsafe_allow_html=True)
    
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

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Developed for Bangkok PM2.5 Analysis Project | © 2025</p>",
    unsafe_allow_html=True
)