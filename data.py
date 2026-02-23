import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
st.set_page_config(page_title="Social Media Analytics", layout="wide")
st.title("📱 Social Media Performance Dashboard")

# 2. Sidebar - Data Loading
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("👋 Please upload the CSV file shown in your image to see the full analysis.")
    st.stop()

# 3. Data Processing
# Calculate Total Engagement (Likes + Comments + Shares)
df['total_engagement'] = df['likes'] + df['comments'] + df['shares']

# 4. Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Posts", len(df))
with col2:
    st.metric("Avg Engagement Rate", f"{df['engagement_rate'].mean():.2%}")
with col3:
    st.metric("Total Likes", f"{df['likes'].sum():,}")
with col4:
    st.metric("Max Views", f"{df['views'].max():,}")

# 5. Visualizations
st.subheader("📊 Performance Insights")
tab1, tab2, tab3 = st.tabs(["Platform Comparison", "Engagement Trends", "Content Length"])

with tab1:
    st.write("### Engagement by Platform & Post Type")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=df, x='platform', y='total_engagement', hue='post_type', ax=ax1)
    st.pyplot(fig1)

with tab2:
    st.write("### Likes vs. Comments Correlation")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.scatterplot(data=df, x='likes', y='comments', size='views', hue='platform', alpha=0.7, ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.write("### Impact of Post Length on Engagement")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.regplot(data=df, x='post_length', y='engagement_rate', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax3)
    st.pyplot(fig3)

# 6. AI Clustering (Grouping posts by performance)
st.divider()
st.subheader("🤖 AI Performance Clustering")
st.write("This groups your posts into categories based on their engagement metrics.")

# Select features for clustering
features = ['likes', 'comments', 'shares', 'views']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['performance_cluster'] = kmeans.fit_predict(scaled_features)

cluster_choice = st.selectbox("Select a Cluster to view posts:", options=[0, 1, 2])
st.dataframe(df[df['performance_cluster'] == cluster_choice], use_container_width=True)

# 7. Raw Data & Download
st.subheader("🏆 Data Explorer")
st.dataframe(df.sort_values(by='engagement_rate', ascending=False))

csv = df.to_csv(index=False).encode('utf-8')

st.download_button("📥 Download Analyzed Data", data=csv, file_name="social_media_analysis.csv")

