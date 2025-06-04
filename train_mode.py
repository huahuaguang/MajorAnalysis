import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from joblib import dump

# 1. Load the preprocessed data
df = pd.read_csv('./datasets/computer_related_text_dataset.csv')

# 2. Text vectorization, to make text vector and get some features
tfidf = TfidfVectorizer(
    max_features=15000,      # Limit the number of features
    stop_words='english',   # Remove English stop words to reduce noise, like the, is, in
    ngram_range=(1, 2),     # Use both unigrams and bigrams(single word and two words)
    min_df=5,               # Ignore words that appear less than 5 times in the docx
    max_df=0.7              # Ignore words that appear in more than 70% of documents
)

# 'text' means the text col
X = tfidf.fit_transform(df['text'])

# 3. Dimensionality reduction for visualization
# Project data in the direction with the largest variance, removing redundant information.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['is_computer_related'], alpha=0.6)
plt.title("PCA Visualization of Text Data (Colored by True Labels)")
plt.colorbar()
plt.show()

# 4. K-Means clustering
k = 2  # divide data into two classes: computer-related and non-computer-related
kmeans = KMeans(
    n_clusters=k,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)

# The centroid is calculated, clustered, and the cluster label to which each data point belongs is returned.
clusters = kmeans.fit_predict(X)

# 5. Evaluate clustering results
# Compare with true labels
ari = adjusted_rand_score(df['is_computer_related'], clusters)
print(f"Adjusted Rand Index (Consistency with true labels): {ari:.3f}")

# Silhouette score
silhouette = silhouette_score(X, clusters)
print(f"Silhouette Score (Clustering quality): {silhouette:.3f}")

# 6. Visualize clustering results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.6)
plt.title("PCA Visualization with K-Means Clustering Results")
plt.colorbar()
plt.show()

# 7. Analyze clustering results
# Add clustering results to the original data
df['cluster'] = clusters

# Examine the composition of each cluster
cluster_summary = df.groupby('cluster')['is_computer_related'].value_counts(normalize=True)
print("\nCluster Results Summary:")
print(cluster_summary.unstack())

# 8. Examine the keywords for each cluster
print("\nKeywords for each cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()
for i in range(k):
    print(f"\nKeywords for Cluster {i}:")
    for ind in order_centroids[i, :15]:  # Examine the top 15 keywords
        print(f" {terms[ind]}", end='')
    print()

# 9. Save the data with clustering results
# df.to_csv('clustered_text_data.csv', index=False)

# Save the trained model and vectorizer
dump(kmeans, './models/kmeans_model.pkl')       # Save the model object
dump(tfidf, './models/tfidf_vectorizer.pkl')   # Save the vectorizer