import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("STAGE 2C: INGREDIENT-BASED MEAL CLUSTERING")
print("="*70)

# ============================================================
# LOAD DATA
# ============================================================

print("\nSTEP 1: LOADING MEAL INGREDIENTS DATA")

meal_ingredients_df = pd.read_csv('data_meal_ingredients.csv')

print(f"✓ Loaded {len(meal_ingredients_df)} meals")

# ============================================================
# BUILD TF-IDF VECTORS
# ============================================================

print("\nSTEP 2: BUILDING TF-IDF INGREDIENT VECTORS")

# Create ingredient documents
food_names = []
ingredient_documents = []

meal_to_ingredients = {}
meal_to_cuisine = {}

for _, row in meal_ingredients_df.iterrows():
    meal_name = row['full_name'].lower().strip()
    ingredients = row['ingredients'].split('|')
    ingredients = [ing.lower().strip() for ing in ingredients]
    
    food_names.append(meal_name)
    ingredient_documents.append(' '.join(ingredients))
    
    meal_to_ingredients[meal_name] = set(ingredients)
    meal_to_cuisine[meal_name] = row['cuisine']

print(f"✓ Created {len(ingredient_documents)} ingredient documents")

# Build TF-IDF matrix
tfidf = TfidfVectorizer(
    min_df=2,  # Ignore ingredients that appear in < 2 meals
    max_df=0.8,  # Ignore ingredients that appear in > 80% of meals
    ngram_range=(1, 1)  # Unigrams only
)

tfidf_matrix = tfidf.fit_transform(ingredient_documents)

print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"✓ Vocabulary size: {len(tfidf.vocabulary_)}")

# ============================================================
# DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================

print("\nSTEP 3: DETERMINING OPTIMAL NUMBER OF CLUSTERS")

# Try different numbers of clusters
cluster_range = range(10, 51, 5)  # Test 10, 15, 20, ..., 50 clusters
silhouette_scores = []

print(f"Testing cluster counts: {list(cluster_range)}")

for n_clusters in cluster_range:
    print(f"  Testing n_clusters={n_clusters}...", end='')
    
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    
    # Fit on dense matrix (AgglomerativeClustering doesn't support sparse)
    cluster_labels = clusterer.fit_predict(tfidf_matrix.toarray())
    
    # Calculate silhouette score
    score = silhouette_score(tfidf_matrix, cluster_labels, metric='cosine')
    silhouette_scores.append(score)
    
    print(f" Silhouette: {score:.3f}")

# Find optimal number of clusters
optimal_idx = np.argmax(silhouette_scores)
optimal_n_clusters = list(cluster_range)[optimal_idx]
optimal_score = silhouette_scores[optimal_idx]

print(f"\n✓ Optimal number of clusters: {optimal_n_clusters}")
print(f"✓ Best silhouette score: {optimal_score:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(list(cluster_range), silhouette_scores, marker='o', linewidth=2, markersize=8)
plt.axvline(optimal_n_clusters, color='red', linestyle='--', 
            label=f'Optimal: {optimal_n_clusters} clusters')
plt.xlabel('Number of Clusters', fontsize=12, fontweight='bold')
plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
plt.title('Elbow Method: Optimal Cluster Count', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('meal_clustering_elbow.png', dpi=300, bbox_inches='tight')
print("✓ Saved: meal_clustering_elbow.png")
plt.close()

# ============================================================
# PERFORM FINAL CLUSTERING
# ============================================================

print(f"\nSTEP 4: CLUSTERING WITH {optimal_n_clusters} CLUSTERS")

final_clusterer = AgglomerativeClustering(
    n_clusters=optimal_n_clusters,
    metric='cosine',
    linkage='average'
)

cluster_labels = final_clusterer.fit_predict(tfidf_matrix.toarray())

print(f"✓ Assigned {len(cluster_labels)} meals to {optimal_n_clusters} clusters")

# ============================================================
# ANALYZE CLUSTERS
# ============================================================

print("\nSTEP 5: ANALYZING CLUSTER QUALITY")

# Add cluster labels to dataframe
clustering_results = pd.DataFrame({
    'meal_name': food_names,
    'cluster': cluster_labels,
    'cuisine': [meal_to_cuisine[name] for name in food_names]
})

# Cluster size distribution
cluster_sizes = clustering_results['cluster'].value_counts().sort_index()
print(f"\nCluster size statistics:")
print(f"  Mean: {cluster_sizes.mean():.1f} meals/cluster")
print(f"  Median: {cluster_sizes.median():.1f} meals/cluster")
print(f"  Min: {cluster_sizes.min()} meals/cluster")
print(f"  Max: {cluster_sizes.max()} meals/cluster")

# Cuisine purity per cluster (how homogeneous)
print(f"\nCuisine purity per cluster:")
cluster_purity = []
for cluster_id in range(optimal_n_clusters):
    cluster_meals = clustering_results[clustering_results['cluster'] == cluster_id]
    if len(cluster_meals) > 0:
        # Most common cuisine in this cluster
        most_common_cuisine = cluster_meals['cuisine'].value_counts().iloc[0]
        purity = most_common_cuisine / len(cluster_meals)
        cluster_purity.append(purity)

print(f"  Mean purity: {np.mean(cluster_purity)*100:.1f}%")
print(f"  (Higher = clusters are more cuisine-specific)")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSTEP 6: SAVING CLUSTER ASSIGNMENTS")

# Save cluster assignments
clustering_results.to_csv('data_meal_clusters.csv', index=False)
print("✓ Saved: data_meal_clusters.csv")

# Save cluster summary
cluster_summary = []
for cluster_id in range(optimal_n_clusters):
    cluster_meals = clustering_results[clustering_results['cluster'] == cluster_id]
    
    # Get top cuisines in this cluster
    cuisine_counts = cluster_meals['cuisine'].value_counts()
    top_cuisines = ', '.join([f"{cuisine} ({count})" 
                             for cuisine, count in cuisine_counts.head(3).items()])
    
    # Get sample meals
    sample_meals = cluster_meals['meal_name'].head(5).tolist()
    
    cluster_summary.append({
        'cluster_id': cluster_id,
        'size': len(cluster_meals),
        'top_cuisines': top_cuisines,
        'sample_meals': ' | '.join(sample_meals)
    })

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df.to_csv('cluster_summary.csv', index=False)
print("✓ Saved: cluster_summary.csv")

# ============================================================
# VISUALIZE CLUSTERS
# ============================================================

print("\nSTEP 7: VISUALIZING CLUSTER DISTRIBUTION")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cluster sizes
ax1 = axes[0]
cluster_sizes.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.8)
ax1.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Meals', fontsize=12, fontweight='bold')
ax1.set_title(f'Cluster Size Distribution ({optimal_n_clusters} clusters)', 
              fontsize=13, fontweight='bold')
ax1.axhline(cluster_sizes.mean(), color='red', linestyle='--', 
            label=f'Mean: {cluster_sizes.mean():.1f}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Cuisine distribution within clusters
ax2 = axes[1]
cuisine_cluster_matrix = pd.crosstab(clustering_results['cluster'], 
                                     clustering_results['cuisine'])
cuisine_cluster_matrix.plot(kind='bar', stacked=True, ax=ax2, 
                            colormap='tab20', alpha=0.8)
ax2.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Meals', fontsize=12, fontweight='bold')
ax2.set_title('Cuisine Composition by Cluster', fontsize=13, fontweight='bold')
ax2.legend(title='Cuisine', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('meal_clustering_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: meal_clustering_analysis.png")
plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("STAGE 2C COMPLETE")
print("="*70)
print(f"""
CLUSTERING SUMMARY:
  • Total meals clustered: {len(clustering_results)}
  • Number of clusters: {optimal_n_clusters}
  • Silhouette score: {optimal_score:.3f}
  • Mean cluster size: {cluster_sizes.mean():.1f} meals
  • Mean cuisine purity: {np.mean(cluster_purity)*100:.1f}%

FILES CREATED:
  ✓ data_meal_clusters.csv - Cluster assignments for each meal
  ✓ cluster_summary.csv - Summary of each cluster
  ✓ meal_clustering_elbow.png - Optimization plot
  ✓ meal_clustering_analysis.png - Cluster analysis
  """
)