import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

 
print("STAGE 7: VISUALIZATION")
 

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

 # LOAD DATA
 
print("\nLOADING DATA...")

# Stage 4 - Preference results
preference_df = pd.read_csv('preference_evaluation_summary_TFIDF.csv')

# Stage 5 - Combined results
liked_safe_df = pd.read_csv('stage5_perspective1_liked_and_safe_TFIDF.csv')
coverage_df = pd.read_csv('stage5_perspective2_coverage_TFIDF.csv')
acceptance_df = pd.read_csv('stage5_perspective3_acceptance_TFIDF.csv')

# Stage 6 - Statistical tests
stats_pref_df = pd.read_csv('stage6_statistical_tests_preference.csv')
stats_comb_df = pd.read_csv('stage6_statistical_tests_combined.csv')

print("✓ Loaded all evaluation results")

 # FIGURE 1: MAIN RESULTS - PREFERENCE METRICS
 
print("\nCreating Figure 1: Preference Evaluation Results...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Clean algorithm names for display
def clean_name(name):
    name = name.replace('_', ' ').title()
    return name.replace('Selected', '(Cuisine)').replace('All', '(Cross-Cuisine)')

preference_df['Clean_Name'] = preference_df['Algorithm'].apply(clean_name)

# Sort by F1 score
preference_sorted = preference_df.sort_values('F1@10 (%)', ascending=False)

# Filter to main algorithms (exclude random/popularity_all for clarity)
main_algos = preference_sorted[~preference_sorted['Algorithm'].isin(['random', 'popularity_all'])]

# Plot 1: F1 Scores
ax1 = axes[0]
bars1 = ax1.barh(range(len(main_algos)), main_algos['F1@10 (%)'].astype(float), 
                  color=['#2ecc71' if 'Selected' in algo else '#95a5a6' 
                         for algo in main_algos['Algorithm']])
ax1.set_yticks(range(len(main_algos)))
ax1.set_yticklabels(main_algos['Clean_Name'], fontsize=10)
ax1.set_xlabel('F1@10 Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('A) Preference Alignment (F1@10)', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 25)

# Add value labels
for i, (idx, row) in enumerate(main_algos.iterrows()):
    ax1.text(float(row['F1@10 (%)']) + 0.5, i, f"{float(row['F1@10 (%)']):.1f}%", 
             va='center', fontsize=9)

# Plot 2: Precision and Recall
ax2 = axes[1]
x = np.arange(len(main_algos))
width = 0.35

precision_bars = ax2.barh(x - width/2, main_algos['Precision@10 (%)'].astype(float), 
                          width, label='Precision', color='#3498db', alpha=0.8)
recall_bars = ax2.barh(x + width/2, main_algos['Recall@10 (%)'].astype(float), 
                       width, label='Recall', color='#e74c3c', alpha=0.8)

ax2.set_yticks(x)
ax2.set_yticklabels(main_algos['Clean_Name'], fontsize=10)
ax2.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('B) Precision vs Recall', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xlim(0, 40)

plt.tight_layout()
plt.savefig('figure1_preference_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure1_preference_metrics.png")
plt.close()

 # FIGURE 2: COMBINED EVALUATION - LIKED & SAFE
 
print("\nCreating Figure 2: Combined Evaluation (Liked & Safe)...")

fig, ax = plt.subplots(figsize=(10, 6))

liked_safe_df['Clean_Name'] = liked_safe_df['Algorithm'].apply(clean_name)
liked_safe_sorted = liked_safe_df.sort_values('Rate (%)', ascending=False)

# Filter main algorithms
main_liked_safe = liked_safe_sorted[~liked_safe_sorted['Algorithm'].isin(['random', 'popularity_all'])]

# Create bars
bars = ax.barh(range(len(main_liked_safe)), 
               main_liked_safe['Rate (%)'], 
               color=['#27ae60' if 'Selected' in algo else '#7f8c8d' 
                      for algo in main_liked_safe['Algorithm']])

ax.set_yticks(range(len(main_liked_safe)))
ax.set_yticklabels(main_liked_safe['Clean_Name'], fontsize=11)
ax.set_xlabel('Liked & Safe Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Liked & Safe Rate: Combined Preference + Safety Evaluation', 
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 8)

# Add value labels
for i, (idx, row) in enumerate(main_liked_safe.iterrows()):
    ax.text(row['Rate (%)'] + 0.15, i, 
            f"{row['Rate (%)']:.1f}% ({row['Liked & Safe']}/{row['Total Recs']})", 
            va='center', fontsize=9)

# Add vertical line at mean
mean_rate = main_liked_safe['Rate (%)'].mean()
ax.axvline(mean_rate, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_rate:.1f}%')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figure2_liked_and_safe.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure2_liked_and_safe.png")
plt.close()

 # FIGURE 3: CUISINE FILTERING EFFECT - NOW WITH HORIZONTAL BARS
 
print("\nCreating Figure 3: Cuisine Filtering Impact...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prepare data for comparison
cuisine_comparison = {
    'Content-Based': {
        'Selected': preference_sorted[preference_sorted['Algorithm'] == 'content_based_selected']['F1@10 (%)'].values[0],
        'All': preference_sorted[preference_sorted['Algorithm'] == 'content_based_all']['F1@10 (%)'].values[0]
    },
    'Collaborative': {
        'Selected': preference_sorted[preference_sorted['Algorithm'] == 'collaborative_selected']['F1@10 (%)'].values[0],
        'All': preference_sorted[preference_sorted['Algorithm'] == 'collaborative_all']['F1@10 (%)'].values[0]
    },
    'Hybrid': {
        'Selected': preference_sorted[preference_sorted['Algorithm'] == 'hybrid_selected']['F1@10 (%)'].values[0],
        'All': preference_sorted[preference_sorted['Algorithm'] == 'hybrid_all']['F1@10 (%)'].values[0]
    }
}

# Get p-values from stats
def get_pvalue(algo_base, stats_df):
    selected = f"{algo_base}_selected"
    all_algo = f"{algo_base}_all"
    row = stats_df[(stats_df['algorithm_1'] == selected) & (stats_df['algorithm_2'] == all_algo)]
    if not row.empty:
        return row.iloc[0]['p_value']
    return None

# Plot 1: Preference (F1) - NOW HORIZONTAL
ax1 = axes[0]
algos = list(cuisine_comparison.keys())
selected_scores = [float(cuisine_comparison[a]['Selected']) for a in algos]
all_scores = [float(cuisine_comparison[a]['All']) for a in algos]

y = np.arange(len(algos))
height = 0.35

# Changed from ax1.bar to ax1.barh for horizontal bars
bars1 = ax1.barh(y + height/2, selected_scores, height, 
                label='Cuisine-Filtered (Selected)', color='#2ecc71', alpha=0.9)
bars2 = ax1.barh(y - height/2, all_scores, height, 
                label='Cross-Cuisine (All)', color='#95a5a6', alpha=0.9)

ax1.set_xlabel('F1@10 Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('A) Cuisine Filtering Effect on Preference', fontsize=13, fontweight='bold')
ax1.set_yticks(y)
ax1.set_yticklabels(algos, fontsize=11)
ax1.legend(fontsize=10, loc='lower right')
ax1.set_xlim(0, 25)

# Add value labels and significance stars - adjusted for horizontal bars
for i, algo in enumerate(algos):
    # Selected bars (now on right side)
    ax1.text(selected_scores[i] + 0.5, i + height/2, f"{selected_scores[i]:.1f}%", 
             va='center', fontsize=9, fontweight='bold')
    # All bars (now on right side)
    ax1.text(all_scores[i] + 0.5, i - height/2, f"{all_scores[i]:.1f}%", 
             va='center', fontsize=9)
    
    # Add significance stars - now to the right of bars
    p_val = get_pvalue(algo.lower().replace('-', '_'), stats_pref_df)
    if p_val is not None:
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        max_score = max(selected_scores[i], all_scores[i])
        ax1.text(max_score + 2.0, i, stars, va='center', fontsize=12, fontweight='bold')

# Plot 2: Combined (Liked & Safe) - NOW HORIZONTAL
ax2 = axes[1]

cuisine_comparison_combined = {
    'Content-Based': {
        'Selected': liked_safe_sorted[liked_safe_sorted['Algorithm'] == 'content_based_selected']['Rate (%)'].values[0],
        'All': liked_safe_sorted[liked_safe_sorted['Algorithm'] == 'content_based_all']['Rate (%)'].values[0]
    },
    'Collaborative': {
        'Selected': liked_safe_sorted[liked_safe_sorted['Algorithm'] == 'collaborative_selected']['Rate (%)'].values[0],
        'All': liked_safe_sorted[liked_safe_sorted['Algorithm'] == 'collaborative_all']['Rate (%)'].values[0]
    },
    'Hybrid': {
        'Selected': liked_safe_sorted[liked_safe_sorted['Algorithm'] == 'hybrid_selected']['Rate (%)'].values[0],
        'All': liked_safe_sorted[liked_safe_sorted['Algorithm'] == 'hybrid_all']['Rate (%)'].values[0]
    }
}

selected_scores_comb = [cuisine_comparison_combined[a]['Selected'] for a in algos]
all_scores_comb = [cuisine_comparison_combined[a]['All'] for a in algos]

# Changed from ax2.bar to ax2.barh for horizontal bars
bars3 = ax2.barh(y + height/2, selected_scores_comb, height, 
                label='Cuisine-Filtered (Selected)', color='#27ae60', alpha=0.9)
bars4 = ax2.barh(y - height/2, all_scores_comb, height, 
                label='Cross-Cuisine (All)', color='#7f8c8d', alpha=0.9)

ax2.set_xlabel('Liked & Safe Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('B) Cuisine Filtering Effect on Combined Metric', fontsize=13, fontweight='bold')
ax2.set_yticks(y)
ax2.set_yticklabels(algos, fontsize=11)
ax2.legend(fontsize=10, loc='lower right')
ax2.set_xlim(0, 8)

# Add value labels and significance stars - adjusted for horizontal bars
for i, algo in enumerate(algos):
    ax2.text(selected_scores_comb[i] + 0.15, i + height/2, f"{selected_scores_comb[i]:.1f}%", 
             va='center', fontsize=9, fontweight='bold')
    ax2.text(all_scores_comb[i] + 0.15, i - height/2, f"{all_scores_comb[i]:.1f}%", 
             va='center', fontsize=9)
    
    # Add significance stars - now to the right of bars
    p_val = get_pvalue(algo.lower().replace('-', '_'), stats_comb_df)
    if p_val is not None:
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        max_score = max(selected_scores_comb[i], all_scores_comb[i])
        ax2.text(max_score + 0.5, i, stars, va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figure3_cuisine_filtering_effect.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure3_cuisine_filtering_effect.png")
plt.close()

 # FIGURE 4: THREE PERSPECTIVES COMPARISON
 
print("\nCreating Figure 4: Three Evaluation Perspectives...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Top 5 algorithms for each perspective
top_algos = ['hybrid_selected', 'collaborative_selected', 'content_based_selected', 
             'hybrid_all', 'collaborative_all']

# Perspective 1: Liked & Safe Rate
ax1 = axes[0]
p1_data = liked_safe_df[liked_safe_df['Algorithm'].isin(top_algos)].sort_values('Rate (%)', ascending=False)
p1_data['Clean_Name'] = p1_data['Algorithm'].apply(clean_name)

bars1 = ax1.barh(range(len(p1_data)), p1_data['Rate (%)'], 
                  color=['#27ae60' if 'selected' in algo else '#95a5a6' 
                         for algo in p1_data['Algorithm']])
ax1.set_yticks(range(len(p1_data)))
ax1.set_yticklabels(p1_data['Clean_Name'], fontsize=9)
ax1.set_xlabel('Rate (%)', fontsize=10, fontweight='bold')
ax1.set_title('A) Liked & Safe Rate', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 7)

for i, (idx, row) in enumerate(p1_data.iterrows()):
    ax1.text(row['Rate (%)'] + 0.15, i, f"{row['Rate (%)']:.1f}%", va='center', fontsize=8)

# Perspective 2: Coverage
ax2 = axes[1]
p2_data = coverage_df[coverage_df['Algorithm'].isin(top_algos)].sort_values('Coverage (%)', ascending=False)
p2_data['Clean_Name'] = p2_data['Algorithm'].apply(clean_name)

bars2 = ax2.barh(range(len(p2_data)), p2_data['Coverage (%)'], 
                  color=['#3498db' if 'selected' in algo else '#95a5a6' 
                         for algo in p2_data['Algorithm']])
ax2.set_yticks(range(len(p2_data)))
ax2.set_yticklabels(p2_data['Clean_Name'], fontsize=9)
ax2.set_xlabel('Coverage (%)', fontsize=10, fontweight='bold')
ax2.set_title('B) Coverage of Liked Foods', fontsize=11, fontweight='bold')
ax2.set_xlim(0, 15)

for i, (idx, row) in enumerate(p2_data.iterrows()):
    ax2.text(row['Coverage (%)'] + 0.3, i, f"{row['Coverage (%)']:.1f}%", va='center', fontsize=8)

# Perspective 3: Acceptance
ax3 = axes[2]
p3_data = acceptance_df[acceptance_df['Algorithm'].isin(top_algos)].sort_values('Acceptance (%)', ascending=False)
p3_data['Clean_Name'] = p3_data['Algorithm'].apply(clean_name)

bars3 = ax3.barh(range(len(p3_data)), p3_data['Acceptance (%)'], 
                  color=['#e74c3c' if 'selected' in algo else '#95a5a6' 
                         for algo in p3_data['Algorithm']])
ax3.set_yticks(range(len(p3_data)))
ax3.set_yticklabels(p3_data['Clean_Name'], fontsize=9)
ax3.set_xlabel('Acceptance (%)', fontsize=10, fontweight='bold')
ax3.set_title('C) Safety-First Acceptance', fontsize=11, fontweight='bold')
ax3.set_xlim(0, 12)

for i, (idx, row) in enumerate(p3_data.iterrows()):
    ax3.text(row['Acceptance (%)'] + 0.25, i, f"{row['Acceptance (%)']:.1f}%", va='center', fontsize=8)

plt.tight_layout()
plt.savefig('figure4_three_perspectives.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure4_three_perspectives.png")
plt.close()

 # FIGURE 5: SUMMARY TABLE
 
print("\nCreating Figure 5: Comprehensive Summary Table...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare summary data
summary_data = []
for algo in ['hybrid_selected', 'collaborative_selected', 'content_based_selected',
             'hybrid_all', 'collaborative_all', 'content_based_all']:
    
    # Get metrics
    pref = preference_df[preference_df['Algorithm'] == algo]
    ls = liked_safe_df[liked_safe_df['Algorithm'] == algo]
    cov = coverage_df[coverage_df['Algorithm'] == algo]
    acc = acceptance_df[acceptance_df['Algorithm'] == algo]
    
    if not pref.empty:
        summary_data.append([
            clean_name(algo),
            f"{float(pref['F1@10 (%)'].values[0]):.1f}%",
            f"{float(pref['Precision@10 (%)'].values[0]):.1f}%",
            f"{float(pref['Recall@10 (%)'].values[0]):.1f}%",
            f"{ls['Rate (%)'].values[0]:.1f}%",
            f"{cov['Coverage (%)'].values[0]:.1f}%",
            f"{acc['Acceptance (%)'].values[0]:.1f}%"
        ])

table = ax.table(cellText=summary_data,
                colLabels=['Algorithm', 'F1@10', 'Precision@10', 'Recall@10', 
                          'Liked & Safe', 'Coverage', 'Acceptance'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.12, 0.12, 0.12, 0.13, 0.13, 0.13])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(summary_data) + 1):
    for j in range(7):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        
        # Highlight best in each column (skip column 0 which is algorithm name)
        if j > 0:
            col_values = [float(row[j].rstrip('%')) for row in summary_data]
            if float(summary_data[i-1][j].rstrip('%')) == max(col_values):
                table[(i, j)].set_facecolor('#d5f4e6')
                table[(i, j)].set_text_props(weight='bold')

plt.title('Comprehensive Algorithm Comparison: All Metrics', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig('figure5_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure5_summary_table.png")
plt.close()

 # CREATE README
 
print("\nCreating visualization guide...")

readme = """# STAGE 7: VISUALIZATION OUTPUTS

## Generated Figures for Thesis

### Figure 1: Preference Evaluation Results
**File:** `figure1_preference_metrics.png`
**Description:** Shows F1, Precision, and Recall scores for all algorithms (horizontal bars)
**Use in thesis:** Section on recommendation quality / preference alignment

### Figure 2: Liked & Safe Rate
**File:** `figure2_liked_and_safe.png`
**Description:** Combined preference + safety evaluation (horizontal bars)
**Use in thesis:** Main results section - this is your PRIMARY metric

### Figure 3: Cuisine Filtering Effect
**File:** `figure3_cuisine_filtering_effect.png`
**Description:** Side-by-side comparison of Selected vs All cuisines with p-values (horizontal bars)
**Use in thesis:** This demonstrates your MAIN CONTRIBUTION (cuisine filtering)
**Key insight:** Shows highly significant improvements (p<0.001) from cuisine filtering

### Figure 4: Three Perspectives
**File:** `figure4_three_perspectives.png`
**Description:** Liked & Safe, Coverage, and Acceptance rates side-by-side (horizontal bars)
**Use in thesis:** Comprehensive evaluation section

### Figure 5: Summary Table
**File:** `figure5_summary_table.png`
**Description:** Complete comparison table of all metrics
**Use in thesis:** Results summary / appendix

## Statistical Significance Notation

In Figure 3, the stars to the right of bars indicate:
- `***` p < 0.001 (highly significant)
- `**`  p < 0.01  (very significant)
- `*`   p < 0.05  (significant)
- `ns`  p ≥ 0.05  (not significant)

## Color Coding

- **Green bars:** Cuisine-filtered algorithms (Selected)
- **Gray bars:** Cross-cuisine algorithms (All)
- **Highlighted cells:** Best performance in each metric (Figure 5)

## Thesis Recommendations

1. Use Figure 3 to demonstrate your main contribution (cuisine filtering)
2. Use Figure 2 as your primary results figure (Liked & Safe rate)
3. Use Figure 1 to show traditional recommendation metrics
4. Include Figure 5 in appendix for complete comparison
5. Reference statistical significance from Stage 6 results

All figures are 300 DPI, publication-ready quality with consistent horizontal bar orientation.
"""

with open('VISUALIZATION_GUIDE.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("✓ Saved: VISUALIZATION_GUIDE.md")

 
print("STAGE 7 COMPLETE")
 
print(f"""
Created 5 publication-ready figures (ALL WITH HORIZONTAL BARS):

  1. figure1_preference_metrics.png - F1, Precision, Recall (horizontal)
  2. figure2_liked_and_safe.png - Primary metric (horizontal)
  3. figure3_cuisine_filtering_effect.png - Main contribution with p-values (horizontal)
  4. figure4_three_perspectives.png - All three evaluation perspectives (horizontal)
  5. figure5_summary_table.png - Comprehensive comparison table

All figures are 300 DPI and ready for thesis inclusion.
""")