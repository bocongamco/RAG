#!/usr/bin/env python3
"""
Analyze and visualize RAG evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('evaluation_results.json', 'r') as f:
    results = json.load(f)

# Calculate averages
def calculate_averages(mode_results):
    metrics = ['EM', 'F1', 'Precision@k', 'Recall@k', 'MRR@k', 'nDCG@k', 'Faithfulness', 'Attribution']
    avgs = {}
    for metric in metrics:
        avgs[metric] = sum(r[metric] for r in mode_results) / len(mode_results)
    return avgs

# Print summary
print("="*80)
print("RAG EVALUATION RESULTS SUMMARY")
print("="*80)

for mode, mode_results in results.items():
    avgs = calculate_averages(mode_results)
    print(f"\n{mode.upper()}")
    print("-"*80)
    print(f"  EM: {avgs['EM']:.3f} | F1: {avgs['F1']:.3f}")
    print(f"  P@3: {avgs['Precision@k']:.3f} | R@3: {avgs['Recall@k']:.3f}")
    print(f"  MRR@3: {avgs['MRR@k']:.3f} | nDCG@3: {avgs['nDCG@k']:.3f}")
    print(f"  Faithfulness: {avgs['Faithfulness']:.3f} | Attribution: {avgs['Attribution']:.3f}")

# Best query analysis
print("\n" + "="*80)
print("BEST PERFORMING QUERIES")
print("="*80)

for mode, mode_results in results.items():
    best = max(mode_results, key=lambda x: x['nDCG@k'])
    print(f"\n{mode}: '{best['query']}'")
    print(f"  nDCG@3: {best['nDCG@k']:.3f} | R@3: {best['Recall@k']:.3f}")

# Worst query analysis
print("\n" + "="*80)
print("WORST PERFORMING QUERIES")
print("="*80)

for mode, mode_results in results.items():
    worst = min(mode_results, key=lambda x: x['nDCG@k'])
    print(f"\n{mode}: '{worst['query']}'")
    print(f"  nDCG@3: {worst['nDCG@k']:.3f} | R@3: {worst['Recall@k']:.3f}")

# Category breakdown
print("\n" + "="*80)
print("PERFORMANCE BY QUERY CATEGORY")
print("="*80)

categories = {}
for mode, mode_results in results.items():
    for result in mode_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            'mode': mode,
            'nDCG': result['nDCG@k'],
            'recall': result['Recall@k']
        })

for cat, cat_results in categories.items():
    avg_ndcg = sum(r['nDCG'] for r in cat_results) / len(cat_results)
    avg_recall = sum(r['recall'] for r in cat_results) / len(cat_results)
    print(f"\n{cat.upper()}: nDCG={avg_ndcg:.3f}, Recall={avg_recall:.3f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RAG System Evaluation Results', fontsize=16, fontweight='bold')

modes = list(results.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# Plot 1: Average metrics comparison
ax1 = axes[0, 0]
metrics = ['F1', 'Precision@k', 'Recall@k', 'nDCG@k']
x = np.arange(len(modes))
width = 0.2

for i, metric in enumerate(metrics):
    values = [calculate_averages(results[mode])[metric] for mode in modes]
    ax1.bar(x + i*width, values, width, label=metric, color=colors[i])

ax1.set_ylabel('Score')
ax1.set_title('Average Metrics by Mode')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(modes, rotation=15)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Faithfulness vs Attribution
ax2 = axes[0, 1]
for i, mode in enumerate(modes):
    avgs = calculate_averages(results[mode])
    ax2.scatter(avgs['Faithfulness'], avgs['Attribution'], 
               s=200, color=colors[i], label=mode, alpha=0.7)

ax2.set_xlabel('Faithfulness')
ax2.set_ylabel('Attribution')
ax2.set_title('Faithfulness vs Attribution')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0.9, 1.1)

# Plot 3: Query category performance
ax3 = axes[1, 0]
categories_list = list(categories.keys())
category_scores = []

for cat in categories_list:
    avg_ndcg = sum(r['nDCG'] for r in categories[cat]) / len(categories[cat])
    category_scores.append(avg_ndcg)

bars = ax3.barh(categories_list, category_scores, color=colors[:len(categories_list)])
ax3.set_xlabel('Average nDCG@3')
ax3.set_title('Performance by Query Category')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Mode comparison radar
ax4 = axes[1, 1]
metrics = ['F1', 'Recall@k', 'nDCG@k', 'Faithfulness']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

ax4 = plt.subplot(2, 2, 4, projection='polar')
for i, mode in enumerate(modes):
    avgs = calculate_averages(results[mode])
    values = [avgs[m] for m in metrics]
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=2, label=mode, color=colors[i])
    ax4.fill(angles, values, alpha=0.15, color=colors[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metrics)
ax4.set_ylim(0, 0.6)
ax4.set_title('Overall Performance Comparison')
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax4.grid(True)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("Visualization saved to: evaluation_results.png")
print("="*80)

plt.show()
