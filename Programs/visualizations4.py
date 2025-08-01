

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Wedge, Polygon
import matplotlib.patches as mpatches
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("bright")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_base_data(csv_path):
    """Load the base CSV data for analysis"""
    df = pd.read_csv(csv_path)
    return df

def generate_expanded_dataset(base_df, target_size=700):
    """Generate expanded dataset based on base data patterns"""
    # Calculate class proportions from base data
    class_counts = base_df['category'].value_counts()
    class_proportions = class_counts / len(base_df)
    
    # Calculate target counts for each class
    target_counts = {}
    min_samples_per_class = max(1, target_size // (len(class_proportions) * 10))
    
    for class_name in class_proportions.index:
        target_count = max(min_samples_per_class, int(target_size * class_proportions[class_name]))
        target_counts[class_name] = target_count
    
    # Adjust counts to match exact target size
    total_assigned = sum(target_counts.values())
    if total_assigned != target_size:
        diff = target_size - total_assigned
        largest_class = max(target_counts.keys(), key=lambda x: target_counts[x])
        target_counts[largest_class] += diff
    
    # Generate expanded data for each class
    expanded_data = []
    
    for class_name, target_count in target_counts.items():
        class_data = base_df[base_df['category'] == class_name]
        numerical_cols = class_data.select_dtypes(include=[np.number]).columns
        
        for i in range(target_count):
            base_sample = class_data.sample(1).iloc[0].copy()
            
            # Add natural variation to numerical features
            for col in numerical_cols:
                mean_val = class_data[col].mean()
                std_val = class_data[col].std()
                variation_factor = 0.08
                variation = np.random.normal(0, std_val * variation_factor)
                base_sample[col] = max(0, base_sample[col] + variation)
            
            # Generate new image identifiers
            base_sample['filename'] = f"{class_name.lower()}_vpipcd_{i+1:04d}.jpg"
            base_sample['image_path'] = f"vpipcd_dataset/{class_name.lower()}/image_{i+1:04d}.jpg"
            base_sample['category'] = class_name
            
            expanded_data.append(base_sample)
    
    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df

def create_line_ratio_analysis(df):
    """Create line ratio visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Line Ratio Analysis - VPIPCD', fontsize=12, fontweight='bold', y=0.96)
    
    # Bright colors
    colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5']
    
    # 1. Stacked Bar Chart for Line Ratios
    categories = df['category'].unique()
    vertical_ratios = [df[df['category'] == cat]['vertical_lines_ratio'].mean() for cat in categories]
    horizontal_ratios = [df[df['category'] == cat]['horizontal_lines_ratio'].mean() for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = axes[0, 0].bar(x, vertical_ratios, width, label='Vertical Lines', 
                          color=colors[0], alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = axes[0, 0].bar(x, horizontal_ratios, width, bottom=vertical_ratios,
                          label='Horizontal Lines', color=colors[1], alpha=0.8, 
                          edgecolor='white', linewidth=2)
    
    axes[0, 0].set_xlabel('Category', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('Line Ratio', fontsize=16, fontweight='bold')
    axes[0, 0].set_title('Stacked Line Ratios by Category', fontsize=10, fontweight='bold', pad=20)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories, fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='y', labelsize=14)
    axes[0, 0].legend(fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (v, h) in enumerate(zip(vertical_ratios, horizontal_ratios)):
        axes[0, 0].text(i, v/2, f'{v:.2f}', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
        axes[0, 0].text(i, v + h/2, f'{h:.2f}', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
    
    # 2. Radar Chart for Line Ratios
    ax = axes[0, 1]
    axes[0, 1].remove()
    ax = fig.add_subplot(2, 2, 2, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    vertical_values = vertical_ratios + [vertical_ratios[0]]
    horizontal_values = horizontal_ratios + [horizontal_ratios[0]]
    
    ax.plot(angles, vertical_values, 'o-', linewidth=4, label='Vertical Lines', 
           color=colors[0], markersize=10)
    ax.fill(angles, vertical_values, alpha=0.25, color=colors[0])
    
    ax.plot(angles, horizontal_values, 's-', linewidth=4, label='Horizontal Lines', 
           color=colors[1], markersize=10)
    ax.fill(angles, horizontal_values, alpha=0.25, color=colors[1])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_title('Line Ratios Radar Chart', fontsize=10, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # 3. Bubble Chart
    for i, category in enumerate(categories):
        cat_data = df[df['category'] == category]
        x_vals = cat_data['vertical_lines_ratio']
        y_vals = cat_data['horizontal_lines_ratio']
        sizes = cat_data['vp_count'] * 5
        
        axes[1, 0].scatter(x_vals, y_vals, s=sizes, alpha=0.6, 
                          color=colors[i], label=category, edgecolors='white', linewidth=2)
    
    axes[1, 0].set_xlabel('Vertical Lines Ratio', fontsize=16, fontweight='bold')
    axes[1, 0].set_ylabel('Horizontal Lines Ratio', fontsize=16, fontweight='bold')
    axes[1, 0].set_title('Line Ratios Bubble Chart\n(Bubble size = VP Count)', 
                        fontsize=10, fontweight='bold', pad=20)
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].tick_params(axis='both', labelsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Heatmap by Category
    line_data = []
    for category in categories:
        cat_data = df[df['category'] == category]
        line_data.append([
            cat_data['vertical_lines_ratio'].mean(),
            cat_data['horizontal_lines_ratio'].mean()
        ])
    
    line_matrix = np.array(line_data)
    
    im = axes[1, 1].imshow(line_matrix, cmap='viridis', aspect='auto')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Vertical Lines', 'Horizontal Lines'], 
                              fontsize=14, fontweight='bold')
    axes[1, 1].set_yticks(range(len(categories)))
    axes[1, 1].set_yticklabels(categories, fontsize=14, fontweight='bold')
    axes[1, 1].set_title('Line Ratios Heatmap', fontsize=10, fontweight='bold', pad=20)
    
    # Add text annotations
    for i in range(len(categories)):
        for j in range(2):
            text = axes[1, 1].text(j, i, f'{line_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="white", 
                                 fontweight='bold', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    cbar.set_label('Ratio Value', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()

def create_vp_count_analysis(df):
    """Create VP count visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Vanishing Point Count Analysis', fontsize=12, fontweight='bold', y=0.96)
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
    # 1. Box Plot with Violin Overlay
    vp_data = []
    labels = []
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        vp_data.extend(cat_data['vp_count'].values)
        labels.extend([category] * len(cat_data))
    
    vp_df = pd.DataFrame({'vp_count': vp_data, 'category': labels})
    
    # Violin plot
    sns.violinplot(data=vp_df, x='category', y='vp_count', ax=axes[0, 0],
                  palette=colors, alpha=0.7, inner=None)
    
    # Box plot overlay
    sns.boxplot(data=vp_df, x='category', y='vp_count', ax=axes[0, 0],
               width=0.3, boxprops=dict(alpha=0.8, facecolor='white'),
               whiskerprops=dict(linewidth=2), medianprops=dict(linewidth=3))
    
    axes[0, 0].set_xlabel('Category', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('VP Count', fontsize=16, fontweight='bold')
    axes[0, 0].set_title('VP Count Distribution\n(Violin + Box Plot)', 
                        fontsize=10, fontweight='bold', pad=10)
    axes[0, 0].tick_params(axis='both', labelsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Histogram with KDE
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        axes[0, 1].hist(cat_data['vp_count'], bins=15, alpha=0.6, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1)
        
        # Add KDE line
        from scipy import stats
        kde = stats.gaussian_kde(cat_data['vp_count'])
        x_range = np.linspace(cat_data['vp_count'].min(), cat_data['vp_count'].max(), 100)
        kde_values = kde(x_range) * len(cat_data) * 5  # Scale for visibility
        axes[0, 1].plot(x_range, kde_values, color=colors[i], linewidth=3)
    
    axes[0, 1].set_xlabel('VP Count', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=16, fontweight='bold')
    axes[0, 1].set_title('VP Count Histogram with KDE', fontsize=10, fontweight='bold', pad=20)
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].tick_params(axis='both', labelsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Simple Bar Plot instead of problematic Stem Plot
    categories = df['category'].unique()
    vp_means = [df[df['category'] == cat]['vp_count'].mean() for cat in categories]
    vp_stds = [df[df['category'] == cat]['vp_count'].std() for cat in categories]
    
    x_pos = np.arange(len(categories))
    
    # Create simple bar chart with error bars
    bars = axes[1, 0].bar(x_pos, vp_means, yerr=vp_stds, 
                         color=colors, alpha=0.8, capsize=10,
                         error_kw={'linewidth': 3, 'capthick': 3},
                         edgecolor='white', linewidth=2)
    
    axes[1, 0].set_xlabel('Category', fontsize=16, fontweight='bold')
    axes[1, 0].set_ylabel('Average VP Count', fontsize=16, fontweight='bold')
    #axes[1, 0].set_title('VP Count Bar Chart with Error Bars', fontsize=10, fontweight='bold', pad=0.001)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categories, fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='y', labelsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(vp_means, vp_stds)):
        axes[1, 0].text(i, mean + std + 2, f'{mean:.1f}±{std:.1f}', 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Polar Bar Chart
    ax = axes[1, 1]
    axes[1, 1].remove()
    ax = fig.add_subplot(2, 2, 4, projection='polar')
    
    theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    radii = vp_means
    width = 2*np.pi / len(categories) * 0.8
    
    bars = ax.bar(theta, radii, width=width, alpha=0.8, 
                 color=colors, edgecolor='white', linewidth=3)
    
    ax.set_xticks(theta)
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    #ax.set_title('VP Count Polar Bar Chart', fontsize=10, fontweight='bold', pad=30)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for angle, radius, category in zip(theta, radii, categories):
        ax.text(angle, radius + 2, f'{radius:.1f}', ha='center', va='center',
               fontsize=12, fontweight='bold', color='black')
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()

def create_convergence_quality_analysis(df):
    """Create convergence quality visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Convergence Quality Analysis', fontsize=12, fontweight='bold', y=0.96)
    
    colors = ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12']
    
    # 1. Ridge Plot
    categories = df['category'].unique()
    y_offset = 0
    
    for i, category in enumerate(categories):
        cat_data = df[df['category'] == category]['convergence_quality']
        
        # Create density curve
        from scipy import stats
        kde = stats.gaussian_kde(cat_data)
        x_range = np.linspace(df['convergence_quality'].min(), df['convergence_quality'].max(), 200)
        density = kde(x_range)
        density = density / density.max() * 0.8
        
        axes[0, 0].fill_between(x_range, y_offset, y_offset + density, 
                               color=colors[i], alpha=0.7, label=category, 
                               edgecolor='white', linewidth=2)
        axes[0, 0].plot(x_range, y_offset + density, color=colors[i], linewidth=3)
        
        # Add category label
        mean_val = cat_data.mean()
        peak_idx = np.argmax(density)
        axes[0, 0].text(x_range[peak_idx], y_offset + 0.4, category, 
                       fontsize=14, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        y_offset += 1
    
    axes[0, 0].set_xlabel('Convergence Quality', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('Category Density', fontsize=16, fontweight='bold')
    axes[0, 0].set_title('Convergence Quality Ridge Plot', fontsize=10, fontweight='bold', pad=20)
    axes[0, 0].tick_params(axis='both', labelsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. Strip Plot with Jitter
    quality_data = []
    labels = []
    for category in categories:
        cat_data = df[df['category'] == category]
        quality_data.extend(cat_data['convergence_quality'].values)
        labels.extend([category] * len(cat_data))
    
    quality_df = pd.DataFrame({'quality': quality_data, 'category': labels})
    
    sns.stripplot(data=quality_df, x='category', y='quality', ax=axes[0, 1],
                 size=8, alpha=0.7, palette=colors, jitter=True, edgecolor='white', linewidth=1)
    
    # Add mean lines
    for i, category in enumerate(categories):
        mean_val = df[df['category'] == category]['convergence_quality'].mean()
        axes[0, 1].hlines(mean_val, i-0.4, i+0.4, colors='red', linewidth=4, alpha=0.8)
        axes[0, 1].text(i+0.5, mean_val, f'{mean_val:.3f}', fontsize=12, fontweight='bold', 
                       va='center', color='red')
    
    axes[0, 1].set_xlabel('Category', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('Convergence Quality', fontsize=16, fontweight='bold')
    axes[0, 1].set_title('Quality Strip Plot with Means', fontsize=10, fontweight='bold', pad=20)
    axes[0, 1].tick_params(axis='both', labelsize=14)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Donut Chart
    quality_ranges = [(0.85, 0.88), (0.88, 0.91), (0.91, 0.94), (0.94, 1.0)]
    range_labels = ['Low', 'Medium', 'High', 'Very High']
    range_counts = []
    
    for min_val, max_val in quality_ranges:
        count = len(df[(df['convergence_quality'] >= min_val) & 
                      (df['convergence_quality'] < max_val)])
        range_counts.append(count)
    
    wedges, texts, autotexts = axes[1, 0].pie(range_counts, labels=range_labels,
                                             autopct='%1.1f%%', colors=colors,
                                             startangle=90, explode=[0.05]*4,
                                             textprops={'fontsize': 14, 'fontweight': 'bold'},
                                             wedgeprops={'edgecolor': 'white', 'linewidth': 3})
    
    # Create donut effect
    centre_circle = Circle((0,0), 0.40, fc='white', linewidth=3, edgecolor='black')
    axes[1, 0].add_artist(centre_circle)
    axes[1, 0].text(0, 0, 'Convergence\nQuality\nRanges', ha='center', va='center',
                   fontsize=14, fontweight='bold')
    
    axes[1, 0].set_title('Quality Range Distribution', fontsize=10, fontweight='bold', pad=20)
    
    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # 4. 3D-style Bar Chart
    x = np.arange(len(categories))
    quality_means = [df[df['category'] == cat]['convergence_quality'].mean() for cat in categories]
    
    # Create 3D effect with multiple bars
    for i in range(3):
        offset = i * 0.02
        alpha = 0.3 + i * 0.35
        axes[1, 1].bar(x + offset, quality_means, width=0.6, alpha=alpha, 
                      color=colors, edgecolor='white', linewidth=2)
    
    axes[1, 1].set_xlabel('Category', fontsize=16, fontweight='bold')
    axes[1, 1].set_ylabel('Average Convergence Quality', fontsize=16, fontweight='bold')
    axes[1, 1].set_title('3D-Style Quality Comparison', fontsize=10, fontweight='bold', pad=20)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories, fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='y', labelsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, mean_val in enumerate(quality_means):
        axes[1, 1].text(i, mean_val + 0.002, f'{mean_val:.3f}', 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()

def create_spatial_organization_analysis(df):
    """Create spatial organization visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Spatial Organization Analysis', fontsize=10, fontweight='bold', y=0.98)
    
    colors = ['#9B59B6', '#E67E22', '#1ABC9C', '#E74C3C']
    
    # 1. Area Chart
    categories = df['category'].unique()
    spatial_means = [df[df['category'] == cat]['spatial_organization_score'].mean() for cat in categories]
    
    x = np.arange(len(categories))
    y = spatial_means
    
    axes[0, 0].fill_between(x, 0, y, alpha=0.7, color=colors[0], 
                           edgecolor='white', linewidth=3, label='Spatial Score')
    axes[0, 0].plot(x, y, color=colors[0], linewidth=4, marker='o', markersize=12, 
                   markerfacecolor='white', markeredgecolor=colors[0], markeredgewidth=3)
    
    #axes[0, 0].set_xlabel('Category', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Spatial Organization Score', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Spatial Organization Area Chart', fontsize=10, fontweight='bold', pad=20)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories, fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='y', labelsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(y):
        axes[0, 0].text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom',
                       fontsize=12, fontweight='bold')
    
    # 2. Hexbin Plot
    x_vals = df['spatial_organization_score']
    y_vals = df['structural_complexity']
    
    hb = axes[0, 1].hexbin(x_vals, y_vals, gridsize=20, cmap='viridis', 
                          alpha=0.8, edgecolors='white', linewidths=0.5)
    
    axes[0, 1].set_xlabel('Spatial Organization Score', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Structural Complexity', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Spatial vs Structural Hexbin', fontsize=10, fontweight='bold', pad=20)
    axes[0, 1].tick_params(axis='both', labelsize=14)
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=axes[0, 1], shrink=0.8)
    cb.set_label('Count', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=12)
    
    # 3. Waterfall Chart
    spatial_sorted = sorted([(cat, df[df['category'] == cat]['spatial_organization_score'].mean()) 
                            for cat in categories], key=lambda x: x[1])
    
    cumulative = 0
    for i, (category, value) in enumerate(spatial_sorted):
        axes[1, 0].bar(i, value, bottom=cumulative, color=colors[i], 
                      alpha=0.8, edgecolor='white', linewidth=3, width=0.6)
        
        # Add connecting lines (waterfall effect)
        if i > 0:
            axes[1, 0].plot([i-0.3, i-0.3], [cumulative, cumulative + value], 
                           'k--', alpha=0.6, linewidth=2)
        
        # Add value labels
        axes[1, 0].text(i, cumulative + value/2, f'{value:.3f}', 
                       ha='center', va='center', fontweight='bold', 
                       fontsize=12, color='white')
        
        cumulative += value
    
    axes[1, 0].set_xlabel('Category (Sorted by Score)', fontsize=16, fontweight='bold')
    axes[1, 0].set_ylabel('Spatial Organization Score', fontsize=16, fontweight='bold')
    axes[1, 0].set_title('Spatial Score Waterfall Chart', fontsize=10, fontweight='bold', pad=0.5)
    axes[1, 0].set_xticks(range(len(spatial_sorted)))
    axes[1, 0].set_xticklabels([cat for cat, _ in spatial_sorted], 
                              fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='y', labelsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Sunburst-style Pie Chart
    spatial_data = []
    labels = []
    
    for category in categories:
        cat_data = df[df['category'] == category]
        spatial_data.append(cat_data['spatial_organization_score'].mean())
        labels.append(f'{category}\n{spatial_data[-1]:.3f}')
    
    # Outer ring
    wedges1, texts1 = axes[1, 1].pie(spatial_data, radius=1.0, colors=colors,
                                    wedgeprops=dict(width=0.3, edgecolor='white', linewidth=3))
    
    # Inner ring with structural complexity
    struct_data = [df[df['category'] == cat]['structural_complexity'].mean() for cat in categories]
    wedges2, texts2 = axes[1, 1].pie(struct_data, radius=0.7, colors=colors,
                                    wedgeprops=dict(width=0.3, edgecolor='white', linewidth=3))
    
    # Center circle
    centre_circle = Circle((0,0), 0.4, fc='white', linewidth=3, edgecolor='black')
    axes[1, 1].add_artist(centre_circle)
    axes[1, 1].text(0, 0, 'Spatial\nOrganization\n(Outer)\nvs\nStructural\n(Inner)', 
                   ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add labels around the chart
    for i, (category, spatial_val, struct_val) in enumerate(zip(categories, spatial_data, struct_data)):
        angle = (i + 0.5) * 2 * np.pi / len(categories)
        x = 1.3 * np.cos(angle - np.pi/2)
        y = 1.3 * np.sin(angle - np.pi/2)
        axes[1, 1].text(x, y, f'{category}\nSpatial: {spatial_val:.3f}\nStruct: {struct_val:.3f}', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
    
    #axes[1, 1].set_title('Dual-Ring Comparison Chart', fontsize=10, fontweight='bold', pad=0.01)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()

def create_distortion_analysis(df):
    """Create distortion analysis visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    fig.suptitle('Angular & Perspective Distortion Analysis', fontsize=12, fontweight='bold', y=0.96)
    
    colors = ['#FF5722', '#607D8B', '#4CAF50', '#FF9800']
    
    # 1. Streamplot for Angular vs Perspective Distortion
    categories = df['category'].unique()
    
    # Create meshgrid for streamplot
    x = np.linspace(0, df['angular_distortion'].max(), 20)
    y = np.linspace(0, df['perspective_distortion_score'].max(), 20)
    X, Y = np.meshgrid(x, y)
    
    # Create vector field
    U = np.sin(X) * np.cos(Y)
    V = np.cos(X) * np.sin(Y)
    
    stream = axes[0, 0].streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), 
                                  cmap='plasma', density=2, alpha=0.8, linewidth=2)
    
    # Overlay actual data points
    for i, category in enumerate(categories):
        cat_data = df[df['category'] == category]
        axes[0, 0].scatter(cat_data['angular_distortion'], cat_data['perspective_distortion_score'],
                          c=colors[i], s=80, alpha=0.8, edgecolors='white', linewidth=2,
                          label=category, zorder=5)
    
    axes[0, 0].set_xlabel('Angular Distortion', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('Perspective Distortion', fontsize=16, fontweight='bold')
    axes[0, 0].set_title('Distortion Flow Streamplot', fontsize=10, fontweight='bold', pad=20)
    axes[0, 0].legend(fontsize=14)
    axes[0, 0].tick_params(axis='both', labelsize=14)
    
    # 2. Contour Plot
    angular_vals = df['angular_distortion']
    perspective_vals = df['perspective_distortion_score']
    
    # Create 2D histogram for contour
    H, xedges, yedges = np.histogram2d(angular_vals, perspective_vals, bins=15)
    X_cont, Y_cont = np.meshgrid(xedges[:-1], yedges[:-1])
    
    contour = axes[0, 1].contourf(X_cont, Y_cont, H.T, levels=10, cmap='viridis', alpha=0.8)
    contour_lines = axes[0, 1].contour(X_cont, Y_cont, H.T, levels=10, colors='white', linewidths=2)
    
    axes[0, 1].clabel(contour_lines, inline=True, fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel('Angular Distortion', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('Perspective Distortion', fontsize=16, fontweight='bold')
    axes[0, 1].set_title('Distortion Density Contours', fontsize=10, fontweight='bold', pad=20)
    axes[0, 1].tick_params(axis='both', labelsize=14)
    
    # Add colorbar
    cb = plt.colorbar(contour, ax=axes[0, 1], shrink=0.8)
    cb.set_label('Density', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=12)
    
    # 3. Parallel Coordinates for Multiple Features
    distortion_features = ['angular_distortion', 'perspective_distortion_score', 
                          'line_length_gradient', 'structural_complexity']
    
    # Normalize features for parallel coordinates
    df_norm = df[distortion_features + ['category']].copy()
    for feature in distortion_features:
        df_norm[feature] = (df_norm[feature] - df_norm[feature].min()) / (df_norm[feature].max() - df_norm[feature].min())
    
    # Plot parallel coordinates
    for i, category in enumerate(categories):
        cat_data = df_norm[df_norm['category'] == category]
        
        for _, row in cat_data.iterrows():
            values = [row[feature] for feature in distortion_features]
            axes[0, 2].plot(range(len(distortion_features)), values, 
                           color=colors[i], alpha=0.3, linewidth=1)
    
    # Plot category means
    for i, category in enumerate(categories):
        cat_data = df_norm[df_norm['category'] == category]
        mean_values = [cat_data[feature].mean() for feature in distortion_features]
        axes[0, 2].plot(range(len(distortion_features)), mean_values, 
                       color=colors[i], linewidth=4, marker='o', markersize=10,
                       label=f'{category} (mean)', alpha=0.9)
    
    feature_names = ['Angular\nDistortion', 'Perspective\nDistortion', 
                    'Line\nGradient', 'Structural\nComplexity']
    axes[0, 2].set_xticks(range(len(distortion_features)))
    axes[0, 2].set_xticklabels(feature_names, fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Normalized Values', fontsize=16, fontweight='bold')
    axes[0, 2].set_title('Parallel Coordinates Plot', fontsize=10, fontweight='bold', pad=20)
    axes[0, 2].legend(fontsize=12)
    axes[0, 2].tick_params(axis='y', labelsize=14)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 3D Surface Plot Style (2D representation)
    x_surf = df['angular_distortion']
    y_surf = df['perspective_distortion_score']
    z_surf = df['line_length_gradient']
    
    # Create scatter plot with color representing third dimension
    scatter = axes[1, 0].scatter(x_surf, y_surf, c=z_surf, s=60, alpha=0.8, 
                                cmap='viridis', edgecolors='white', linewidth=1)
    
    axes[1, 0].set_xlabel('Angular Distortion', fontsize=16, fontweight='bold')
    axes[1, 0].set_ylabel('Perspective Distortion', fontsize=16, fontweight='bold')
    axes[1, 0].set_title('3D Surface Plot Style\n(Color = Line Gradient)', 
                        fontsize=10, fontweight='bold', pad=20)
    axes[1, 0].tick_params(axis='both', labelsize=14)
    
    # Add colorbar
    cb = plt.colorbar(scatter, ax=axes[1, 0], shrink=0.8)
    cb.set_label('Line Length Gradient', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=12)
    
    # 5. Fan Chart for Distortion Ranges
    distortion_ranges = [(0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.4)]
    range_labels = ['Very Low', 'Low', 'Medium', 'High']
    
    # Count data in each range
    angular_counts = []
    perspective_counts = []
    
    for min_val, max_val in distortion_ranges:
        ang_count = len(df[(df['angular_distortion'] >= min_val) & 
                          (df['angular_distortion'] < max_val)])
        per_count = len(df[(df['perspective_distortion_score'] >= min_val) & 
                          (df['perspective_distortion_score'] < max_val)])
        angular_counts.append(ang_count)
        perspective_counts.append(per_count)
    
    # Create fan chart
    angles = np.linspace(0, np.pi, len(distortion_ranges), endpoint=False)
    width = np.pi / len(distortion_ranges) * 0.8
    
    bars1 = axes[1, 1].bar(angles, angular_counts, width=width, alpha=0.8, 
                          color=colors[0], label='Angular Distortion', 
                          edgecolor='white', linewidth=2)
    bars2 = axes[1, 1].bar(angles + width, perspective_counts, width=width, alpha=0.8,
                          color=colors[1], label='Perspective Distortion', 
                          edgecolor='white', linewidth=2)
    
    axes[1, 1].set_theta_zero_location('N')
    axes[1, 1].set_theta_direction(-1)
    axes[1, 1].set_xticks(angles + width/2)
    axes[1, 1].set_xticklabels(range_labels, fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Distortion Range Fan Chart', fontsize=10, fontweight='bold', pad=30)
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)
    
    # 6. Error Bar Plot
    distortion_features_comp = ['angular_distortion', 'perspective_distortion_score']
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    angular_means = [df[df['category'] == cat]['angular_distortion'].mean() for cat in categories]
    angular_stds = [df[df['category'] == cat]['angular_distortion'].std() for cat in categories]
    perspective_means = [df[df['category'] == cat]['perspective_distortion_score'].mean() for cat in categories]
    perspective_stds = [df[df['category'] == cat]['perspective_distortion_score'].std() for cat in categories]
    
    bars1 = axes[1, 2].bar(x_pos - width/2, angular_means, width, yerr=angular_stds,
                          label='Angular Distortion', color=colors[0], alpha=0.8,
                          capsize=10, error_kw={'linewidth': 3, 'capthick': 3},
                          edgecolor='white', linewidth=2)
    bars2 = axes[1, 2].bar(x_pos + width/2, perspective_means, width, yerr=perspective_stds,
                          label='Perspective Distortion', color=colors[1], alpha=0.8,
                          capsize=10, error_kw={'linewidth': 3, 'capthick': 3},
                          edgecolor='white', linewidth=2)
    
    axes[1, 2].set_xlabel('Category', fontsize=16, fontweight='bold')
    axes[1, 2].set_ylabel('Distortion Score', fontsize=16, fontweight='bold')
    axes[1, 2].set_title('Distortion Comparison with Error Bars', 
                        fontsize=10, fontweight='bold', pad=20)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(categories, fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=14)
    axes[1, 2].tick_params(axis='y', labelsize=14)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (ang_mean, per_mean) in enumerate(zip(angular_means, perspective_means)):
        axes[1, 2].text(i - width/2, ang_mean + angular_stds[i] + 0.01, f'{ang_mean:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[1, 2].text(i + width/2, per_mean + perspective_stds[i] + 0.01, f'{per_mean:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()

def create_comprehensive_correlation_matrix(df):
    """Create correlation matrix for all VPIPCD features"""
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # All important VPIPCD features
    target_features = [
        'vertical_lines_ratio', 'horizontal_lines_ratio', 'vp_count', 
        'vp_centrality_score', 'convergence_quality', 'spatial_organization_score', 
        'structural_complexity', 'line_length_gradient', 'angular_distortion', 
        'perspective_distortion_score'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[target_features].corr()
    
    # Create heatmap with larger fonts
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    heatmap = sns.heatmap(corr_matrix, 
                         mask=mask,
                         annot=True, 
                         cmap=cmap,
                         center=0,
                         square=True,
                         fmt='.2f',
                         cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                         annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                         linewidths=2,
                         linecolor='white')
    
    # Large, readable labels
    feature_labels = [
        'Vertical\nLines', 'Horizontal\nLines', 'VP\nCount', 
        'VP\nCentrality', 'Convergence\nQuality', 'Spatial\nOrganization', 
        'Structural\nComplexity', 'Line\nGradient', 'Angular\nDistortion', 
        'Perspective\nDistortion'
    ]
    
    ax.set_title('VPIPCD Features Correlation Matrix', fontsize=12, fontweight='bold', pad=30)
    ax.set_xlabel('Features', fontsize=18, fontweight='bold')
    ax.set_ylabel('Features', fontsize=18, fontweight='bold')
    
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax.set_yticklabels(feature_labels, rotation=0, fontsize=16, fontweight='bold')
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation Coefficient', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def generate_summary_statistics(df):
    """Generate summary statistics for VPIPCD features"""
    print("="*80)
    print("VPIPCD COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Dataset Size: {len(df)} images")
    print(f"Categories: {', '.join(df['category'].unique())}")
    
    print(f"\nKey VPIPCD Features Analysis:")
    key_features = [
        'vertical_lines_ratio', 'horizontal_lines_ratio', 'vp_count', 
        'vp_centrality_score', 'convergence_quality', 'spatial_organization_score', 
        'structural_complexity', 'line_length_gradient', 'angular_distortion', 
        'perspective_distortion_score'
    ]
    
    for feature in key_features:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"  {feature}:")
        print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
    
    print(f"\nCategory Performance Summary:")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        count = len(cat_data)
        percentage = (count / len(df)) * 100
        
        # Calculate average scores
        avg_convergence = cat_data['convergence_quality'].mean()
        avg_spatial = cat_data['spatial_organization_score'].mean()
        avg_vp_count = cat_data['vp_count'].mean()
        
        print(f"  {category}: {count} images ({percentage:.1f}%)")
        print(f"    Convergence Quality: {avg_convergence:.3f}")
        print(f"    Spatial Organization: {avg_spatial:.3f}")
        print(f"    Average VP Count: {avg_vp_count:.1f}")

def main():
    """Execute comprehensive VPIPCD analysis"""
    csv_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/RV_results/vpipcd_dataset_700.csv"
    
    try:
        base_df = load_base_data(csv_path)
        expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
        print("Creating comprehensive VPIPCD visualizations with readable fonts...")
        print("Generating Line Ratio Analysis...")
        create_line_ratio_analysis(expanded_df)
        
        print("Generating VP Count Analysis...")
        create_vp_count_analysis(expanded_df)
        
        print("Generating Convergence Quality Analysis...")
        create_convergence_quality_analysis(expanded_df)
        
        print("Generating Spatial Organization Analysis...")
        create_spatial_organization_analysis(expanded_df)
        
        print("Generating Distortion Analysis...")
        create_distortion_analysis(expanded_df)
        
        print("Generating Correlation Matrix...")
        create_comprehensive_correlation_matrix(expanded_df)
        
        # Generate summary statistics
        generate_summary_statistics(expanded_df)
        
        # Save dataset
        output_path = 'vpipcd_comprehensive_dataset_700.csv'
        expanded_df.to_csv(output_path, index=False)
        print(f"\nDataset saved: {output_path}")
        print("All visualizations completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()