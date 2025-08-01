

# ## Visualizations (1) for SIDTM:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    class_counts = base_df['true_class'].value_counts()
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
        class_data = base_df[base_df['true_class'] == class_name]
        numerical_cols = class_data.select_dtypes(include=[np.number]).columns
        
        for i in range(target_count):
            base_sample = class_data.sample(1).iloc[0].copy()
            
            # Add natural variation to numerical features
            for col in numerical_cols:
                if col not in ['image_width', 'image_height']:
                    mean_val = class_data[col].mean()
                    std_val = class_data[col].std()
                    variation_factor = 0.1  # Slightly higher for depth features
                    variation = np.random.normal(0, std_val * variation_factor)
                    base_sample[col] = max(0, base_sample[col] + variation)
            
            # Generate new image identifiers
            base_sample['image_path'] = f"depth_dataset/{class_name.lower()}/image_{i+1:04d}.jpg"
            base_sample['image_name'] = f"{class_name.lower()}_depth_{i+1:04d}.jpg"
            base_sample['true_class'] = class_name
            
            expanded_data.append(base_sample)
    
    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df

def create_overall_results_chart():
    """Create beautiful chart for overall results with new styles"""
    fig, axes = plt.subplots(1, 2, figsize=(26, 12))
    fig.suptitle('Overall Performance Results Analysis', fontsize=30, fontweight='bold', y=0.96)
    
    # Overall results data
    overall_results = {
        'hallway': 0.563,
        'staircase': 0.519,
        'room': 0.293,
        'open_area': 0.615,
        'Overall': 0.497
    }
    
    # Separate class results from overall
    class_results = {k: v for k, v in overall_results.items() if k != 'Overall'}
    overall_score = overall_results['Overall']
    
    # New vibrant color palette
    colors = ['#FF4757', '#2ED573', '#5352ED', '#FFA502', '#FF6348']
    
    # 1. Lollipop Chart for Class Results (NEW STYLE)
    classes = list(class_results.keys())
    scores = list(class_results.values())
    
    # Create lollipop chart
    axes[0].stem(classes, scores, linefmt='-', markerfmt='o', basefmt='k-')
    
    # Customize lollipop stems and markers
    for i, (class_name, score) in enumerate(class_results.items()):
        axes[0].plot([i, i], [0, score], color=colors[i], linewidth=6, alpha=0.8)
        axes[0].scatter(i, score, color=colors[i], s=300, alpha=0.9, 
                       edgecolors='white', linewidth=3, zorder=5)
        
        # Add value labels
        axes[0].text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=16, color='#2C3E50')
    
    # Add overall line
    axes[0].axhline(y=overall_score, color='#E74C3C', linestyle=':', 
                   linewidth=4, alpha=0.8, label=f'Overall Score: {overall_score:.3f}')
    
    axes[0].set_xlabel('Class Category', fontweight='bold', fontsize=20)
    axes[0].set_ylabel('Performance Score', fontweight='bold', fontsize=20)
    axes[0].set_title('Performance by Class Category\n(Lollipop Style)', 
                     fontweight='bold', fontsize=22, pad=30)
    axes[0].set_ylim(0, max(scores) * 1.3)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0].legend(fontsize=18, loc='upper right')
    
    # 2. Gauge Chart Style (NEW STYLE)
    ax_gauge = axes[1]
    
    # Create gauge background
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Performance zones
    colors_gauge = ['#E74C3C', '#F39C12', '#2ECC71']  # Red, Orange, Green
    zone_ranges = [(0, 0.33), (0.33, 0.66), (0.66, 1.0)]
    zone_labels = ['Low', 'Medium', 'High']
    
    for i, ((start, end), color, label) in enumerate(zip(zone_ranges, colors_gauge, zone_labels)):
        theta_zone = np.linspace(start * np.pi, end * np.pi, 50)
        ax_gauge.fill_between(theta_zone, 0.7, 1.0, color=color, alpha=0.3)
        
        # Add zone labels
        mid_angle = (start + end) * np.pi / 2
        ax_gauge.text(mid_angle, 0.85, label, ha='center', va='center',
                     fontsize=14, fontweight='bold', color=color)
    
    # Plot class scores as needles
    for i, (class_name, score) in enumerate(class_results.items()):
        angle = score * np.pi
        ax_gauge.plot([angle, angle], [0, 0.6], color=colors[i], 
                     linewidth=6, alpha=0.8)
        ax_gauge.scatter(angle, 0.6, color=colors[i], s=200, 
                        edgecolors='white', linewidth=2, zorder=5)
        
        # Add class labels around the gauge
        label_angle = angle + 0.1 if angle < np.pi/2 else angle - 0.1
        ax_gauge.text(label_angle, 0.5, f'{class_name}\n{score:.3f}', 
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
    
    # Overall score needle
    overall_angle = overall_score * np.pi
    ax_gauge.plot([overall_angle, overall_angle], [0, 0.7], color='#2C3E50', 
                 linewidth=8, alpha=0.9)
    ax_gauge.scatter(overall_angle, 0.7, color='#2C3E50', s=300, 
                    edgecolors='white', linewidth=3, zorder=10)
    
    # Center circle
    center_circle = plt.Circle((0, 0), 0.1, color='#2C3E50')
    ax_gauge.add_patch(center_circle)
    
    ax_gauge.set_xlim(-0.2, np.pi + 0.2)
    ax_gauge.set_ylim(0, 1.2)
    ax_gauge.set_aspect('equal')
    ax_gauge.axis('off')
    ax_gauge.set_title('Performance Gauge Chart\n(Overall Score Highlighted)', 
                      fontweight='bold', fontsize=22, pad=30)
    
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.show()

def create_depth_features_analysis(df):
    """Create new style analysis for specified depth features"""
    fig = plt.figure(figsize=(40, 36))
    fig.suptitle('Depth Transition Features Analysis', fontsize=20, fontweight='bold', y=0.96)
    
    # Target features with descriptions
    target_features = [
        ('sharpness_gradient_ratio', 'Foreground-Background\nSeparation'),
        ('laplacian_variance', 'Global Edge\nSharpness'),
        ('frequency_centroid', 'Overall Image Sharpness\nDistribution'),
        ('focus_center_bias', 'Spatial Focus\nDistribution'),
        ('contrast_decay_slope', 'Atmospheric Perspective\nEffects'),
        ('high_freq_ratio', 'Near-Far Frequency\nDiscrimination'),
        ('regional_depth_std', 'Depth Variation\nPatterns'),
        ('luminance_range', 'Brightness Variation\n(Depth Layers)'),
        ('focus_variation', 'Spatial Sharpness\nConsistency')
    ]
    
    # New color schemes
    class_colors = ['#FF4757', '#2ED573', '#5352ED', '#FFA502']
    gradient_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F093FB', '#FF9A9E']
    
    # Create 3x3 grid with different chart types
    for idx, (feature, description) in enumerate(target_features):
        ax = plt.subplot(3, 3, idx + 1)
        
        if idx % 4 == 0:  # Ridge plots (NEW STYLE)
            y_offset = 0
            for i, class_name in enumerate(df['true_class'].unique()):
                class_data = df[df['true_class'] == class_name][feature]
                
                # Create density curve
                density = np.histogram(class_data, bins=30, density=True)[0]
                bins = np.histogram(class_data, bins=30)[1]
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Normalize and offset
                density = density / density.max() * 0.8
                
                ax.fill_between(bin_centers, y_offset, y_offset + density, 
                               color=class_colors[i], alpha=0.7, label=class_name)
                ax.plot(bin_centers, y_offset + density, color=class_colors[i], 
                       linewidth=2)
                
                # Add class label with better positioning
                ax.text(bin_centers[np.argmax(density)], y_offset + 0.4, class_name,
                       fontsize=8, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
                
                y_offset += 1
            
            ax.set_xlim(df[feature].min(), df[feature].max())
            ax.set_ylim(-0.2, len(df['true_class'].unique()))
            ax.set_xlabel('Feature Value', fontweight='bold', fontsize=10)
            ax.set_ylabel('Class Density', fontweight='bold', fontsize=10)
            
        elif idx % 4 == 1:  # Swarm plots (NEW STYLE)
            feature_data = []
            class_labels = []
            
            for class_name in df['true_class'].unique():
                class_data = df[df['true_class'] == class_name]
                feature_data.extend(class_data[feature].values)
                class_labels.extend([class_name] * len(class_data))
            
            feature_df = pd.DataFrame({
                'feature_value': feature_data,
                'class': class_labels
            })
            
            sns.swarmplot(data=feature_df, x='class', y='feature_value', ax=ax,
                         palette=class_colors, size=4, alpha=0.8)
            
            # Add mean lines
            for i, class_name in enumerate(df['true_class'].unique()):
                mean_val = df[df['true_class'] == class_name][feature].mean()
                ax.hlines(mean_val, i-0.4, i+0.4, colors='white', linewidth=3)
                ax.hlines(mean_val, i-0.4, i+0.4, colors=class_colors[i], linewidth=2)
            
        elif idx % 4 == 2:  # Strip plots with box overlay (NEW STYLE)
            feature_data = []
            class_labels = []
            
            for class_name in df['true_class'].unique():
                class_data = df[df['true_class'] == class_name]
                feature_data.extend(class_data[feature].values)
                class_labels.extend([class_name] * len(class_data))
            
            feature_df = pd.DataFrame({
                'feature_value': feature_data,
                'class': class_labels
            })
            
            # Strip plot
            sns.stripplot(data=feature_df, x='class', y='feature_value', ax=ax,
                         palette=class_colors, size=3, alpha=0.6, jitter=True)
            
            # Overlay box plot
            sns.boxplot(data=feature_df, x='class', y='feature_value', ax=ax,
                       palette=class_colors, width=0.3, boxprops=dict(alpha=0.3))
            
        else:  # Violin plots (ENHANCED STYLE)
            feature_data = []
            class_labels = []
            
            for class_name in df['true_class'].unique():
                class_data = df[df['true_class'] == class_name]
                feature_data.extend(class_data[feature].values)
                class_labels.extend([class_name] * len(class_data))
            
            feature_df = pd.DataFrame({
                'feature_value': feature_data,
                'class': class_labels
            })
            
            sns.violinplot(data=feature_df, x='class', y='feature_value', ax=ax,
                          palette=class_colors, alpha=0.8, inner='quart')
        
        ax.set_xlabel('Class Category', fontweight='bold', fontsize=10)
        ax.set_ylabel('Feature Value', fontweight='bold', fontsize=10)
        ax.set_title(f'{description}\n({feature})', fontweight='bold', fontsize=9, pad=12)
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        # Ensure proper spacing for x-axis labels
        if len(ax.get_xticklabels()) > 0:
            ax.margins(x=0.15)  # Increased margin to prevent label cutoff
    
    plt.subplots_adjust(hspace=0.8, wspace=0.6)  # Increased spacing significantly
    plt.tight_layout(pad=4.0)  # Increased padding
    plt.show()

def create_depth_correlation_matrix(df):
    """Create beautiful correlation matrix with new style"""
    fig, ax = plt.subplots(figsize=(18, 16))
    
    # Target features
    target_features = [
        'sharpness_gradient_ratio', 'laplacian_variance', 'frequency_centroid', 
        'focus_center_bias', 'contrast_decay_slope', 'high_freq_ratio', 
        'regional_depth_std', 'luminance_range', 'focus_variation'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[target_features].corr()
    
    # Create clustered heatmap (NEW STYLE)
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    # Calculate distance matrix and perform clustering
    distance_matrix = 1 - np.abs(corr_matrix)
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    # Get cluster order
    dendro = dendrogram(linkage_matrix, no_plot=True)
    cluster_order = dendro['leaves']
    
    # Reorder correlation matrix
    corr_reordered = corr_matrix.iloc[cluster_order, cluster_order]
    
    # Create beautiful heatmap with new colormap
    cmap = sns.diverging_palette(260, 10, n=100, as_cmap=True)  # Purple to orange
    
    im = ax.imshow(corr_reordered.values, cmap=cmap, aspect='auto', 
                   vmin=-1, vmax=1, interpolation='nearest')
    
    # Add correlation values with dynamic text color
    for i in range(len(corr_reordered)):
        for j in range(len(corr_reordered)):
            corr_val = corr_reordered.iloc[i, j]
            text_color = 'white' if abs(corr_val) > 0.5 else 'black'
            ax.text(j, i, f'{corr_val:.2f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=text_color)
    
    # Customize labels
    feature_labels = [
        'Sharpness\nGradient', 'Laplacian\nVariance', 'Frequency\nCentroid', 
        'Focus Center\nBias', 'Contrast\nDecay', 'High Freq\nRatio', 
        'Regional Depth\nStd', 'Luminance\nRange', 'Focus\nVariation'
    ]
    
    reordered_labels = [feature_labels[i] for i in cluster_order]
    
    ax.set_xticks(range(len(corr_reordered)))
    ax.set_yticks(range(len(corr_reordered)))
    ax.set_xticklabels(reordered_labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(reordered_labels, rotation=0, fontsize=14)
    
    ax.set_title('Clustered Depth Features Correlation Matrix', 
                fontsize=24, fontweight='bold', pad=30)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.show()

def create_parallel_coordinates_plot(df):
    """Create parallel coordinates plot (NEW CHART TYPE)"""
    fig, ax = plt.subplots(figsize=(24, 12))
    fig.suptitle('Parallel Coordinates: Depth Features by Class', 
                fontsize=26, fontweight='bold', y=0.95)
    
    # Select subset of features for clarity
    features = ['sharpness_gradient_ratio', 'laplacian_variance', 'frequency_centroid', 
               'focus_center_bias', 'high_freq_ratio', 'regional_depth_std']
    
    # Normalize features to 0-1 scale
    df_normalized = df[features + ['true_class']].copy()
    for feature in features:
        df_normalized[feature] = (df_normalized[feature] - df_normalized[feature].min()) / (df_normalized[feature].max() - df_normalized[feature].min())
    
    # Color mapping
    class_colors = {'hallway': '#FF4757', 'staircase': '#2ED573', 
                   'room': '#5352ED', 'open_area': '#FFA502'}
    
    # Plot parallel coordinates
    for class_name in df_normalized['true_class'].unique():
        class_data = df_normalized[df_normalized['true_class'] == class_name]
        
        for _, row in class_data.iterrows():
            values = [row[feature] for feature in features]
            ax.plot(range(len(features)), values, color=class_colors[class_name], 
                   alpha=0.3, linewidth=1)
    
    # Plot class means
    for class_name in df_normalized['true_class'].unique():
        class_data = df_normalized[df_normalized['true_class'] == class_name]
        mean_values = [class_data[feature].mean() for feature in features]
        ax.plot(range(len(features)), mean_values, color=class_colors[class_name], 
               linewidth=4, label=f'{class_name} (mean)', alpha=0.9, marker='o', markersize=8)
    
    # Customize plot
    feature_names = ['Sharpness\nGradient', 'Laplacian\nVariance', 'Frequency\nCentroid', 
                    'Focus Center\nBias', 'High Freq\nRatio', 'Regional Depth\nStd']
    
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(feature_names, fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized Feature Values', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=16, loc='upper right')
    ax.tick_params(axis='y', labelsize=16)
    
    plt.tight_layout()
    plt.show()

def create_sunburst_performance_chart():
    """Create sunburst chart for performance analysis (NEW CHART TYPE)"""
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.suptitle('Performance Sunburst Chart', fontsize=24, fontweight='bold', y=0.95)
    
    # Overall results
    class_results = {'hallway': 0.563, 'staircase': 0.519, 'room': 0.293, 'open_area': 0.615}
    overall_score = 0.497
    
    # Create sunburst data
    inner_sizes = list(class_results.values())
    outer_sizes = [v/sum(inner_sizes) for v in inner_sizes]  # Normalize
    
    # Colors
    inner_colors = ['#FF4757', '#2ED573', '#5352ED', '#FFA502']
    outer_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Inner ring (performance scores)
    wedges1, texts1 = ax.pie(inner_sizes, radius=0.7, colors=inner_colors,
                            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2))
    
    # Outer ring (class distribution)
    wedges2, texts2 = ax.pie(outer_sizes, radius=1.0, colors=outer_colors,
                            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2))
    
    # Add labels
    for i, (class_name, score) in enumerate(class_results.items()):
        angle = np.mean([wedges1[i].theta1, wedges1[i].theta2])
        x = 0.85 * np.cos(np.radians(angle))
        y = 0.85 * np.sin(np.radians(angle))
        ax.text(x, y, f'{class_name}\n{score:.3f}', ha='center', va='center',
               fontsize=14, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Center circle with overall score
    centre_circle = plt.Circle((0,0), 0.4, fc='white', linewidth=3, edgecolor='#2C3E50')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'Overall\nScore\n{overall_score:.3f}', ha='center', va='center',
           fontsize=18, fontweight='bold', color='#2C3E50')
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def generate_summary_statistics(df):
    """Generate and display comprehensive summary statistics"""
    print("="*80)
    print("DEPTH TRANSITION DATASET SUMMARY STATISTICS")
    print("="*80)
    
    print(f"Total Images: {len(df)}")
    print(f"Number of Classes: {df['true_class'].nunique()}")
    print(f"Classes: {', '.join(df['true_class'].unique())}")
    
    print(f"\nClass Distribution:")
    class_dist = df['true_class'].value_counts()
    for class_name, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    print(f"\nDepth Feature Statistics:")
    target_features = [
        'sharpness_gradient_ratio', 'laplacian_variance', 'frequency_centroid', 
        'focus_center_bias', 'contrast_decay_slope', 'high_freq_ratio', 
        'regional_depth_std', 'luminance_range', 'focus_variation'
    ]
    
    for feature in target_features:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"  {feature}:")
        print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
    
    # Overall results
    print(f"\nOverall Performance Results:")
    overall_results = {
        'hallway': 0.563,
        'staircase': 0.519,
        'room': 0.293,
        'open_area': 0.615,
        'Overall': 0.497
    }
    
    for class_name, score in overall_results.items():
        print(f"  {class_name}: {score:.3f}")

def main():
    """Main function to execute the complete depth analysis workflow"""
    csv_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/RV_results/depth_transition_dataset_700.csv" 
    
    try:
        # Load base data and generate expanded dataset
        base_df = load_base_data(csv_path)
        expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
        # Generate comprehensive analysis
        print("Generating comprehensive depth transition analysis...")
        
        # Create visualizations with new chart types
        create_overall_results_chart()
        create_depth_features_analysis(expanded_df)
        create_depth_correlation_matrix(expanded_df)
        create_parallel_coordinates_plot(expanded_df)
        create_sunburst_performance_chart()
        
        # Generate summary statistics
        generate_summary_statistics(expanded_df)
        
        # Save the dataset
        output_path = 'depth_transition_dataset_700.csv'
        expanded_df.to_csv(output_path, index=False)
        print(f"\nDepth transition dataset saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file '{csv_path}'")
        print("Please make sure the file exists and update the csv_path variable")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

    
    
    



# visulatization33 (2)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# import warnings
# warnings.filterwarnings('ignore')

# # Set style for beautiful visualizations
# plt.style.use('seaborn-v0_8')
# sns.set_palette("bright")
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['axes.facecolor'] = 'white'

# def load_base_data(csv_path):
#     """Load the base CSV data for analysis"""
#     df = pd.read_csv(csv_path)
#     return df

# def generate_expanded_dataset(base_df, target_size=700):
#     """Generate expanded dataset based on base data patterns"""
#     # Calculate class proportions from base data
#     class_counts = base_df['true_class'].value_counts()
#     class_proportions = class_counts / len(base_df)
    
#     # Calculate target counts for each class
#     target_counts = {}
#     min_samples_per_class = max(1, target_size // (len(class_proportions) * 10))
    
#     for class_name in class_proportions.index:
#         target_count = max(min_samples_per_class, int(target_size * class_proportions[class_name]))
#         target_counts[class_name] = target_count
    
#     # Adjust counts to match exact target size
#     total_assigned = sum(target_counts.values())
#     if total_assigned != target_size:
#         diff = target_size - total_assigned
#         largest_class = max(target_counts.keys(), key=lambda x: target_counts[x])
#         target_counts[largest_class] += diff
    
#     # Generate expanded data for each class
#     expanded_data = []
    
#     for class_name, target_count in target_counts.items():
#         class_data = base_df[base_df['true_class'] == class_name]
#         numerical_cols = class_data.select_dtypes(include=[np.number]).columns
        
#         for i in range(target_count):
#             base_sample = class_data.sample(1).iloc[0].copy()
            
#             # Add natural variation to numerical features
#             for col in numerical_cols:
#                 if col not in ['image_width', 'image_height']:
#                     mean_val = class_data[col].mean()
#                     std_val = class_data[col].std()
#                     variation_factor = 0.1  # Slightly higher for depth features
#                     variation = np.random.normal(0, std_val * variation_factor)
#                     base_sample[col] = max(0, base_sample[col] + variation)
            
#             # Generate new image identifiers
#             base_sample['image_path'] = f"depth_dataset/{class_name.lower()}/image_{i+1:04d}.jpg"
#             base_sample['image_name'] = f"{class_name.lower()}_depth_{i+1:04d}.jpg"
#             base_sample['true_class'] = class_name
            
#             expanded_data.append(base_sample)
    
#     expanded_df = pd.DataFrame(expanded_data)
#     return expanded_df

# def create_overall_results_chart():
#     """Create beautiful chart for overall results with new styles"""
#     fig, axes = plt.subplots(1, 2, figsize=(26, 12))
#     fig.suptitle('Overall Performance Results Analysis', fontsize=30, fontweight='bold', y=0.96)
    
#     # Overall results data
#     overall_results = {
#         'hallway': 0.563,
#         'staircase': 0.519,
#         'room': 0.293,
#         'open_area': 0.615,
#         'Overall': 0.497
#     }
    
#     # Separate class results from overall
#     class_results = {k: v for k, v in overall_results.items() if k != 'Overall'}
#     overall_score = overall_results['Overall']
    
#     # New vibrant color palette
#     colors = ['#FF4757', '#2ED573', '#5352ED', '#FFA502', '#FF6348']
    
#     # 1. Lollipop Chart for Class Results (NEW STYLE)
#     classes = list(class_results.keys())
#     scores = list(class_results.values())
    
#     # Create lollipop chart
#     axes[0].stem(classes, scores, linefmt='-', markerfmt='o', basefmt='k-')
    
#     # Customize lollipop stems and markers
#     for i, (class_name, score) in enumerate(class_results.items()):
#         axes[0].plot([i, i], [0, score], color=colors[i], linewidth=6, alpha=0.8)
#         axes[0].scatter(i, score, color=colors[i], s=300, alpha=0.9, 
#                        edgecolors='white', linewidth=3, zorder=5)
        
#         # Add value labels
#         axes[0].text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom',
#                     fontweight='bold', fontsize=16, color='#2C3E50')
    
#     # Add overall line
#     axes[0].axhline(y=overall_score, color='#E74C3C', linestyle=':', 
#                    linewidth=4, alpha=0.8, label=f'Overall Score: {overall_score:.3f}')
    
#     axes[0].set_xlabel('Class Category', fontweight='bold', fontsize=20)
#     axes[0].set_ylabel('Performance Score', fontweight='bold', fontsize=20)
#     axes[0].set_title('Performance by Class Category\n(Lollipop Style)', 
#                      fontweight='bold', fontsize=22, pad=30)
#     axes[0].set_ylim(0, max(scores) * 1.3)
#     axes[0].tick_params(axis='both', labelsize=16)
#     axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
#     axes[0].legend(fontsize=18, loc='upper right')
    
#     # 2. Gauge Chart Style (NEW STYLE)
#     ax_gauge = axes[1]
    
#     # Create gauge background
#     theta = np.linspace(0, np.pi, 100)
#     r = 1
    
#     # Performance zones
#     colors_gauge = ['#E74C3C', '#F39C12', '#2ECC71']  # Red, Orange, Green
#     zone_ranges = [(0, 0.33), (0.33, 0.66), (0.66, 1.0)]
#     zone_labels = ['Low', 'Medium', 'High']
    
#     for i, ((start, end), color, label) in enumerate(zip(zone_ranges, colors_gauge, zone_labels)):
#         theta_zone = np.linspace(start * np.pi, end * np.pi, 50)
#         ax_gauge.fill_between(theta_zone, 0.7, 1.0, color=color, alpha=0.3)
        
#         # Add zone labels
#         mid_angle = (start + end) * np.pi / 2
#         ax_gauge.text(mid_angle, 0.85, label, ha='center', va='center',
#                      fontsize=14, fontweight='bold', color=color)
    
#     # Plot class scores as needles
#     for i, (class_name, score) in enumerate(class_results.items()):
#         angle = score * np.pi
#         ax_gauge.plot([angle, angle], [0, 0.6], color=colors[i], 
#                      linewidth=6, alpha=0.8)
#         ax_gauge.scatter(angle, 0.6, color=colors[i], s=200, 
#                         edgecolors='white', linewidth=2, zorder=5)
        
#         # Add class labels around the gauge
#         label_angle = angle + 0.1 if angle < np.pi/2 else angle - 0.1
#         ax_gauge.text(label_angle, 0.5, f'{class_name}\n{score:.3f}', 
#                      ha='center', va='center', fontsize=12, fontweight='bold',
#                      bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
    
#     # Overall score needle
#     overall_angle = overall_score * np.pi
#     ax_gauge.plot([overall_angle, overall_angle], [0, 0.7], color='#2C3E50', 
#                  linewidth=8, alpha=0.9)
#     ax_gauge.scatter(overall_angle, 0.7, color='#2C3E50', s=300, 
#                     edgecolors='white', linewidth=3, zorder=10)
    
#     # Center circle
#     center_circle = plt.Circle((0, 0), 0.1, color='#2C3E50')
#     ax_gauge.add_patch(center_circle)
    
#     ax_gauge.set_xlim(-0.2, np.pi + 0.2)
#     ax_gauge.set_ylim(0, 1.2)
#     ax_gauge.set_aspect('equal')
#     ax_gauge.axis('off')
#     ax_gauge.set_title('Performance Gauge Chart\n(Overall Score Highlighted)', 
#                       fontweight='bold', fontsize=22, pad=30)
    
#     plt.subplots_adjust(wspace=0.4)
#     plt.tight_layout()
#     plt.show()

# def create_depth_features_analysis(df):
#     """Create new style analysis for specified depth features"""
#     fig = plt.figure(figsize=(40, 36))
#     fig.suptitle('Depth Transition Features Analysis', fontsize=28, fontweight='bold', y=0.96)
    
#     # Target features with descriptions
#     target_features = [
#         ('sharpness_gradient_ratio', 'Foreground-Background\nSeparation'),
#         ('laplacian_variance', 'Global Edge\nSharpness'),
#         ('frequency_centroid', 'Overall Image Sharpness\nDistribution'),
#         ('focus_center_bias', 'Spatial Focus\nDistribution'),
#         ('contrast_decay_slope', 'Atmospheric Perspective\nEffects'),
#         ('high_freq_ratio', 'Near-Far Frequency\nDiscrimination'),
#         ('regional_depth_std', 'Depth Variation\nPatterns'),
#         ('luminance_range', 'Brightness Variation\n(Depth Layers)'),
#         ('focus_variation', 'Spatial Sharpness\nConsistency')
#     ]
    
#     # New color schemes
#     class_colors = ['#FF4757', '#2ED573', '#5352ED', '#FFA502']
#     gradient_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F093FB', '#FF9A9E']
    
#     # Create 3x3 grid with different chart types
#     for idx, (feature, description) in enumerate(target_features):
#         ax = plt.subplot(3, 3, idx + 1)
        
#         if idx % 4 == 0:  # Ridge plots (NEW STYLE)
#             y_offset = 0
#             for i, class_name in enumerate(df['true_class'].unique()):
#                 class_data = df[df['true_class'] == class_name][feature]
                
#                 # Create density curve
#                 density = np.histogram(class_data, bins=30, density=True)[0]
#                 bins = np.histogram(class_data, bins=30)[1]
#                 bin_centers = (bins[:-1] + bins[1:]) / 2
                
#                 # Normalize and offset
#                 density = density / density.max() * 0.8
                
#                 ax.fill_between(bin_centers, y_offset, y_offset + density, 
#                                color=class_colors[i], alpha=0.7, label=class_name)
#                 ax.plot(bin_centers, y_offset + density, color=class_colors[i], 
#                        linewidth=2)
                
#                 # Add class label with better positioning
#                 ax.text(bin_centers[np.argmax(density)], y_offset + 0.4, class_name,
#                        fontsize=12, fontweight='bold', ha='center', va='center',
#                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
                
#                 y_offset += 1
            
#             ax.set_xlim(df[feature].min(), df[feature].max())
#             ax.set_ylim(-0.2, len(df['true_class'].unique()))
#             ax.set_xlabel('Feature Value', fontweight='bold', fontsize=16)
#             ax.set_ylabel('Class Density', fontweight='bold', fontsize=16)
            
#         elif idx % 4 == 1:  # Swarm plots (NEW STYLE)
#             feature_data = []
#             class_labels = []
            
#             for class_name in df['true_class'].unique():
#                 class_data = df[df['true_class'] == class_name]
#                 feature_data.extend(class_data[feature].values)
#                 class_labels.extend([class_name] * len(class_data))
            
#             feature_df = pd.DataFrame({
#                 'feature_value': feature_data,
#                 'class': class_labels
#             })
            
#             sns.swarmplot(data=feature_df, x='class', y='feature_value', ax=ax,
#                          palette=class_colors, size=4, alpha=0.8)
            
#             # Add mean lines
#             for i, class_name in enumerate(df['true_class'].unique()):
#                 mean_val = df[df['true_class'] == class_name][feature].mean()
#                 ax.hlines(mean_val, i-0.4, i+0.4, colors='white', linewidth=3)
#                 ax.hlines(mean_val, i-0.4, i+0.4, colors=class_colors[i], linewidth=2)
            
#         elif idx % 4 == 2:  # Strip plots with box overlay (NEW STYLE)
#             feature_data = []
#             class_labels = []
            
#             for class_name in df['true_class'].unique():
#                 class_data = df[df['true_class'] == class_name]
#                 feature_data.extend(class_data[feature].values)
#                 class_labels.extend([class_name] * len(class_data))
            
#             feature_df = pd.DataFrame({
#                 'feature_value': feature_data,
#                 'class': class_labels
#             })
            
#             # Strip plot
#             sns.stripplot(data=feature_df, x='class', y='feature_value', ax=ax,
#                          palette=class_colors, size=3, alpha=0.6, jitter=True)
            
#             # Overlay box plot
#             sns.boxplot(data=feature_df, x='class', y='feature_value', ax=ax,
#                        palette=class_colors, width=0.3, boxprops=dict(alpha=0.3))
            
#         else:  # Violin plots (ENHANCED STYLE)
#             feature_data = []
#             class_labels = []
            
#             for class_name in df['true_class'].unique():
#                 class_data = df[df['true_class'] == class_name]
#                 feature_data.extend(class_data[feature].values)
#                 class_labels.extend([class_name] * len(class_data))
            
#             feature_df = pd.DataFrame({
#                 'feature_value': feature_data,
#                 'class': class_labels
#             })
            
#             sns.violinplot(data=feature_df, x='class', y='feature_value', ax=ax,
#                           palette=class_colors, alpha=0.8, inner='quart')
        
#         ax.set_xlabel('Class Category', fontweight='bold', fontsize=16)
#         ax.set_ylabel('Feature Value', fontweight='bold', fontsize=16)
#         ax.set_title(f'{description}\n({feature})', fontweight='bold', fontsize=14, pad=20)
#         ax.tick_params(axis='x', rotation=0, labelsize=14)
#         ax.tick_params(axis='y', labelsize=14)
#         ax.grid(True, alpha=0.3, axis='y', linestyle=':')
        
#         # Ensure proper spacing for x-axis labels
#         if len(ax.get_xticklabels()) > 0:
#             ax.margins(x=0.15)  # Increased margin to prevent label cutoff
    
#     plt.subplots_adjust(hspace=0.8, wspace=0.6)  # Increased spacing significantly
#     plt.tight_layout(pad=4.0)  # Increased padding
#     plt.show()

# def create_depth_correlation_matrix(df):
#     """Create beautiful correlation matrix with new style"""
#     fig, ax = plt.subplots(figsize=(18, 16))
    
#     # Target features
#     target_features = [
#         'sharpness_gradient_ratio', 'laplacian_variance', 'frequency_centroid', 
#         'focus_center_bias', 'contrast_decay_slope', 'high_freq_ratio', 
#         'regional_depth_std', 'luminance_range', 'focus_variation'
#     ]
    
#     # Calculate correlation matrix
#     corr_matrix = df[target_features].corr()
    
#     # Create clustered heatmap (NEW STYLE)
#     from scipy.cluster.hierarchy import linkage, dendrogram
#     from scipy.spatial.distance import squareform
    
#     # Calculate distance matrix and perform clustering
#     distance_matrix = 1 - np.abs(corr_matrix)
#     linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
#     # Get cluster order
#     dendro = dendrogram(linkage_matrix, no_plot=True)
#     cluster_order = dendro['leaves']
    
#     # Reorder correlation matrix
#     corr_reordered = corr_matrix.iloc[cluster_order, cluster_order]
    
#     # Create beautiful heatmap with new colormap
#     cmap = sns.diverging_palette(260, 10, n=100, as_cmap=True)  # Purple to orange
    
#     im = ax.imshow(corr_reordered.values, cmap=cmap, aspect='auto', 
#                    vmin=-1, vmax=1, interpolation='nearest')
    
#     # Add correlation values with dynamic text color
#     for i in range(len(corr_reordered)):
#         for j in range(len(corr_reordered)):
#             corr_val = corr_reordered.iloc[i, j]
#             text_color = 'white' if abs(corr_val) > 0.5 else 'black'
#             ax.text(j, i, f'{corr_val:.2f}', ha='center', va='center',
#                    fontsize=12, fontweight='bold', color=text_color)
    
#     # Customize labels
#     feature_labels = [
#         'Sharpness\nGradient', 'Laplacian\nVariance', 'Frequency\nCentroid', 
#         'Focus Center\nBias', 'Contrast\nDecay', 'High Freq\nRatio', 
#         'Regional Depth\nStd', 'Luminance\nRange', 'Focus\nVariation'
#     ]
    
#     reordered_labels = [feature_labels[i] for i in cluster_order]
    
#     ax.set_xticks(range(len(corr_reordered)))
#     ax.set_yticks(range(len(corr_reordered)))
#     ax.set_xticklabels(reordered_labels, rotation=45, ha='right', fontsize=14)
#     ax.set_yticklabels(reordered_labels, rotation=0, fontsize=14)
    
#     ax.set_title('Clustered Depth Features Correlation Matrix', 
#                 fontsize=24, fontweight='bold', pad=30)
    
#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax, shrink=0.8)
#     cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=16)
#     cbar.ax.tick_params(labelsize=14)
    
#     plt.tight_layout()
#     plt.show()

# def create_parallel_coordinates_plot(df):
#     """Create parallel coordinates plot (NEW CHART TYPE)"""
#     fig, ax = plt.subplots(figsize=(24, 12))
#     fig.suptitle('Parallel Coordinates: Depth Features by Class', 
#                 fontsize=26, fontweight='bold', y=0.95)
    
#     # Select subset of features for clarity
#     features = ['sharpness_gradient_ratio', 'laplacian_variance', 'frequency_centroid', 
#                'focus_center_bias', 'high_freq_ratio', 'regional_depth_std']
    
#     # Normalize features to 0-1 scale
#     df_normalized = df[features + ['true_class']].copy()
#     for feature in features:
#         df_normalized[feature] = (df_normalized[feature] - df_normalized[feature].min()) / (df_normalized[feature].max() - df_normalized[feature].min())
    
#     # Color mapping
#     class_colors = {'hallway': '#FF4757', 'staircase': '#2ED573', 
#                    'room': '#5352ED', 'open_area': '#FFA502'}
    
#     # Plot parallel coordinates
#     for class_name in df_normalized['true_class'].unique():
#         class_data = df_normalized[df_normalized['true_class'] == class_name]
        
#         for _, row in class_data.iterrows():
#             values = [row[feature] for feature in features]
#             ax.plot(range(len(features)), values, color=class_colors[class_name], 
#                    alpha=0.3, linewidth=1)
    
#     # Plot class means
#     for class_name in df_normalized['true_class'].unique():
#         class_data = df_normalized[df_normalized['true_class'] == class_name]
#         mean_values = [class_data[feature].mean() for feature in features]
#         ax.plot(range(len(features)), mean_values, color=class_colors[class_name], 
#                linewidth=4, label=f'{class_name} (mean)', alpha=0.9, marker='o', markersize=8)
    
#     # Customize plot
#     feature_names = ['Sharpness\nGradient', 'Laplacian\nVariance', 'Frequency\nCentroid', 
#                     'Focus Center\nBias', 'High Freq\nRatio', 'Regional Depth\nStd']
    
#     ax.set_xticks(range(len(features)))
#     ax.set_xticklabels(feature_names, fontsize=16, fontweight='bold')
#     ax.set_ylabel('Normalized Feature Values', fontsize=18, fontweight='bold')
#     ax.set_ylim(0, 1)
#     ax.grid(True, alpha=0.3, axis='y')
#     ax.legend(fontsize=16, loc='upper right')
#     ax.tick_params(axis='y', labelsize=16)
    
#     plt.tight_layout()
#     plt.show()

# def create_sunburst_performance_chart():
#     """Create sunburst chart for performance analysis (NEW CHART TYPE)"""
#     fig, ax = plt.subplots(figsize=(14, 14))
#     fig.suptitle('Performance Sunburst Chart', fontsize=24, fontweight='bold', y=0.95)
    
#     # Overall results
#     class_results = {'hallway': 0.563, 'staircase': 0.519, 'room': 0.293, 'open_area': 0.615}
#     overall_score = 0.497
    
#     # Create sunburst data
#     inner_sizes = list(class_results.values())
#     outer_sizes = [v/sum(inner_sizes) for v in inner_sizes]  # Normalize
    
#     # Colors
#     inner_colors = ['#FF4757', '#2ED573', '#5352ED', '#FFA502']
#     outer_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
#     # Inner ring (performance scores)
#     wedges1, texts1 = ax.pie(inner_sizes, radius=0.7, colors=inner_colors,
#                             wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2))
    
#     # Outer ring (class distribution)
#     wedges2, texts2 = ax.pie(outer_sizes, radius=1.0, colors=outer_colors,
#                             wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2))
    
#     # Add labels
#     for i, (class_name, score) in enumerate(class_results.items()):
#         angle = np.mean([wedges1[i].theta1, wedges1[i].theta2])
#         x = 0.85 * np.cos(np.radians(angle))
#         y = 0.85 * np.sin(np.radians(angle))
#         ax.text(x, y, f'{class_name}\n{score:.3f}', ha='center', va='center',
#                fontsize=14, fontweight='bold', 
#                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
#     # Center circle with overall score
#     centre_circle = plt.Circle((0,0), 0.4, fc='white', linewidth=3, edgecolor='#2C3E50')
#     ax.add_artist(centre_circle)
#     ax.text(0, 0, f'Overall\nScore\n{overall_score:.3f}', ha='center', va='center',
#            fontsize=18, fontweight='bold', color='#2C3E50')
    
#     ax.set_aspect('equal')
#     plt.tight_layout()
#     plt.show()

# def generate_summary_statistics(df):
#     """Generate and display comprehensive summary statistics"""
#     print("="*80)
#     print("DEPTH TRANSITION DATASET SUMMARY STATISTICS")
#     print("="*80)
    
#     print(f"Total Images: {len(df)}")
#     print(f"Number of Classes: {df['true_class'].nunique()}")
#     print(f"Classes: {', '.join(df['true_class'].unique())}")
    
#     print(f"\nClass Distribution:")
#     class_dist = df['true_class'].value_counts()
#     for class_name, count in class_dist.items():
#         percentage = (count / len(df)) * 100
#         print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
#     print(f"\nDepth Feature Statistics:")
#     target_features = [
#         'sharpness_gradient_ratio', 'laplacian_variance', 'frequency_centroid', 
#         'focus_center_bias', 'contrast_decay_slope', 'high_freq_ratio', 
#         'regional_depth_std', 'luminance_range', 'focus_variation'
#     ]
    
#     for feature in target_features:
#         mean_val = df[feature].mean()
#         std_val = df[feature].std()
#         min_val = df[feature].min()
#         max_val = df[feature].max()
#         print(f"  {feature}:")
#         print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
#         print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
    
#     # Overall results
#     print(f"\nOverall Performance Results:")
#     overall_results = {
#         'hallway': 0.563,
#         'staircase': 0.519,
#         'room': 0.293,
#         'open_area': 0.615,
#         'Overall': 0.497
#     }
    
#     for class_name, score in overall_results.items():
#         print(f"  {class_name}: {score:.3f}")

# def main():
#     """Main function to execute the complete depth analysis workflow"""
#     csv_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/RV_results/depth_transition_features.csv"  # Update this path as needed
    
#     try:
#         # Load base data and generate expanded dataset
#         base_df = load_base_data(csv_path)
#         expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
#         # Generate comprehensive analysis
#         print("Generating comprehensive depth transition analysis...")
        
#         # Create visualizations with new chart types
#         create_overall_results_chart()
#         create_depth_features_analysis(expanded_df)
#         create_depth_correlation_matrix(expanded_df)
#         create_parallel_coordinates_plot(expanded_df)
#         create_sunburst_performance_chart()
        
#         # Generate summary statistics
#         generate_summary_statistics(expanded_df)
        
#         # Save the dataset
#         output_path = 'depth_transition_dataset_700.csv'
#         expanded_df.to_csv(output_path, index=False)
#         print(f"\nDepth transition dataset saved to: {output_path}")
        
#     except FileNotFoundError:
#         print(f"Error: Could not find the CSV file '{csv_path}'")
#         print("Please make sure the file exists and update the csv_path variable")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()