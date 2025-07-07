



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
    """
    Generate expanded dataset based on base data patterns
    """
    # Calculate class proportions from base data
    class_counts = base_df['true_class'].value_counts()
    class_proportions = class_counts / len(base_df)
    
    # Calculate target counts for each class
    target_counts = {}
    remaining_samples = target_size
    
    # Ensure each class gets appropriate representation
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
        
        # Calculate statistics for numerical columns
        numerical_cols = class_data.select_dtypes(include=[np.number]).columns
        
        for i in range(target_count):
            # Create new sample based on existing patterns
            base_sample = class_data.sample(1).iloc[0].copy()
            
            # Add natural variation to numerical features
            for col in numerical_cols:
                if col not in ['image_width', 'image_height']:
                    mean_val = class_data[col].mean()
                    std_val = class_data[col].std()
                    variation_factor = 0.1
                    variation = np.random.normal(0, std_val * variation_factor)
                    base_sample[col] = max(0, base_sample[col] + variation)
            
            # Generate new image identifiers
            base_sample['image_path'] = f"dataset_images/{class_name.lower()}/image_{i+1:04d}.jpg"
            base_sample['image_name'] = f"{class_name.lower()}_image_{i+1:04d}.jpg"
            base_sample['true_class'] = class_name
            
            expanded_data.append(base_sample)
    
    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df

def create_comprehensive_bar_charts(df):
    """Create comprehensive bar charts for the dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Edge Orientation Features Analysis', fontsize=20, fontweight='bold', y=0.95)
    
    # Beautiful color palettes
    vibrant_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    gradient_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    # 1. Class distribution
    class_counts = df['true_class'].value_counts()
    bars1 = axes[0, 0].bar(range(len(class_counts)), class_counts.values, 
                          color=vibrant_colors[:len(class_counts)], 
                          edgecolor='white', linewidth=2, alpha=0.9)
    axes[0, 0].set_xlabel('Class Category', fontweight='bold', fontsize=8)
    axes[0, 0].set_ylabel('Number of Images', fontweight='bold', fontsize=14)

    axes[0, 0].set_title('Dataset Class Distribution', fontweight='bold', fontsize=16, pad=20)
    axes[0, 0].set_xticks(range(len(class_counts)))
    axes[0, 0].set_xticklabels(class_counts.index, rotation=45, ha='right', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on bars with beautiful styling
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12, color='#2C3E50')
    
    # 2. Mean edge strength by class
    mean_edge_by_class = df.groupby('true_class')['mean_edge_strength'].mean()
    bars2 = axes[0, 1].bar(range(len(mean_edge_by_class)), mean_edge_by_class.values, 
                          color=gradient_colors[:len(mean_edge_by_class)],
                          edgecolor='white', linewidth=2, alpha=0.9)
    axes[0, 1].set_xlabel('Class Category', fontweight='bold', fontsize=10)
    axes[0, 1].set_ylabel('Mean Edge Strength', fontweight='bold', fontsize=14)
    axes[0, 1].set_title('Average Edge Strength by Class', fontweight='bold', fontsize=16, pad=20)
    axes[0, 1].set_xticks(range(len(mean_edge_by_class)))
    axes[0, 1].set_xticklabels(mean_edge_by_class.index, rotation=45, ha='right', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12, color='#2C3E50')
    
    # 3. Orientation entropy by class
    entropy_by_class = df.groupby('true_class')['orientation_entropy'].mean()
    bars3 = axes[1, 0].bar(range(len(entropy_by_class)), entropy_by_class.values, 
                          color=['#FF9A9E', '#FECFEF', '#FECFEF', '#FC466B'],
                          edgecolor='white', linewidth=2, alpha=0.9)
    axes[1, 0].set_xlabel('Class Category', fontweight='bold', fontsize=10)
    axes[1, 0].set_ylabel('Orientation Entropy', fontweight='bold', fontsize=10)
    axes[1, 0].set_title('Average Orientation Entropy by Class', fontweight='bold', fontsize=10, pad=20)
    axes[1, 0].set_xticks(range(len(entropy_by_class)))
    axes[1, 0].set_xticklabels(entropy_by_class.index, rotation=45, ha='right', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12, color='#2C3E50')
    
    # 4. Edge density distribution histogram with gradient
    n, bins, patches = axes[1, 1].hist(df['edge_density'], bins=30, alpha=0.8, 
                                      edgecolor='white', linewidth=1.5)
    
    # Color gradient for histogram
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.plasma(i / len(patches)))
    
    axes[1, 1].set_xlabel('Edge Density', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Frequency', fontweight='bold', fontsize=14)
    axes[1, 1].set_title('Edge Density Distribution', fontweight='bold', fontsize=10, pad=20)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

def create_detailed_pie_charts(df):
    """Create detailed pie charts for categorical distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Dataset Distribution Analysis', fontsize=20, fontweight='bold', y=0.95)
    
    # Beautiful bright colors
    bright_colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bright_colors2 = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    # 1. Class distribution pie chart
    class_counts = df['true_class'].value_counts()
    
    wedges1, texts1, autotexts1 = axes[0].pie(class_counts.values, labels=class_counts.index, 
                                             autopct='%1.1f%%', colors=bright_colors1[:len(class_counts)], 
                                             startangle=90, explode=[0.05]*len(class_counts),
                                             textprops={'fontsize': 14, 'fontweight': 'bold'},
                                             wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title('Class Category Distribution', fontsize=10, fontweight='bold', pad=30)
    
    # Make percentage text more visible
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # 2. Dominant orientation distribution
    dominant_orient_counts = df['dominant_orientation_idx'].value_counts()
    orientation_labels = ['Horizontal', 'Vertical', 'Diagonal 45°', 'Diagonal 135°']
    
    # Map indices to meaningful labels
    mapped_labels = []
    mapped_values = []
    for idx, count in dominant_orient_counts.items():
        if idx < len(orientation_labels):
            mapped_labels.append(orientation_labels[int(idx)])
            mapped_values.append(count)
    
    wedges2, texts2, autotexts2 = axes[1].pie(mapped_values, labels=mapped_labels, 
                                             autopct='%1.1f%%', colors=bright_colors2[:len(mapped_values)], 
                                             startangle=90, explode=[0.05]*len(mapped_values),
                                             textprops={'fontsize': 14, 'fontweight': 'bold'},
                                             wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Dominant Orientation Distribution', fontsize=16, fontweight='bold', pad=30)
    
    # Make percentage text more visible
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    plt.show()

def create_beautiful_correlation_heatmap(df):
    """Create beautiful correlation heatmap like the example shown"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select key numerical features for correlation analysis
    correlation_features = [
        'mean_edge_strength', 'orientation_entropy', 'edge_density', 
        'geometric_regularity', 'pattern_uniformity', 'orientation_concentration',
        'horizontal_ratio', 'vertical_ratio', 'diagonal_45_ratio', 'diagonal_135_ratio'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[correlation_features].corr()
    
    # Create beautiful heatmap with custom colormap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    # Custom colormap for beautiful visualization
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Create heatmap
    heatmap = sns.heatmap(corr_matrix, 
                         mask=mask,
                         annot=True, 
                         cmap=cmap,
                         center=0,
                         square=True,
                         fmt='.2f',
                         cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                         annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                         linewidths=1,
                         linecolor='white')
    
    # Customize the plot
    ax.set_title('Feature Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Features', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def create_advanced_analysis_charts(df):
    """Create advanced analysis charts with bright, beautiful colors"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Advanced Feature Analysis', fontsize=20, fontweight='bold', y=0.95)
    
    # Beautiful color palettes
    vibrant_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    class_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
    # 1. Pattern uniformity by class
    pattern_by_class = df.groupby('true_class')['pattern_uniformity'].mean()
    bars1 = axes[0, 0].bar(range(len(pattern_by_class)), pattern_by_class.values,
                          color=vibrant_colors[:len(pattern_by_class)],
                          edgecolor='white', linewidth=2, alpha=0.9)
    axes[0, 0].set_xlabel('Class Category', fontweight='bold', fontsize=10)
    axes[0, 0].set_ylabel('Pattern Uniformity', fontweight='bold', fontsize=14)
    axes[0, 0].set_title('Pattern Uniformity by Class', fontweight='bold', fontsize=16, pad=20)
    axes[0, 0].set_xticks(range(len(pattern_by_class)))
    axes[0, 0].set_xticklabels(pattern_by_class.index, rotation=45, ha='right', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=11, color='#2C3E50')
    
    # 2. Geometric regularity analysis
    geometric_by_class = df.groupby('true_class')['geometric_regularity'].mean()
    bars2 = axes[0, 1].bar(range(len(geometric_by_class)), geometric_by_class.values,
                          color=['#FF9A9E', '#FECFEF', '#FECFEF', '#FC466B'],
                          edgecolor='white', linewidth=2, alpha=0.9)
    axes[0, 1].set_xlabel('Class Category', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Geometric Regularity', fontweight='bold', fontsize=14)
    axes[0, 1].set_title('Geometric Regularity by Class', fontweight='bold', fontsize=16, pad=20)
    axes[0, 1].set_xticks(range(len(geometric_by_class)))
    axes[0, 1].set_xticklabels(geometric_by_class.index, rotation=45, ha='right', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # 3. Orientation concentration scatter plot
    for i, class_name in enumerate(df['true_class'].unique()):
        class_data = df[df['true_class'] == class_name]
        axes[1, 0].scatter(class_data['orientation_concentration'], 
                          class_data['orientation_entropy'],
                          label=class_name, alpha=0.7, s=80, 
                          color=class_colors[i], edgecolors='white', linewidth=1)
    axes[1, 0].set_xlabel('Orientation Concentration', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylabel('Orientation Entropy', fontweight='bold', fontsize=14)
    axes[1, 0].set_title('Orientation Concentration vs Entropy', fontweight='bold', fontsize=16, pad=20)
    axes[1, 0].legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 4. Diagonal vs horizontal/vertical ratio
    for i, class_name in enumerate(df['true_class'].unique()):
        class_data = df[df['true_class'] == class_name]
        axes[1, 1].scatter(class_data['total_diagonal_ratio'],
                          class_data['horizontal_ratio'] + class_data['vertical_ratio'],
                          label=class_name, alpha=0.7, s=80, 
                          color=class_colors[i], edgecolors='white', linewidth=1)
    axes[1, 1].set_xlabel('Total Diagonal Ratio', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Horizontal + Vertical Ratio', fontweight='bold', fontsize=14)
    axes[1, 1].set_title('Diagonal vs Horizontal/Vertical Ratios', fontweight='bold', fontsize=16, pad=20)
    axes[1, 1].legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

def generate_summary_statistics(df):
    """Generate and display summary statistics"""
    print("="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Total Images: {len(df)}")
    print(f"Number of Classes: {df['true_class'].nunique()}")
    print(f"Classes: {', '.join(df['true_class'].unique())}")
    
    print(f"\nClass Distribution:")
    class_dist = df['true_class'].value_counts()
    for class_name, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    print(f"\nDominant Orientation Distribution:")
    orient_dist = df['dominant_orientation_idx'].value_counts()
    orientation_names = ['Horizontal', 'Vertical', 'Diagonal 45°', 'Diagonal 135°']
    for idx, count in orient_dist.items():
        if idx < len(orientation_names):
            percentage = (count / len(df)) * 100
            print(f"  {orientation_names[int(idx)]}: {count} images ({percentage:.1f}%)")
    
    print(f"\nKey Feature Statistics:")
    key_features = ['mean_edge_strength', 'orientation_entropy', 'edge_density', 
                   'geometric_regularity', 'pattern_uniformity']
    
    for feature in key_features:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"  {feature}:")
        print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")

def main():
    """Main function to execute the complete workflow"""
    csv_path = '/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/All_RV_results/edge_orientation_features_700.csv'  
    
    try:
        # Load base data and generate expanded dataset
        base_df = load_base_data(csv_path)
        expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
        # Generate comprehensive analysis
        print("Generating comprehensive dataset analysis...")
        
        # Create all visualizations
        create_comprehensive_bar_charts(expanded_df)
        create_detailed_pie_charts(expanded_df)
        create_beautiful_correlation_heatmap(expanded_df)
        create_advanced_analysis_charts(expanded_df)
        
        # Generate summary statistics
        generate_summary_statistics(expanded_df)
        
        # Save the dataset
        output_path = 'edge_orientation_features_dataset.csv'
        expanded_df.to_csv(output_path, index=False)
        print(f"\nDataset saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file '{csv_path}'")
        print("Please make sure the file exists and update the csv_path variable")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()