

# # visualizations for RCVDA:

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
#                     variation_factor = 0.08  # Smaller variation for color features
#                     variation = np.random.normal(0, std_val * variation_factor)
#                     base_sample[col] = max(0, base_sample[col] + variation)
            
#             # Generate new image identifiers
#             base_sample['image_path'] = f"color_dataset/{class_name.lower()}/image_{i+1:04d}.jpg"
#             base_sample['image_name'] = f"{class_name.lower()}_color_{i+1:04d}.jpg"
#             base_sample['true_class'] = class_name
            
#             expanded_data.append(base_sample)
    
#     expanded_df = pd.DataFrame(expanded_data)
#     return expanded_df

# def create_stunning_class_distribution_charts(df):
#     """Create multiple beautiful class distribution visualizations"""
#     fig, axes = plt.subplots(2, 2, figsize=(22, 18))
#     fig.suptitle('Color Dataset Class Distribution Analysis', fontsize=22, fontweight='bold', y=0.95)
    
#     class_counts = df['true_class'].value_counts()
#     stunning_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
#     # 1. Horizontal Bar Chart with Gradient
#     bars1 = axes[0, 0].barh(range(len(class_counts)), class_counts.values, 
#                            color=stunning_colors[:len(class_counts)], 
#                            edgecolor='white', linewidth=3, alpha=0.9, height=0.6)
#     axes[0, 0].set_xlabel('Number of Images', fontweight='bold', fontsize=10)
#     axes[0, 0].set_ylabel('Class Category', fontweight='bold', fontsize=10)
#     axes[0, 0].set_title('Horizontal Class Distribution', fontweight='bold', fontsize=10, pad=25)
#     axes[0, 0].set_yticks(range(len(class_counts)))
#     axes[0, 0].set_yticklabels(class_counts.index, fontsize=14)
#     axes[0, 0].grid(True, alpha=0.3, axis='x')
    
#     # Add value labels
#     for i, bar in enumerate(bars1):
#         width = bar.get_width()
#         axes[0, 0].text(width + 5, bar.get_y() + bar.get_height()/2,
#                        f'{int(width)}', ha='left', va='center', 
#                        fontweight='bold', fontsize=14, color='#2C3E50')
    
#     # 2. Donut Chart
#     wedges, texts, autotexts = axes[0, 1].pie(class_counts.values, labels=class_counts.index,
#                                              autopct='%1.1f%%', colors=stunning_colors[:len(class_counts)],
#                                              startangle=90, explode=[0.1]*len(class_counts),
#                                              textprops={'fontsize': 14, 'fontweight': 'bold'},
#                                              wedgeprops={'edgecolor': 'white', 'linewidth': 3})
    
#     # Create donut effect
#     centre_circle = plt.Circle((0,0), 0.40, fc='white', linewidth=3, edgecolor='#2C3E50')
#     axes[0, 1].add_artist(centre_circle)
#     axes[0, 1].set_title('Class Distribution Donut Chart', fontweight='bold', fontsize=18, pad=25)
    
#     # Make percentage text more visible
#     for autotext in autotexts:
#         autotext.set_color('white')
#         autotext.set_fontweight('bold')
#         autotext.set_fontsize(13)
    
#     # 3. Stacked Bar Chart (Cumulative)
#     cumulative_values = np.cumsum(class_counts.values)
#     bars3 = axes[1, 0].bar(['Total Dataset'], [cumulative_values[-1]], 
#                           color='lightgray', edgecolor='white', linewidth=3, alpha=0.3)
    
#     bottom = 0
#     for i, (class_name, count) in enumerate(class_counts.items()):
#         axes[1, 0].bar(['Total Dataset'], [count], bottom=bottom,
#                       color=stunning_colors[i], edgecolor='white', linewidth=2,
#                       label=f'{class_name} ({count})', alpha=0.9)
        
#         # Add text in the middle of each segment
#         axes[1, 0].text(0, bottom + count/2, f'{class_name}\n{count}', 
#                        ha='center', va='center', fontweight='bold', 
#                        fontsize=12, color='white')
#         bottom += count
    
#     axes[1, 0].set_ylabel('Number of Images', fontweight='bold', fontsize=16)
#     axes[1, 0].set_title('Stacked Class Distribution', fontweight='bold', fontsize=18, pad=25)
#     axes[1, 0].legend(fontsize=12, loc='upper right', frameon=True, fancybox=True)
    
#     # 4. Polar Bar Chart
#     theta = np.linspace(0.0, 2 * np.pi, len(class_counts), endpoint=False)
#     ax_polar = plt.subplot(2, 2, 4, projection='polar')
#     bars4 = ax_polar.bar(theta, class_counts.values, width=0.8, 
#                         color=stunning_colors[:len(class_counts)], 
#                         alpha=0.8, edgecolor='white', linewidth=2)
    
#     ax_polar.set_xticks(theta)
#     ax_polar.set_xticklabels(class_counts.index, fontsize=12, fontweight='bold')
#     ax_polar.set_title('Polar Class Distribution', fontweight='bold', fontsize=18, pad=30)
#     ax_polar.grid(True, alpha=0.3)
    
#     plt.subplots_adjust(hspace=0.4, wspace=0.3)
#     plt.tight_layout()
#     plt.show()

# def create_rgb_analysis_charts(df):
#     """Create RGB color analysis across different image sections"""
#     fig, axes = plt.subplots(2, 3, figsize=(22, 14))
#     fig.suptitle('RGB Color Analysis Across Image Sections', fontsize=15, fontweight='bold', y=0.95)
    
#     sections = ['top', 'middle', 'bottom']
#     rgb_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
#     # 1-3. Line plots for RGB means across sections
#     for i, color_channel in enumerate(['r', 'g', 'b']):
#         for j, class_name in enumerate(df['true_class'].unique()):
#             class_data = df[df['true_class'] == class_name]
#             means = [class_data[f'{section}_mean_{color_channel}'].mean() for section in sections]
            
#             axes[0, i].plot(sections, means, marker='o', linewidth=4, markersize=10,
#                            label=class_name, color=plt.cm.Set1(j), alpha=0.9)
        
#         axes[0, i].set_xlabel('Image Section', fontweight='bold', fontsize=10)
#         axes[0, i].set_ylabel(f'{color_channel.upper()} Channel Mean', fontweight='bold', fontsize=14)
#         axes[0, i].set_title(f'{color_channel.upper()} Channel Distribution', fontweight='bold', fontsize=16)
#         axes[0, i].legend(fontsize=12, frameon=True, fancybox=True)
#         axes[0, i].grid(True, alpha=0.3)
#         axes[0, i].set_ylim(0, 255)
    
#     # 4. Saturation comparison across sections
#     saturation_data = []
#     section_labels = []
#     class_labels = []
    
#     for section in sections:
#         for class_name in df['true_class'].unique():
#             class_data = df[df['true_class'] == class_name]
#             saturation_data.extend(class_data[f'{section}_saturation_mean'].values)
#             section_labels.extend([section.capitalize()] * len(class_data))
#             class_labels.extend([class_name] * len(class_data))
    
#     saturation_df = pd.DataFrame({
#         'saturation': saturation_data,
#         'section': section_labels,
#         'class': class_labels
#     })
    
#     # Violin plot for saturation
#     sns.violinplot(data=saturation_df, x='section', y='saturation', hue='class',
#                    ax=axes[1, 0], palette='Set2', alpha=0.8)
#     axes[1, 0].set_xlabel('Image Section', fontweight='bold', fontsize=10)
#     axes[1, 0].set_ylabel('Saturation Mean', fontweight='bold', fontsize=10)
#     axes[1, 0].set_title('Saturation Distribution by Section', fontweight='bold', fontsize=16)
#     axes[1, 0].legend(fontsize=12)
    
#     # 5. Color entropy comparison
#     entropy_features = ['top_color_entropy', 'middle_color_entropy', 'bottom_color_entropy']
#     entropy_means = [df[feature].mean() for feature in entropy_features]
    
#     bars5 = axes[1, 1].bar(sections, entropy_means, 
#                           color=['#FF9A9E', '#FECFEF', '#FC466B'], 
#                           edgecolor='white', linewidth=3, alpha=0.9)
#     axes[1, 1].set_xlabel('Image Section', fontweight='bold', fontsize=10)
#     axes[1, 1].set_ylabel('Average Color Entropy', fontweight='bold', fontsize=14)
#     axes[1, 1].set_title('Color Entropy by Section', fontweight='bold', fontsize=16)
#     axes[1, 1].grid(True, alpha=0.3, axis='y')
    
#     # Add value labels
#     for bar in bars5:
#         height = bar.get_height()
#         axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
#                        f'{height:.2f}', ha='center', va='bottom', 
#                        fontweight='bold', fontsize=12, color='#2C3E50')
    
#     # 6. Color variance heatmap
#     variance_features = ['top_color_variance', 'middle_color_variance', 'bottom_color_variance']
#     variance_by_class = df.groupby('true_class')[variance_features].mean()
    
#     # Rename columns for better display
#     variance_by_class.columns = ['Top', 'Middle', 'Bottom']
    
#     im = axes[1, 2].imshow(variance_by_class.values, cmap='viridis', aspect='auto')
#     axes[1, 2].set_xticks(range(len(variance_by_class.columns)))
#     axes[1, 2].set_yticks(range(len(variance_by_class.index)))
#     axes[1, 2].set_xticklabels(variance_by_class.columns, fontsize=12)
#     axes[1, 2].set_yticklabels(variance_by_class.index, fontsize=12)
#     axes[1, 2].set_title('Color Variance Heatmap', fontweight='bold', fontsize=16)
    
#     # Add text annotations
#     for i in range(len(variance_by_class.index)):
#         for j in range(len(variance_by_class.columns)):
#             text = axes[1, 2].text(j, i, f'{variance_by_class.iloc[i, j]:.0f}',
#                                  ha="center", va="center", color="white", fontweight='bold')
    
#     plt.tight_layout()
#     plt.show()

# def create_advanced_color_correlation_matrix(df):
#     """Create beautiful correlation matrix for color features"""
#     fig, ax = plt.subplots(figsize=(16, 12))
    
#     # Select comprehensive color features for correlation analysis
#     color_features = [
#         'global_color_entropy', 'global_color_variance', 'global_color_uniformity',
#         'color_temperature_score', 'natural_color_score', 'artificial_color_score',
#         'top_saturation_mean', 'middle_saturation_mean', 'bottom_saturation_mean',
#         'top_value_mean', 'middle_value_mean', 'bottom_value_mean',
#         'vertical_gradient_smoothness', 'horizontal_gradient_smoothness',
#         'top_middle_color_correlation', 'middle_bottom_color_correlation',
#         'overall_color_transition', 'max_min_variance_ratio'
#     ]
    
#     # Calculate correlation matrix
#     corr_matrix = df[color_features].corr()
    
#     # Create beautiful heatmap with custom colormap
#     mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
#     # Custom colormap for beautiful visualization
#     cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
#     # Create heatmap
#     heatmap = sns.heatmap(corr_matrix, 
#                          mask=mask,
#                          annot=True, 
#                          cmap=cmap,
#                          center=0,
#                          square=True,
#                          fmt='.2f',
#                          cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
#                          annot_kws={'fontsize': 9, 'fontweight': 'bold'},
#                          linewidths=1,
#                          linecolor='white')
    
#     # Customize the plot
#     ax.set_title('Color Features Correlation Matrix', fontsize=20, fontweight='bold', pad=25)
#     ax.set_xlabel('Color Features', fontsize=16, fontweight='bold')
#     ax.set_ylabel('Color Features', fontsize=16, fontweight='bold')
    
#     # Rotate labels for better readability
#     plt.xticks(rotation=45, ha='right', fontsize=11)
#     plt.yticks(rotation=0, fontsize=11)
    
#     plt.tight_layout()
#     plt.show()

# def create_temperature_and_nature_analysis(df):
#     """Create analysis for color temperature and natural vs artificial scoring"""
#     fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#     fig.suptitle('Color Temperature & Nature Analysis', fontsize=22, fontweight='bold', y=0.95)
    
#     class_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
#     # 1. Color Temperature Distribution by Class
#     for i, class_name in enumerate(df['true_class'].unique()):
#         class_data = df[df['true_class'] == class_name]
#         axes[0, 0].hist(class_data['color_temperature_score'], bins=20, alpha=0.7,
#                        label=class_name, color=class_colors[i], edgecolor='white')
    
#     axes[0, 0].set_xlabel('Color Temperature Score', fontweight='bold', fontsize=14)
#     axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=14)
#     axes[0, 0].set_title('Color Temperature Distribution by Class', fontweight='bold', fontsize=16)
#     axes[0, 0].legend(fontsize=12)
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # 2. Natural vs Artificial Color Scatter Plot
#     for i, class_name in enumerate(df['true_class'].unique()):
#         class_data = df[df['true_class'] == class_name]
#         axes[0, 1].scatter(class_data['natural_color_score'], 
#                           class_data['artificial_color_score'],
#                           label=class_name, alpha=0.7, s=80, 
#                           color=class_colors[i], edgecolors='white', linewidth=1)
    
#     axes[0, 1].set_xlabel('Natural Color Score', fontweight='bold', fontsize=14)
#     axes[0, 1].set_ylabel('Artificial Color Score', fontweight='bold', fontsize=14)
#     axes[0, 1].set_title('Natural vs Artificial Color Scoring', fontweight='bold', fontsize=16)
#     axes[0, 1].legend(fontsize=12)
#     axes[0, 1].grid(True, alpha=0.3)
    
#     # 3. Box Plot for Color Uniformity
#     uniformity_data = []
#     class_labels = []
#     for class_name in df['true_class'].unique():
#         class_data = df[df['true_class'] == class_name]
#         uniformity_data.extend(class_data['global_color_uniformity'].values)
#         class_labels.extend([class_name] * len(class_data))
    
#     uniformity_df = pd.DataFrame({
#         'uniformity': uniformity_data,
#         'class': class_labels
#     })
    
#     sns.boxplot(data=uniformity_df, x='class', y='uniformity', ax=axes[1, 0],
#                palette=class_colors, width=0.6)
#     axes[1, 0].set_xlabel('Class Category', fontweight='bold', fontsize=14)
#     axes[1, 0].set_ylabel('Global Color Uniformity', fontweight='bold', fontsize=14)
#     axes[1, 0].set_title('Color Uniformity Distribution', fontweight='bold', fontsize=16)
#     axes[1, 0].tick_params(axis='x', rotation=45)
    
#     # 4. Gradient Smoothness Comparison
#     gradient_features = ['vertical_gradient_smoothness', 'horizontal_gradient_smoothness']
#     gradient_means = df.groupby('true_class')[gradient_features].mean()
    
#     x = np.arange(len(gradient_means.index))
#     width = 0.35
    
#     bars1 = axes[1, 1].bar(x - width/2, gradient_means['vertical_gradient_smoothness'], 
#                           width, label='Vertical Gradient', color='#FF6B6B', 
#                           edgecolor='white', linewidth=2, alpha=0.9)
#     bars2 = axes[1, 1].bar(x + width/2, gradient_means['horizontal_gradient_smoothness'], 
#                           width, label='Horizontal Gradient', color='#4ECDC4', 
#                           edgecolor='white', linewidth=2, alpha=0.9)
    
#     axes[1, 1].set_xlabel('Class Category', fontweight='bold', fontsize=14)
#     axes[1, 1].set_ylabel('Gradient Smoothness', fontweight='bold', fontsize=14)
#     axes[1, 1].set_title('Gradient Smoothness Comparison', fontweight='bold', fontsize=16)
#     axes[1, 1].set_xticks(x)
#     axes[1, 1].set_xticklabels(gradient_means.index, rotation=45, ha='right')
#     axes[1, 1].legend(fontsize=12)
#     axes[1, 1].grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     plt.show()

# def create_transition_analysis_charts(df):
#     """Create analysis for color transitions and correlations"""
#     fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#     fig.suptitle('Color Transition & Correlation Analysis', fontsize=22, fontweight='bold', y=0.95)
    
#     # 1. Radar Chart for Transition Features
#     transition_features = ['top_middle_color_correlation', 'middle_bottom_color_correlation', 
#                           'top_bottom_color_correlation', 'overall_color_transition']
    
#     # Calculate means for each class
#     class_means = df.groupby('true_class')[transition_features].mean()
    
#     # Radar chart setup
#     angles = np.linspace(0, 2 * np.pi, len(transition_features), endpoint=False).tolist()
#     angles += angles[:1]  # Complete the circle
    
#     ax_radar = plt.subplot(2, 2, 1, projection='polar')
    
#     colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
#     for i, (class_name, values) in enumerate(class_means.iterrows()):
#         values_list = values.tolist()
#         values_list += values_list[:1]  # Complete the circle
        
#         ax_radar.plot(angles, values_list, 'o-', linewidth=3, 
#                      label=class_name, color=colors[i], alpha=0.8)
#         ax_radar.fill(angles, values_list, alpha=0.25, color=colors[i])
    
#     ax_radar.set_xticks(angles[:-1])
#     ax_radar.set_xticklabels(['Top-Middle\nCorrelation', 'Middle-Bottom\nCorrelation', 
#                              'Top-Bottom\nCorrelation', 'Overall\nTransition'], fontsize=10)
#     ax_radar.set_title('Color Transition Radar Chart', fontweight='bold', fontsize=16, pad=30)
#     ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
#     ax_radar.grid(True, alpha=0.3)
    
#     # 2. Area Chart for Variance Ratios
#     variance_features = ['top_middle_variance_ratio', 'middle_bottom_variance_ratio', 'max_min_variance_ratio']
#     variance_means = df.groupby('true_class')[variance_features].mean()
    
#     x = range(len(variance_means.index))
#     colors_area = ['#FF9A9E', '#FECFEF', '#FC466B']
    
#     bottom = np.zeros(len(variance_means.index))
#     for i, feature in enumerate(variance_features):
#         axes[0, 1].fill_between(x, bottom, bottom + variance_means[feature], 
#                                alpha=0.8, color=colors_area[i], 
#                                label=feature.replace('_', ' ').title())
#         bottom += variance_means[feature]
    
#     axes[0, 1].set_xlabel('Class Category', fontweight='bold', fontsize=14)
#     axes[0, 1].set_ylabel('Variance Ratio', fontweight='bold', fontsize=14)
#     axes[0, 1].set_title('Stacked Variance Ratios by Class', fontweight='bold', fontsize=10)
#     axes[0, 1].set_xticks(x)
#     axes[0, 1].set_xticklabels(variance_means.index, rotation=45, ha='right')
#     axes[0, 1].legend(fontsize=12)
#     axes[0, 1].grid(True, alpha=0.3, axis='y')
    
#     # 3. Stream Graph for Color Percentages
#     percentage_features = ['top_blue_percentage', 'top_green_percentage',
#                           'middle_blue_percentage', 'middle_green_percentage',
#                           'bottom_blue_percentage', 'bottom_green_percentage']
    
#     percentage_data = []
#     sections = ['Top', 'Middle', 'Bottom']
#     colors_stream = ['#3498db', '#27ae60']  # Blue and Green
    
#     for section in sections:
#         blue_col = f'{section.lower()}_blue_percentage'
#         green_col = f'{section.lower()}_green_percentage'
#         percentage_data.append([df[blue_col].mean(), df[green_col].mean()])
    
#     percentage_array = np.array(percentage_data).T
    
#     axes[1, 0].stackplot(sections, percentage_array[0], percentage_array[1],
#                         labels=['Blue %', 'Green %'], colors=colors_stream, alpha=0.8)
#     axes[1, 0].set_xlabel('Image Section', fontweight='bold', fontsize=10)
#     axes[1, 0].set_ylabel('Average Percentage', fontweight='bold', fontsize=10)
#     axes[1, 0].set_title('Blue & Green Percentages Across Sections', fontweight='bold', fontsize=16)
#     axes[1, 0].legend(fontsize=12)
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # 4. 3D-style Bar Chart for Dominant Color Counts
#     dominant_features = ['top_dominant_color_count', 'middle_dominant_color_count', 'bottom_dominant_color_count']
#     dominant_means = df.groupby('true_class')[dominant_features].mean()
    
#     x = np.arange(len(dominant_means.index))
#     y = np.arange(len(dominant_features))
#     X, Y = np.meshgrid(x, y)
    
#     # Flatten for bar3d
#     x_flat = X.flatten()
#     y_flat = Y.flatten()
#     z_flat = np.zeros_like(x_flat)
#     dx = dy = 0.6
#     dz = dominant_means.values.T.flatten()
    
#     # Create 3D effect with multiple bar layers
#     colors_3d = ['#FF6B6B', '#4ECDC4', '#45B7D1']
#     for i in range(len(dominant_features)):
#         section_data = dominant_means.iloc[:, i].values
#         axes[1, 1].bar(x + i*0.25, section_data, width=0.2, 
#                       color=colors_3d[i], alpha=0.8, 
#                       label=dominant_features[i].replace('_', ' ').title(),
#                       edgecolor='white', linewidth=1)
    
#     axes[1, 1].set_xlabel('Class Category', fontweight='bold', fontsize=14)
#     axes[1, 1].set_ylabel('Dominant Color Count', fontweight='bold', fontsize=14)
#     axes[1, 1].set_title('Dominant Color Counts by Section', fontweight='bold', fontsize=16)
#     axes[1, 1].set_xticks(x + 0.25)
#     axes[1, 1].set_xticklabels(dominant_means.index, rotation=45, ha='right')
#     axes[1, 1].legend(fontsize=11)
#     axes[1, 1].grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     plt.show()

# def create_comprehensive_feature_overview(df):
#     """Create comprehensive overview of all major feature categories"""
#     fig = plt.figure(figsize=(24, 18))
#     fig.suptitle('Comprehensive Color Feature Analysis Overview', fontsize=24, fontweight='bold', y=0.95)
    
#     # Create a complex grid layout
#     gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
#     # 1. RGB Mean Values Heatmap (spans 2x2)
#     ax1 = fig.add_subplot(gs[0:2, 0:2])
#     rgb_features = []
#     for section in ['top', 'middle', 'bottom']:
#         for color in ['r', 'g', 'b']:
#             rgb_features.append(f'{section}_mean_{color}')
    
#     rgb_data = df.groupby('true_class')[rgb_features].mean()
#     rgb_data.columns = [col.replace('_mean_', ' ').replace('_', ' ').title() for col in rgb_data.columns]
    
#     im1 = ax1.imshow(rgb_data.values, cmap='RdYlBu_r', aspect='auto')
#     ax1.set_xticks(range(len(rgb_data.columns)))
#     ax1.set_yticks(range(len(rgb_data.index)))
#     ax1.set_xticklabels(rgb_data.columns, rotation=45, ha='right', fontsize=10)
#     ax1.set_yticklabels(rgb_data.index, fontsize=12)
#     ax1.set_title('RGB Values Across Sections & Classes', fontweight='bold', fontsize=16)
    
#     # Add colorbar
#     cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
#     cbar1.set_label('RGB Value', fontweight='bold')
    
#     # 2. Global Features Comparison (top right)
#     ax2 = fig.add_subplot(gs[0, 2:])
#     global_features = ['global_color_entropy', 'global_color_variance', 'global_color_uniformity']
#     global_normalized = df.groupby('true_class')[global_features].mean()
    
#     # Normalize for better comparison
#     for col in global_normalized.columns:
#         global_normalized[col] = (global_normalized[col] - global_normalized[col].min()) / (global_normalized[col].max() - global_normalized[col].min())
    
#     x = np.arange(len(global_normalized.index))
#     width = 0.25
#     colors_global = ['#E74C3C', '#3498DB', '#2ECC71']
    
#     for i, (feature, color) in enumerate(zip(global_features, colors_global)):
#         ax2.bar(x + i*width, global_normalized[feature], width, 
#                label=feature.replace('_', ' ').title(), color=color, 
#                alpha=0.8, edgecolor='white', linewidth=1)
    
#     ax2.set_xlabel('Class Category', fontweight='bold', fontsize=12)
#     ax2.set_ylabel('Normalized Values', fontweight='bold', fontsize=12)
#     ax2.set_title('Global Color Features (Normalized)', fontweight='bold', fontsize=14)
#     ax2.set_xticks(x + width)
#     ax2.set_xticklabels(global_normalized.index, rotation=45, ha='right')
#     ax2.legend(fontsize=10)
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     # 3. Temperature vs Nature Score (bottom left)
#     ax3 = fig.add_subplot(gs[1, 2:])
#     temp_nature_data = df.groupby('true_class')[['color_temperature_score', 'natural_color_score', 'artificial_color_score']].mean()
    
#     # Create grouped bar chart
#     x = np.arange(len(temp_nature_data.index))
#     width = 0.25
#     colors_temp = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
#     for i, (feature, color) in enumerate(zip(['color_temperature_score', 'natural_color_score', 'artificial_color_score'], colors_temp)):
#         ax3.bar(x + i*width, temp_nature_data[feature], width,
#                label=feature.replace('_', ' ').title(), color=color,
#                alpha=0.8, edgecolor='white', linewidth=1)
    
#     ax3.set_xlabel('Class Category', fontweight='bold', fontsize=12)
#     ax3.set_ylabel('Score Values', fontweight='bold', fontsize=12)
#     ax3.set_title('Temperature & Nature Scores', fontweight='bold', fontsize=14)
#     ax3.set_xticks(x + width)
#     ax3.set_xticklabels(temp_nature_data.index, rotation=45, ha='right')
#     ax3.legend(fontsize=10)
#     ax3.grid(True, alpha=0.3, axis='y')
    
#     # 4. Saturation Trends Line Chart (spans bottom)
#     ax4 = fig.add_subplot(gs[2:, :])
#     sections = ['top', 'middle', 'bottom']
#     class_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
#     for i, class_name in enumerate(df['true_class'].unique()):
#         class_data = df[df['true_class'] == class_name]
#         saturation_means = [class_data[f'{section}_saturation_mean'].mean() for section in sections]
#         value_means = [class_data[f'{section}_value_mean'].mean() for section in sections]
        
#         # Plot saturation
#         ax4.plot(sections, saturation_means, marker='o', linewidth=4, markersize=12,
#                 label=f'{class_name} (Saturation)', color=class_colors[i], alpha=0.8)
        
#         # Plot value (brightness) with dashed line
#         ax4.plot(sections, value_means, marker='s', linewidth=3, markersize=10,
#                 linestyle='--', color=class_colors[i], alpha=0.6,
#                 label=f'{class_name} (Value)')
    
#     ax4.set_xlabel('Image Section', fontweight='bold', fontsize=10)
#     ax4.set_ylabel('Color Saturation & Value', fontweight='bold', fontsize=10)
#     ax4.set_title('Saturation & Value Trends Across Image Sections', fontweight='bold', fontsize=14)
#     ax4.legend(fontsize=12, ncol=2, frameon=True, fancybox=True)
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# def generate_summary_statistics(df):
#     """Generate and display comprehensive summary statistics"""
#     print("="*70)
#     print("COLOR DISTRIBUTION DATASET SUMMARY STATISTICS")
#     print("="*70)
    
#     print(f"Total Images: {len(df)}")
#     print(f"Number of Classes: {df['true_class'].nunique()}")
#     print(f"Classes: {', '.join(df['true_class'].unique())}")
#     print(f"Total Features: {len(df.columns) - 3}")  # Excluding path, name, class
    
#     print(f"\nClass Distribution:")
#     class_dist = df['true_class'].value_counts()
#     for class_name, count in class_dist.items():
#         percentage = (count / len(df)) * 100
#         print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
#     print(f"\nKey Color Feature Statistics:")
#     key_features = ['global_color_entropy', 'global_color_variance', 'global_color_uniformity',
#                    'color_temperature_score', 'natural_color_score', 'artificial_color_score']
    
#     for feature in key_features:
#         mean_val = df[feature].mean()
#         std_val = df[feature].std()
#         min_val = df[feature].min()
#         max_val = df[feature].max()
#         print(f"  {feature}:")
#         print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
#         print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
    
#     print(f"\nRGB Statistics (0-255 scale):")
#     rgb_features = ['top_mean_r', 'top_mean_g', 'top_mean_b', 
#                    'middle_mean_r', 'middle_mean_g', 'middle_mean_b',
#                    'bottom_mean_r', 'bottom_mean_g', 'bottom_mean_b']
    
#     for feature in rgb_features:
#         mean_val = df[feature].mean()
#         print(f"  {feature}: {mean_val:.1f}")

# def main():
#     """Main function to execute the complete color analysis workflow"""
#     csv_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/All_RV_results/color_distribution_features.csv" # Update this path as needed
    
#     try:
#         # Load base data and generate expanded dataset
#         base_df = load_base_data(csv_path)
#         expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
#         # Generate comprehensive analysis
#         print("Generating comprehensive color distribution analysis...")
        
#         # Create all visualizations
#         create_stunning_class_distribution_charts(expanded_df)
#         create_rgb_analysis_charts(expanded_df)
#         create_advanced_color_correlation_matrix(expanded_df)
#         create_temperature_and_nature_analysis(expanded_df)
#         create_transition_analysis_charts(expanded_df)
#         create_comprehensive_feature_overview(expanded_df)
        
#         # Generate summary statistics
#         generate_summary_statistics(expanded_df)
        
#         # Save the dataset
#         output_path = 'color_distribution_dataset_700.csv'
#         expanded_df.to_csv(output_path, index=False)
#         print(f"\nColor distribution dataset saved to: {output_path}")
        
#     except FileNotFoundError:
#         print(f"Error: Could not find the CSV file '{csv_path}'")
#         print("Please make sure the file exists and update the csv_path variable")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()











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
                    variation_factor = 0.08  # Smaller variation for color features
                    variation = np.random.normal(0, std_val * variation_factor)
                    base_sample[col] = max(0, base_sample[col] + variation)
            
            # Generate new image identifiers
            base_sample['image_path'] = f"color_dataset/{class_name.lower()}/image_{i+1:04d}.jpg"
            base_sample['image_name'] = f"{class_name.lower()}_color_{i+1:04d}.jpg"
            base_sample['true_class'] = class_name
            
            expanded_data.append(base_sample)
    
    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df

def create_stunning_class_distribution_charts(df):
    """Create multiple beautiful class distribution visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Color Dataset Class Distribution Analysis', fontsize=26, fontweight='bold', y=0.96)
    
    class_counts = df['true_class'].value_counts()
    stunning_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # 1. Horizontal Bar Chart with Gradient
    bars1 = axes[0, 0].barh(range(len(class_counts)), class_counts.values, 
                           color=stunning_colors[:len(class_counts)], 
                           edgecolor='white', linewidth=3, alpha=0.9, height=0.6)
    axes[0, 0].set_xlabel('Number of Images', fontweight='bold', fontsize=18)
    axes[0, 0].set_ylabel('Class Category', fontweight='bold', fontsize=18)
    axes[0, 0].set_title('Horizontal Class Distribution', fontweight='bold', fontsize=20, pad=30)
    axes[0, 0].set_yticks(range(len(class_counts)))
    axes[0, 0].set_yticklabels(class_counts.index, fontsize=16)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].tick_params(axis='x', labelsize=14)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        axes[0, 0].text(width + 5, bar.get_y() + bar.get_height()/2,
                       f'{int(width)}', ha='left', va='center', 
                       fontweight='bold', fontsize=16, color='#2C3E50')
    
    # 2. Donut Chart
    wedges, texts, autotexts = axes[0, 1].pie(class_counts.values, labels=class_counts.index,
                                             autopct='%1.1f%%', colors=stunning_colors[:len(class_counts)],
                                             startangle=90, explode=[0.1]*len(class_counts),
                                             textprops={'fontsize': 16, 'fontweight': 'bold'},
                                             wedgeprops={'edgecolor': 'white', 'linewidth': 3})
    
    # Create donut effect
    centre_circle = plt.Circle((0,0), 0.40, fc='white', linewidth=3, edgecolor='#2C3E50')
    axes[0, 1].add_artist(centre_circle)
    axes[0, 1].set_title('Class Distribution Donut Chart', fontweight='bold', fontsize=20, pad=30)
    
    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(15)
    
    # 3. Stacked Bar Chart (Cumulative)
    cumulative_values = np.cumsum(class_counts.values)
    bars3 = axes[1, 0].bar(['Total Dataset'], [cumulative_values[-1]], 
                          color='lightgray', edgecolor='white', linewidth=3, alpha=0.3)
    
    bottom = 0
    for i, (class_name, count) in enumerate(class_counts.items()):
        axes[1, 0].bar(['Total Dataset'], [count], bottom=bottom,
                      color=stunning_colors[i], edgecolor='white', linewidth=2,
                      label=f'{class_name} ({count})', alpha=0.9)
        
        # Add text in the middle of each segment
        axes[1, 0].text(0, bottom + count/2, f'{class_name}\n{count}', 
                       ha='center', va='center', fontweight='bold', 
                       fontsize=14, color='white')
        bottom += count
    
    axes[1, 0].set_ylabel('Number of Images', fontweight='bold', fontsize=18)
    axes[1, 0].set_title('Stacked Class Distribution', fontweight='bold', fontsize=20, pad=30)
    axes[1, 0].legend(fontsize=14, loc='upper right', frameon=True, fancybox=True)
    axes[1, 0].tick_params(axis='both', labelsize=14)
    
    # 4. Polar Bar Chart
    theta = np.linspace(0.0, 2 * np.pi, len(class_counts), endpoint=False)
    ax_polar = plt.subplot(2, 2, 4, projection='polar')
    bars4 = ax_polar.bar(theta, class_counts.values, width=0.8, 
                        color=stunning_colors[:len(class_counts)], 
                        alpha=0.8, edgecolor='white', linewidth=2)
    
    ax_polar.set_xticks(theta)
    ax_polar.set_xticklabels(class_counts.index, fontsize=14, fontweight='bold')
    ax_polar.set_title('Polar Class Distribution', fontweight='bold', fontsize=20, pad=40)
    ax_polar.grid(True, alpha=0.3)
    ax_polar.tick_params(axis='y', labelsize=12)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.show()

def create_rgb_analysis_charts(df):
    """Create RGB color analysis across different image sections"""
    fig, axes = plt.subplots(2, 3, figsize=(26, 18))
    fig.suptitle('RGB Color Analysis Across Image Sections', fontsize=26, fontweight='bold', y=0.96)
    
    sections = ['top', 'middle', 'bottom']
    rgb_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1-3. Line plots for RGB means across sections
    for i, color_channel in enumerate(['r', 'g', 'b']):
        for j, class_name in enumerate(df['true_class'].unique()):
            class_data = df[df['true_class'] == class_name]
            means = [class_data[f'{section}_mean_{color_channel}'].mean() for section in sections]
            
            axes[0, i].plot(sections, means, marker='o', linewidth=5, markersize=12,
                           label=class_name, color=plt.cm.Set1(j), alpha=0.9)
        
        axes[0, i].set_xlabel('Image Section', fontweight='bold', fontsize=16)
        axes[0, i].set_ylabel(f'{color_channel.upper()} Channel Mean', fontweight='bold', fontsize=16)
        axes[0, i].set_title(f'{color_channel.upper()} Channel Distribution', fontweight='bold', fontsize=18, pad=25)
        axes[0, i].legend(fontsize=14, frameon=True, fancybox=True)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim(0, 255)
        axes[0, i].tick_params(axis='both', labelsize=14)
    
    # 4. Saturation comparison across sections
    saturation_data = []
    section_labels = []
    class_labels = []
    
    for section in sections:
        for class_name in df['true_class'].unique():
            class_data = df[df['true_class'] == class_name]
            saturation_data.extend(class_data[f'{section}_saturation_mean'].values)
            section_labels.extend([section.capitalize()] * len(class_data))
            class_labels.extend([class_name] * len(class_data))
    
    saturation_df = pd.DataFrame({
        'saturation': saturation_data,
        'section': section_labels,
        'class': class_labels
    })
    
    # Violin plot for saturation
    sns.violinplot(data=saturation_df, x='section', y='saturation', hue='class',
                   ax=axes[1, 0], palette='Set2', alpha=0.8)
    axes[1, 0].set_xlabel('Image Section', fontweight='bold', fontsize=16)
    axes[1, 0].set_ylabel('Saturation Mean', fontweight='bold', fontsize=16)
    axes[1, 0].set_title('Saturation Distribution by Section', fontweight='bold', fontsize=18, pad=25)
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].tick_params(axis='both', labelsize=14)
    
    # 5. Color entropy comparison
    entropy_features = ['top_color_entropy', 'middle_color_entropy', 'bottom_color_entropy']
    entropy_means = [df[feature].mean() for feature in entropy_features]
    
    bars5 = axes[1, 1].bar(sections, entropy_means, 
                          color=['#FF9A9E', '#FECFEF', '#FC466B'], 
                          edgecolor='white', linewidth=3, alpha=0.9)
    axes[1, 1].set_xlabel('Image Section', fontweight='bold', fontsize=16)
    axes[1, 1].set_ylabel('Average Color Entropy', fontweight='bold', fontsize=16)
    axes[1, 1].set_title('Color Entropy by Section', fontweight='bold', fontsize=18, pad=25)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='both', labelsize=14)
    
    # Add value labels
    for bar in bars5:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{height:.2f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=14, color='#2C3E50')
    
    # 6. Color variance heatmap
    variance_features = ['top_color_variance', 'middle_color_variance', 'bottom_color_variance']
    variance_by_class = df.groupby('true_class')[variance_features].mean()
    
    # Rename columns for better display
    variance_by_class.columns = ['Top', 'Middle', 'Bottom']
    
    im = axes[1, 2].imshow(variance_by_class.values, cmap='viridis', aspect='auto')
    axes[1, 2].set_xticks(range(len(variance_by_class.columns)))
    axes[1, 2].set_yticks(range(len(variance_by_class.index)))
    axes[1, 2].set_xticklabels(variance_by_class.columns, fontsize=14)
    axes[1, 2].set_yticklabels(variance_by_class.index, fontsize=14)
    axes[1, 2].set_title('Color Variance Heatmap', fontweight='bold', fontsize=18, pad=25)
    
    # Add text annotations
    for i in range(len(variance_by_class.index)):
        for j in range(len(variance_by_class.columns)):
            text = axes[1, 2].text(j, i, f'{variance_by_class.iloc[i, j]:.0f}',
                                 ha="center", va="center", color="white", fontweight='bold', fontsize=14)
    
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.tight_layout()
    plt.show()

def create_advanced_color_correlation_matrix(df):
    """Create beautiful correlation matrix for color features"""
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Select comprehensive color features for correlation analysis
    color_features = [
        'global_color_entropy', 'global_color_variance', 'global_color_uniformity',
        'color_temperature_score', 'natural_color_score', 'artificial_color_score',
        'top_saturation_mean', 'middle_saturation_mean', 'bottom_saturation_mean',
        'top_value_mean', 'middle_value_mean', 'bottom_value_mean',
        'vertical_gradient_smoothness', 'horizontal_gradient_smoothness',
        'top_middle_color_correlation', 'middle_bottom_color_correlation',
        'overall_color_transition', 'max_min_variance_ratio'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[color_features].corr()
    
    # Create beautiful heatmap with custom colormap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
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
                         annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                         linewidths=1,
                         linecolor='white')
    
    # Customize the plot
    ax.set_title('Color Features Correlation Matrix', fontsize=24, fontweight='bold', pad=30)
    ax.set_xlabel('Color Features', fontsize=18, fontweight='bold')
    ax.set_ylabel('Color Features', fontsize=18, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.show()

def create_temperature_and_nature_analysis(df):
    """Create analysis for color temperature and natural vs artificial scoring"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Color Temperature & Nature Analysis', fontsize=26, fontweight='bold', y=0.96)
    
    class_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
    # 1. Color Temperature Distribution by Class
    for i, class_name in enumerate(df['true_class'].unique()):
        class_data = df[df['true_class'] == class_name]
        axes[0, 0].hist(class_data['color_temperature_score'], bins=20, alpha=0.7,
                       label=class_name, color=class_colors[i], edgecolor='white')
    
    axes[0, 0].set_xlabel('Color Temperature Score', fontweight='bold', fontsize=16)
    axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=16)
    axes[0, 0].set_title('Color Temperature Distribution by Class', fontweight='bold', fontsize=18, pad=25)
    axes[0, 0].legend(fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=14)
    
    # 2. Natural vs Artificial Color Scatter Plot
    for i, class_name in enumerate(df['true_class'].unique()):
        class_data = df[df['true_class'] == class_name]
        axes[0, 1].scatter(class_data['natural_color_score'], 
                          class_data['artificial_color_score'],
                          label=class_name, alpha=0.7, s=80, 
                          color=class_colors[i], edgecolors='white', linewidth=1)
    
    axes[0, 1].set_xlabel('Natural Color Score', fontweight='bold', fontsize=16)
    axes[0, 1].set_ylabel('Artificial Color Score', fontweight='bold', fontsize=16)
    axes[0, 1].set_title('Natural vs Artificial Color Scoring', fontweight='bold', fontsize=18, pad=25)
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=14)
    
    # 3. Box Plot for Color Uniformity
    uniformity_data = []
    class_labels = []
    for class_name in df['true_class'].unique():
        class_data = df[df['true_class'] == class_name]
        uniformity_data.extend(class_data['global_color_uniformity'].values)
        class_labels.extend([class_name] * len(class_data))
    
    uniformity_df = pd.DataFrame({
        'uniformity': uniformity_data,
        'class': class_labels
    })
    
    sns.boxplot(data=uniformity_df, x='class', y='uniformity', ax=axes[1, 0],
               palette=class_colors, width=0.6)
    axes[1, 0].set_xlabel('Class Category', fontweight='bold', fontsize=16)
    axes[1, 0].set_ylabel('Global Color Uniformity', fontweight='bold', fontsize=16)
    axes[1, 0].set_title('Color Uniformity Distribution', fontweight='bold', fontsize=18, pad=25)
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=14)
    axes[1, 0].tick_params(axis='y', labelsize=14)
    
    # 4. Gradient Smoothness Comparison
    gradient_features = ['vertical_gradient_smoothness', 'horizontal_gradient_smoothness']
    gradient_means = df.groupby('true_class')[gradient_features].mean()
    
    x = np.arange(len(gradient_means.index))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, gradient_means['vertical_gradient_smoothness'], 
                          width, label='Vertical Gradient', color='#FF6B6B', 
                          edgecolor='white', linewidth=2, alpha=0.9)
    bars2 = axes[1, 1].bar(x + width/2, gradient_means['horizontal_gradient_smoothness'], 
                          width, label='Horizontal Gradient', color='#4ECDC4', 
                          edgecolor='white', linewidth=2, alpha=0.9)
    
    axes[1, 1].set_xlabel('Class Category', fontweight='bold', fontsize=16)
    axes[1, 1].set_ylabel('Gradient Smoothness', fontweight='bold', fontsize=16)
    axes[1, 1].set_title('Gradient Smoothness Comparison', fontweight='bold', fontsize=18, pad=25)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(gradient_means.index, rotation=45, ha='right', fontsize=14)
    axes[1, 1].legend(fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='y', labelsize=14)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.show()

def create_transition_analysis_charts(df):
    """Create analysis for color transitions and correlations"""
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Color Transition & Correlation Analysis', fontsize=26, fontweight='bold', y=0.96)
    
    # 1. Radar Chart for Transition Features
    transition_features = ['top_middle_color_correlation', 'middle_bottom_color_correlation', 
                          'top_bottom_color_correlation', 'overall_color_transition']
    
    # Calculate means for each class
    class_means = df.groupby('true_class')[transition_features].mean()
    
    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, len(transition_features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax_radar = plt.subplot(2, 2, 1, projection='polar')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (class_name, values) in enumerate(class_means.iterrows()):
        values_list = values.tolist()
        values_list += values_list[:1]  # Complete the circle
        
        ax_radar.plot(angles, values_list, 'o-', linewidth=4, 
                     label=class_name, color=colors[i], alpha=0.8, markersize=8)
        ax_radar.fill(angles, values_list, alpha=0.25, color=colors[i])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(['Top-Middle\nCorrelation', 'Middle-Bottom\nCorrelation', 
                             'Top-Bottom\nCorrelation', 'Overall\nTransition'], fontsize=12)
    ax_radar.set_title('Color Transition Radar Chart', fontweight='bold', fontsize=18, pad=40)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=14)
    ax_radar.grid(True, alpha=0.3)
    ax_radar.tick_params(axis='y', labelsize=12)
    
    # 2. Area Chart for Variance Ratios
    variance_features = ['top_middle_variance_ratio', 'middle_bottom_variance_ratio', 'max_min_variance_ratio']
    variance_means = df.groupby('true_class')[variance_features].mean()
    
    x = range(len(variance_means.index))
    colors_area = ['#FF9A9E', '#FECFEF', '#FC466B']
    
    bottom = np.zeros(len(variance_means.index))
    for i, feature in enumerate(variance_features):
        axes[0, 1].fill_between(x, bottom, bottom + variance_means[feature], 
                               alpha=0.8, color=colors_area[i], 
                               label=feature.replace('_', ' ').title())
        bottom += variance_means[feature]
    
    axes[0, 1].set_xlabel('Class Category', fontweight='bold', fontsize=16)
    axes[0, 1].set_ylabel('Variance Ratio', fontweight='bold', fontsize=16)
    axes[0, 1].set_title('Stacked Variance Ratios by Class', fontweight='bold', fontsize=18, pad=25)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(variance_means.index, rotation=45, ha='right', fontsize=14)
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='y', labelsize=14)
    
    # 3. Stream Graph for Color Percentages
    percentage_features = ['top_blue_percentage', 'top_green_percentage',
                          'middle_blue_percentage', 'middle_green_percentage',
                          'bottom_blue_percentage', 'bottom_green_percentage']
    
    percentage_data = []
    sections = ['Top', 'Middle', 'Bottom']
    colors_stream = ['#3498db', '#27ae60']  # Blue and Green
    
    for section in sections:
        blue_col = f'{section.lower()}_blue_percentage'
        green_col = f'{section.lower()}_green_percentage'
        percentage_data.append([df[blue_col].mean(), df[green_col].mean()])
    
    percentage_array = np.array(percentage_data).T
    
    axes[1, 0].stackplot(sections, percentage_array[0], percentage_array[1],
                        labels=['Blue %', 'Green %'], colors=colors_stream, alpha=0.8)
    axes[1, 0].set_xlabel('Image Section', fontweight='bold', fontsize=16)
    axes[1, 0].set_ylabel('Average Percentage', fontweight='bold', fontsize=16)
    axes[1, 0].set_title('Blue & Green Percentages Across Sections', fontweight='bold', fontsize=18, pad=25)
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=14)
    
    # 4. 3D-style Bar Chart for Dominant Color Counts
    dominant_features = ['top_dominant_color_count', 'middle_dominant_color_count', 'bottom_dominant_color_count']
    dominant_means = df.groupby('true_class')[dominant_features].mean()
    
    x = np.arange(len(dominant_means.index))
    width = 0.25
    
    # Create 3D effect with multiple bar layers
    colors_3d = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i in range(len(dominant_features)):
        section_data = dominant_means.iloc[:, i].values
        axes[1, 1].bar(x + i*width, section_data, width=width, 
                      color=colors_3d[i], alpha=0.8, 
                      label=dominant_features[i].replace('_', ' ').title(),
                      edgecolor='white', linewidth=1)
    
    axes[1, 1].set_xlabel('Class Category', fontweight='bold', fontsize=16)
    axes[1, 1].set_ylabel('Dominant Color Count', fontweight='bold', fontsize=16)
    axes[1, 1].set_title('Dominant Color Counts by Section', fontweight='bold', fontsize=18, pad=25)
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(dominant_means.index, rotation=45, ha='right', fontsize=14)
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='y', labelsize=14)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.show()

def create_comprehensive_feature_overview(df):
    """Create comprehensive overview of all major feature categories"""
    fig = plt.figure(figsize=(28, 22))
    fig.suptitle('Comprehensive Color Feature Analysis Overview', fontsize=28, fontweight='bold', y=0.96)
    
    # Create a complex grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
    
    # 1. RGB Mean Values Heatmap (spans 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    rgb_features = []
    for section in ['top', 'middle', 'bottom']:
        for color in ['r', 'g', 'b']:
            rgb_features.append(f'{section}_mean_{color}')
    
    rgb_data = df.groupby('true_class')[rgb_features].mean()
    rgb_data.columns = [col.replace('_mean_', ' ').replace('_', ' ').title() for col in rgb_data.columns]
    
    im1 = ax1.imshow(rgb_data.values, cmap='RdYlBu_r', aspect='auto')
    ax1.set_xticks(range(len(rgb_data.columns)))
    ax1.set_yticks(range(len(rgb_data.index)))
    ax1.set_xticklabels(rgb_data.columns, rotation=45, ha='right', fontsize=12)
    ax1.set_yticklabels(rgb_data.index, fontsize=14)
    ax1.set_title('RGB Values Across Sections & Classes', fontweight='bold', fontsize=18, pad=20)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('RGB Value', fontweight='bold', fontsize=14)
    
    # 2. Global Features Comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    global_features = ['global_color_entropy', 'global_color_variance', 'global_color_uniformity']
    global_normalized = df.groupby('true_class')[global_features].mean()
    
    # Normalize for better comparison
    for col in global_normalized.columns:
        global_normalized[col] = (global_normalized[col] - global_normalized[col].min()) / (global_normalized[col].max() - global_normalized[col].min())
    
    x = np.arange(len(global_normalized.index))
    width = 0.25
    colors_global = ['#E74C3C', '#3498DB', '#2ECC71']
    
    for i, (feature, color) in enumerate(zip(global_features, colors_global)):
        ax2.bar(x + i*width, global_normalized[feature], width, 
               label=feature.replace('_', ' ').title(), color=color, 
               alpha=0.8, edgecolor='white', linewidth=1)
    
    ax2.set_xlabel('Class Category', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Normalized Values', fontweight='bold', fontsize=14)
    ax2.set_title('Global Color Features (Normalized)', fontweight='bold', fontsize=16, pad=20)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(global_normalized.index, rotation=45, ha='right', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', labelsize=12)
    
    # 3. Temperature vs Nature Score (middle right)
    ax3 = fig.add_subplot(gs[1, 2:])
    temp_nature_data = df.groupby('true_class')[['color_temperature_score', 'natural_color_score', 'artificial_color_score']].mean()
    
    # Create grouped bar chart
    x = np.arange(len(temp_nature_data.index))
    width = 0.25
    colors_temp = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (feature, color) in enumerate(zip(['color_temperature_score', 'natural_color_score', 'artificial_color_score'], colors_temp)):
        ax3.bar(x + i*width, temp_nature_data[feature], width,
               label=feature.replace('_', ' ').title(), color=color,
               alpha=0.8, edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('Class Category', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Score Values', fontweight='bold', fontsize=14)
    ax3.set_title('Temperature & Nature Scores', fontweight='bold', fontsize=16, pad=20)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(temp_nature_data.index, rotation=45, ha='right', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='both', labelsize=12)
    
    # 4. Saturation Trends Line Chart (spans bottom)
    ax4 = fig.add_subplot(gs[2:, :])
    sections = ['top', 'middle', 'bottom']
    class_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
    for i, class_name in enumerate(df['true_class'].unique()):
        class_data = df[df['true_class'] == class_name]
        saturation_means = [class_data[f'{section}_saturation_mean'].mean() for section in sections]
        value_means = [class_data[f'{section}_value_mean'].mean() for section in sections]
        
        # Plot saturation
        ax4.plot(sections, saturation_means, marker='o', linewidth=5, markersize=14,
                label=f'{class_name} (Saturation)', color=class_colors[i], alpha=0.8)
        
        # Plot value (brightness) with dashed line
        ax4.plot(sections, value_means, marker='s', linewidth=4, markersize=12,
                linestyle='--', color=class_colors[i], alpha=0.6,
                label=f'{class_name} (Value)')
    
    ax4.set_xlabel('Image Section', fontweight='bold', fontsize=18)
    ax4.set_ylabel('Color Saturation & Value', fontweight='bold', fontsize=18)
    ax4.set_title('Saturation & Value Trends Across Image Sections', fontweight='bold', fontsize=20, pad=30)
    ax4.legend(fontsize=14, ncol=2, frameon=True, fancybox=True)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    plt.show()

def generate_summary_statistics(df):
    """Generate and display comprehensive summary statistics"""
    print("="*70)
    print("COLOR DISTRIBUTION DATASET SUMMARY STATISTICS")
    print("="*70)
    
    print(f"Total Images: {len(df)}")
    print(f"Number of Classes: {df['true_class'].nunique()}")
    print(f"Classes: {', '.join(df['true_class'].unique())}")
    print(f"Total Features: {len(df.columns) - 3}")  # Excluding path, name, class
    
    print(f"\nClass Distribution:")
    class_dist = df['true_class'].value_counts()
    for class_name, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    print(f"\nKey Color Feature Statistics:")
    key_features = ['global_color_entropy', 'global_color_variance', 'global_color_uniformity',
                   'color_temperature_score', 'natural_color_score', 'artificial_color_score']
    
    for feature in key_features:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"  {feature}:")
        print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
    
    print(f"\nRGB Statistics (0-255 scale):")
    rgb_features = ['top_mean_r', 'top_mean_g', 'top_mean_b', 
                   'middle_mean_r', 'middle_mean_g', 'middle_mean_b',
                   'bottom_mean_r', 'bottom_mean_g', 'bottom_mean_b']
    
    for feature in rgb_features:
        mean_val = df[feature].mean()
        print(f"  {feature}: {mean_val:.1f}")

def main():
    """Main function to execute the complete color analysis workflow"""
    csv_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/RV_results/color_distribution_dataset_700.csv"  
    
    try:
        # Load base data and generate expanded dataset
        base_df = load_base_data(csv_path)
        expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
        # Generate comprehensive analysis
        print("Generating comprehensive color distribution analysis...")
        
        # Create all visualizations
        create_stunning_class_distribution_charts(expanded_df)
        create_rgb_analysis_charts(expanded_df)
        create_advanced_color_correlation_matrix(expanded_df)
        create_temperature_and_nature_analysis(expanded_df)
        create_transition_analysis_charts(expanded_df)
        create_comprehensive_feature_overview(expanded_df)
        
        # Generate summary statistics
        generate_summary_statistics(expanded_df)
        
        # Save the dataset
        output_path = 'color_distribution_dataset_700.csv'
        expanded_df.to_csv(output_path, index=False)
        print(f"\nColor distribution dataset saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file '{csv_path}'")
        print("Please make sure the file exists and update the csv_path variable")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()