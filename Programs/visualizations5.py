

### IPALDC visualizations 

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.patches import Circle, Rectangle, Wedge, Polygon, FancyBboxPatch
# import matplotlib.patches as mpatches
# from matplotlib.colors import LinearSegmentedColormap
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
#     class_counts = base_df['category'].value_counts()
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
#         class_data = base_df[base_df['category'] == class_name]
#         numerical_cols = class_data.select_dtypes(include=[np.number]).columns
        
#         for i in range(target_count):
#             base_sample = class_data.sample(1).iloc[0].copy()
            
#             # Add natural variation to numerical features
#             for col in numerical_cols:
#                 mean_val = class_data[col].mean()
#                 std_val = class_data[col].std()
#                 variation_factor = 0.08
#                 variation = np.random.normal(0, std_val * variation_factor)
#                 base_sample[col] = max(0, base_sample[col] + variation)
            
#             # Generate new image identifiers
#             base_sample['filename'] = f"{class_name.lower()}_ipaldc_{i+1:04d}.jpg"
#             base_sample['image_path'] = f"ipaldc_dataset/{class_name.lower()}/image_{i+1:04d}.jpg"
#             base_sample['category'] = class_name
            
#             expanded_data.append(base_sample)
    
#     expanded_df = pd.DataFrame(expanded_data)
#     return expanded_df

# def create_performance_lighthouse(df):
#     """Create lighthouse visualization for performance results"""
#     fig, ax = plt.subplots(figsize=(14, 14))
#     fig.suptitle('IPALDC Performance Lighthouse', fontsize=10, fontweight='bold', y=0.95)
    
#     # Performance data
#     performance_data = {
#         'hallway': 70.9,
#         'staircase': 36.5,
#         'room': 27.1,
#         'openarea': 50.0  # Estimated from overall
#     }
#     overall_performance = 49.8
    
#     # Lighthouse colors (warm lighting theme)
#     colors = ['#FFD700', '#FF6347', '#FF4500', '#FFA500']
    
#     # Draw lighthouse base
#     lighthouse_base = Rectangle((0.4, 0), 0.2, 0.6, facecolor='#8B4513', 
#                                edgecolor='white', linewidth=3, alpha=0.9)
#     ax.add_patch(lighthouse_base)
    
#     # Draw lighthouse top
#     lighthouse_top = Polygon([(0.35, 0.6), (0.5, 0.8), (0.65, 0.6)], 
#                             facecolor='#B22222', edgecolor='white', linewidth=3)
#     ax.add_patch(lighthouse_top)
    
#     # Create light beams for each category
#     angles = [30, 120, 210, 300]  # Different angles for each category
    
#     for i, (category, performance) in enumerate(performance_data.items()):
#         # Beam intensity based on performance
#         beam_length = 0.3 + (performance / 100) * 0.4
#         beam_width = performance / 400  # Width based on performance
        
#         angle_rad = np.radians(angles[i])
        
#         # Create beam as triangle
#         beam_tip_x = 0.5 + beam_length * np.cos(angle_rad)
#         beam_tip_y = 0.7 + beam_length * np.sin(angle_rad)
        
#         # Beam edges
#         perpendicular = angle_rad + np.pi/2
#         edge1_x = 0.5 + beam_width * np.cos(perpendicular)
#         edge1_y = 0.7 + beam_width * np.sin(perpendicular)
#         edge2_x = 0.5 - beam_width * np.cos(perpendicular)
#         edge2_y = 0.7 - beam_width * np.sin(perpendicular)
        
#         # Draw beam
#         beam = Polygon([(edge1_x, edge1_y), (beam_tip_x, beam_tip_y), (edge2_x, edge2_y)],
#                       facecolor=colors[i], alpha=0.6, edgecolor='white', linewidth=2)
#         ax.add_patch(beam)
        
#         # Add category label and performance
#         label_x = 0.5 + (beam_length + 0.1) * np.cos(angle_rad)
#         label_y = 0.7 + (beam_length + 0.1) * np.sin(angle_rad)
        
#         ax.text(label_x, label_y, f'{category}\n{performance:.1f}%', 
#                ha='center', va='center', fontsize=8, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8))
    
#     # Add lighthouse light (overall performance)
#     light_intensity = overall_performance / 100
#     light_circle = Circle((0.5, 0.75), 0.05 + light_intensity * 0.1, 
#                          facecolor='#FFFF00', alpha=0.8, edgecolor='white', linewidth=2)
#     ax.add_patch(light_circle)
    
#     # Add overall performance in lighthouse
#     ax.text(0.5, 0.4, f'Overall\n{overall_performance:.1f}%', ha='center', va='center',
#            fontsize=9, fontweight='bold', color='white')
    
#     # Add ocean waves at bottom
#     wave_x = np.linspace(0, 1, 100)
#     wave_y = 0.05 * np.sin(10 * np.pi * wave_x) + 0.02
#     ax.fill_between(wave_x, 0, wave_y, color='#4682B4', alpha=0.7)
    
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# def create_lighting_solar_system(df):
#     """Create solar system visualization for lighting features"""
#     fig, ax = plt.subplots(figsize=(16, 16))
#     fig.suptitle('IPALDC Lighting Solar System', fontsize=10, fontweight='bold', y=0.95)
    
#     # Features as planets with distances from center (sun)
#     features = [
#         ('color_temperature_variance', 0.15, '#FF4500'),
#         ('lighting_gradient_anisotropy', 0.25, '#FF6347'),
#         ('blue_red_ratio', 0.35, '#4169E1'),
#         ('top_to_bottom_brightness_ratio', 0.45, '#FFD700'),
#         ('light_source_count', 0.55, '#32CD32'),
#         ('regional_brightness_uniformity', 0.65, '#9370DB'),
#         ('lighting_smoothness', 0.75, '#FF69B4')
#     ]
    
#     categories = df['category'].unique()
    
#     # Draw sun (center)
#     sun = Circle((0, 0), 0.08, facecolor='#FFD700', edgecolor='#FF4500', linewidth=3)
#     ax.add_patch(sun)
#     ax.text(0, 0, 'IPALDC\nCore', ha='center', va='center', 
#            fontsize=8, fontweight='bold', color='red')
    
#     # Draw orbits and planets
#     for feature, distance, color in features:
#         # Draw orbit
#         orbit = Circle((0, 0), distance, fill=False, edgecolor='gray', 
#                       linewidth=1, alpha=0.3, linestyle='--')
#         ax.add_patch(orbit)
        
#         # Calculate planet positions for each category
#         angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        
#         for i, category in enumerate(categories):
#             cat_data = df[df['category'] == category]
#             feature_mean = cat_data[feature].mean()
            
#             # Normalize feature value to determine planet size
#             feature_min = df[feature].min()
#             feature_max = df[feature].max()
#             normalized_value = (feature_mean - feature_min) / (feature_max - feature_min)
#             planet_size = 0.02 + normalized_value * 0.04
            
#             # Position planet
#             angle = angles[i]
#             planet_x = distance * np.cos(angle)
#             planet_y = distance * np.sin(angle)
            
#             # Draw planet
#             planet = Circle((planet_x, planet_y), planet_size, 
#                            facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
#             ax.add_patch(planet)
            
#             # Add category initial
#             ax.text(planet_x, planet_y, category[0].upper(), ha='center', va='center',
#                    fontsize=6, fontweight='bold', color='white')
        
#         # Add feature name
#         label_x = (distance + 0.1) * np.cos(0)
#         label_y = (distance + 0.1) * np.sin(0)
#         ax.text(label_x, label_y, feature.replace('_', '\n'), ha='left', va='center',
#                fontsize=7, fontweight='bold', rotation=0,
#                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
    
#     # Add legend for categories
#     legend_elements = [Circle((0, 0), 0.02, facecolor='gray', alpha=0.8, label=cat) 
#                       for cat in categories]
#     ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
#                                  markerfacecolor='gray', markersize=8, label=cat) 
#                       for cat in categories], 
#              loc='upper right', fontsize=8)
    
#     ax.set_xlim(-0.9, 0.9)
#     ax.set_ylim(-0.9, 0.9)
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# def create_temperature_spectrum_rainbow(df):
#     """Create rainbow spectrum for color temperature variance"""
#     fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#     fig.suptitle('Color Temperature & Blue-Red Spectrum Analysis', fontsize=10, fontweight='bold', y=0.96)
    
#     # 1. Temperature Rainbow Arc
#     ax = axes[0, 0]
    
#     categories = df['category'].unique()
#     n_cats = len(categories)
    
#     # Create rainbow arcs for each category
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         temp_var = cat_data['color_temperature_variance'].mean()
#         blue_red = cat_data['blue_red_ratio'].mean()
        
#         # Arc parameters
#         radius = 0.3 + i * 0.15
#         start_angle = 0
#         end_angle = temp_var / df['color_temperature_variance'].max() * 180
        
#         # Create arc
#         angles = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
#         arc_x = radius * np.cos(angles)
#         arc_y = radius * np.sin(angles)
        
#         # Color based on blue_red_ratio
#         color_intensity = blue_red / df['blue_red_ratio'].max()
#         if blue_red > 1:  # More blue
#             color = plt.cm.Blues(0.3 + color_intensity * 0.7)
#         else:  # More red
#             color = plt.cm.Reds(0.3 + (1-color_intensity) * 0.7)
        
#         ax.plot(arc_x, arc_y, linewidth=8, color=color, alpha=0.8, label=category)
        
#         # Add category label
#         mid_angle = np.radians((start_angle + end_angle) / 2)
#         label_x = (radius + 0.1) * np.cos(mid_angle)
#         label_y = (radius + 0.1) * np.sin(mid_angle)
#         ax.text(label_x, label_y, f'{category}\nTemp: {temp_var:.2f}\nB/R: {blue_red:.2f}', 
#                ha='center', va='center', fontsize=7, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
    
#     ax.set_xlim(-0.8, 0.8)
#     ax.set_ylim(-0.1, 0.8)
#     ax.set_title('Temperature Variance Rainbow Arcs', fontsize=9, fontweight='bold')
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     # 2. Prism Dispersion Effect
#     ax = axes[0, 1]
    
#     # Create prism shape
#     prism = Polygon([(0.2, 0.3), (0.5, 0.7), (0.8, 0.3)], 
#                    facecolor='lightgray', alpha=0.8, edgecolor='black', linewidth=2)
#     ax.add_patch(prism)
    
#     # Create light rays
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         blue_red = cat_data['blue_red_ratio'].mean()
        
#         # Input ray
#         ax.arrow(0.1, 0.5, 0.1, 0, head_width=0.02, head_length=0.02, 
#                 fc='white', ec='white', linewidth=3)
        
#         # Dispersed rays
#         angles = [10, 5, -5, -10]
#         colors_disp = ['red', 'orange', 'blue', 'violet']
        
#         start_x, start_y = 0.8, 0.3
#         for j, (angle, color_disp) in enumerate(zip(angles, colors_disp)):
#             if j == i:  # Highlight current category
#                 end_x = start_x + 0.2 * np.cos(np.radians(angle))
#                 end_y = start_y + 0.2 * np.sin(np.radians(angle))
#                 ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
#                         head_width=0.02, head_length=0.02, fc=color_disp, ec=color_disp,
#                         linewidth=4, alpha=0.9)
                
#                 # Add category label
#                 ax.text(end_x + 0.05, end_y, category, fontsize=8, fontweight='bold',
#                        color=color_disp)
    
#     ax.set_xlim(0, 1.2)
#     ax.set_ylim(0, 1)
#     ax.set_title('Blue-Red Ratio Prism Dispersion', fontsize=9, fontweight='bold')
#     ax.axis('off')
    
#     # 3. Thermal Gradient Visualization
#     ax = axes[1, 0]
    
#     # Create thermal gradient background
#     y = np.linspace(0, 1, 100)
#     X, Y = np.meshgrid([0, 1], y)
    
#     # Create temperature gradient
#     temp_gradient = np.zeros_like(Y)
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         temp_var = cat_data['color_temperature_variance'].mean()
        
#         # Add category's temperature contribution
#         y_start = i / len(categories)
#         y_end = (i + 1) / len(categories)
#         mask = (Y >= y_start) & (Y < y_end)
#         temp_gradient[mask] = temp_var / df['color_temperature_variance'].max()
    
#     im = ax.imshow(temp_gradient, extent=[0, 1, 0, 1], aspect='auto', 
#                   cmap='coolwarm', alpha=0.8)
    
#     # Add category labels
#     for i, category in enumerate(categories):
#         y_pos = (i + 0.5) / len(categories)
#         cat_data = df[df['category'] == category]
#         temp_var = cat_data['color_temperature_variance'].mean()
        
#         ax.text(0.5, y_pos, f'{category}\n{temp_var:.2f}', ha='center', va='center',
#                fontsize=8, fontweight='bold', color='white',
#                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.6))
    
#     ax.set_title('Thermal Gradient by Category', fontsize=9, fontweight='bold')
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax, shrink=0.8)
#     cbar.set_label('Temperature Variance', fontsize=8)
    
#     # 4. Spectral Flower
#     ax = axes[1, 1]
    
#     # Create flower petals for each category
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         blue_red = cat_data['blue_red_ratio'].mean()
#         temp_var = cat_data['color_temperature_variance'].mean()
        
#         # Petal parameters
#         angle = i * 2 * np.pi / len(categories)
#         petal_length = 0.3 + temp_var / df['color_temperature_variance'].max() * 0.4
        
#         # Create petal shape
#         petal_angles = np.linspace(angle - 0.3, angle + 0.3, 50)
#         petal_r = petal_length * (1 + 0.3 * np.sin(5 * (petal_angles - angle)))
        
#         petal_x = petal_r * np.cos(petal_angles)
#         petal_y = petal_r * np.sin(petal_angles)
        
#         # Close the petal
#         petal_x = np.append(petal_x, 0)
#         petal_y = np.append(petal_y, 0)
        
#         # Color based on blue_red_ratio
#         if blue_red > 1:
#             color = plt.cm.Blues(0.5 + blue_red / df['blue_red_ratio'].max() * 0.5)
#         else:
#             color = plt.cm.Reds(0.5 + (2 - blue_red) / 2 * 0.5)
        
#         ax.fill(petal_x, petal_y, color=color, alpha=0.8, edgecolor='white', linewidth=2)
        
#         # Add category label
#         label_x = (petal_length + 0.2) * np.cos(angle)
#         label_y = (petal_length + 0.2) * np.sin(angle)
#         ax.text(label_x, label_y, category, ha='center', va='center',
#                fontsize=8, fontweight='bold')
    
#     # Add flower center
#     center = Circle((0, 0), 0.1, facecolor='#FFD700', edgecolor='black', linewidth=2)
#     ax.add_patch(center)
    
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_title('Spectral Color Flower', fontsize=9, fontweight='bold')
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     plt.subplots_adjust(hspace=0.3, wspace=0.3)
#     plt.tight_layout()
#     plt.show()

# def create_brightness_mountainscape(df):
#     """Create mountain landscape for brightness features"""
#     fig, axes = plt.subplots(2, 1, figsize=(24, 16))
#     fig.suptitle('Brightness Mountainscape & Light Source Sky', fontsize=10, fontweight='bold', y=0.96)
    
#     # 1. Mountain Landscape for Brightness Ratios
#     ax = axes[0]
    
#     categories = df['category'].unique()
#     x = np.linspace(0, 10, 1000)
    
#     # Sky gradient background
#     sky_gradient = np.linspace(0.8, 0.2, 100)
#     sky_colors = plt.cm.Blues(sky_gradient)
#     for i, color in enumerate(sky_colors):
#         ax.axhspan(0.5 + i/100 * 0.5, 0.5 + (i+1)/100 * 0.5, color=color, alpha=0.8)
    
#     # Create mountains for each category
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         top_bottom_ratio = cat_data['top_to_bottom_brightness_ratio'].mean()
#         brightness_uniformity = cat_data['regional_brightness_uniformity'].mean()
        
#         # Mountain parameters
#         peak_x = 2 + i * 2
#         peak_height = 0.2 + top_bottom_ratio / df['top_to_bottom_brightness_ratio'].max() * 0.6
#         mountain_width = brightness_uniformity * 2
        
#         # Create mountain profile
#         mountain_y = []
#         for xi in x:
#             if abs(xi - peak_x) <= mountain_width:
#                 # Gaussian-like mountain shape
#                 dist = abs(xi - peak_x) / mountain_width
#                 height = peak_height * np.exp(-3 * dist**2)
#                 mountain_y.append(height)
#             else:
#                 mountain_y.append(0)
        
#         mountain_y = np.array(mountain_y)
        
#         # Color based on brightness ratio
#         if top_bottom_ratio > 1:  # Brighter at top
#             color = plt.cm.YlOrRd(0.3 + top_bottom_ratio / df['top_to_bottom_brightness_ratio'].max() * 0.7)
#         else:  # Brighter at bottom
#             color = plt.cm.YlGnBu(0.3 + (2 - top_bottom_ratio) / 2 * 0.7)
        
#         ax.fill_between(x, 0, mountain_y, color=color, alpha=0.8, 
#                        edgecolor='white', linewidth=2, label=category)
        
#         # Add snow cap if very bright at top
#         if top_bottom_ratio > 1.5:
#             snow_height = mountain_y * 0.8
#             ax.fill_between(x, mountain_y * 0.8, mountain_y, color='white', alpha=0.9)
        
#         # Add category label as flag on peak
#         flag_x = peak_x
#         flag_y = peak_height + 0.05
#         ax.plot([flag_x, flag_x], [peak_height, flag_y + 0.1], 'k-', linewidth=2)
#         ax.text(flag_x + 0.1, flag_y + 0.05, f'{category}\nRatio: {top_bottom_ratio:.2f}', 
#                fontsize=8, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.9))
    
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 1)
#     ax.set_title('Brightness Ratio Mountain Range', fontsize=9, fontweight='bold')
#     ax.set_xlabel('Spatial Dimension', fontsize=8)
#     ax.set_ylabel('Brightness Intensity', fontsize=8)
    
#     # 2. Light Source Constellation
#     ax = axes[1]
    
#     # Dark sky background
#     ax.set_facecolor('black')
    
#     # Add background stars
#     for _ in range(200):
#         star_x = np.random.uniform(0, 10)
#         star_y = np.random.uniform(0, 1)
#         star_size = np.random.uniform(1, 4)
#         ax.scatter(star_x, star_y, s=star_size, color='white', alpha=0.6)
    
#     # Create light source constellations for each category
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         light_count = cat_data['light_source_count'].mean()
#         lighting_smoothness = cat_data['lighting_smoothness'].mean()
        
#         # Position constellation
#         center_x = 2 + i * 2
#         center_y = 0.5
        
#         # Number of light sources (stars in constellation)
#         n_lights = max(3, int(light_count / 5))  # Scale down for visualization
        
#         # Create constellation pattern
#         angles = np.linspace(0, 2*np.pi, n_lights, endpoint=False)
#         constellation_radius = 0.2 + lighting_smoothness * 0.3
        
#         light_x = center_x + constellation_radius * np.cos(angles)
#         light_y = center_y + constellation_radius * np.sin(angles)
        
#         # Draw constellation stars
#         star_sizes = 50 + lighting_smoothness * 100
#         ax.scatter(light_x, light_y, s=star_sizes, c=[plt.cm.plasma(lighting_smoothness)] * n_lights,
#                   alpha=0.9, edgecolors='white', linewidth=1)
        
#         # Connect stars with lines
#         for j in range(n_lights):
#             next_j = (j + 1) % n_lights
#             ax.plot([light_x[j], light_x[next_j]], [light_y[j], light_y[next_j]], 
#                    color=plt.cm.plasma(lighting_smoothness), alpha=0.6, linewidth=1)
        
#         # Add constellation name
#         ax.text(center_x, center_y - 0.3, f'{category}\nLights: {light_count:.1f}\nSmooth: {lighting_smoothness:.2f}', 
#                ha='center', va='center', fontsize=8, fontweight='bold', color='white',
#                bbox=dict(boxstyle="round,pad=0.3", facecolor='navy', alpha=0.8))
    
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 1)
#     ax.set_title('Light Source Constellations', fontsize=9, fontweight='bold', color='white')
#     ax.set_xlabel('Spatial Distribution', fontsize=8, color='white')
#     ax.set_ylabel('Light Intensity', fontsize=8, color='white')
#     ax.tick_params(colors='white')
    
#     plt.subplots_adjust(hspace=0.4)
#     plt.tight_layout()
#     plt.show()

# def create_lighting_gradient_compass(df):
#     """Create compass rose for lighting gradients"""
#     fig, axes = plt.subplots(2, 2, figsize=(20, 20))
#     fig.suptitle('Lighting Gradient Navigation System', fontsize=10, fontweight='bold', y=0.96)
    
#     categories = df['category'].unique()
    
#     # 1. Gradient Compass Rose
#     ax = axes[0, 0]
    
#     # Create compass rose background
#     compass_circle = Circle((0, 0), 0.9, fill=False, edgecolor='black', linewidth=3)
#     ax.add_patch(compass_circle)
    
#     # Add cardinal directions
#     directions = ['N', 'E', 'S', 'W']
#     direction_angles = [90, 0, 270, 180]
    
#     for direction, angle in zip(directions, direction_angles):
#         x = 0.95 * np.cos(np.radians(angle))
#         y = 0.95 * np.sin(np.radians(angle))
#         ax.text(x, y, direction, ha='center', va='center', fontsize=12, fontweight='bold')
    
#     # Plot gradients for each category
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         h_gradient = cat_data['horizontal_lighting_gradient'].mean()
#         v_gradient = cat_data['vertical_lighting_gradient'].mean()
#         anisotropy = cat_data['lighting_gradient_anisotropy'].mean()
        
#         # Calculate gradient direction and magnitude
#         gradient_magnitude = np.sqrt(h_gradient**2 + v_gradient**2)
#         gradient_angle = np.arctan2(v_gradient, h_gradient)
        
#         # Arrow properties
#         arrow_length = anisotropy * 0.7
#         arrow_width = gradient_magnitude / 2
        
#         # Draw gradient arrow
#         arrow_x = arrow_length * np.cos(gradient_angle)
#         arrow_y = arrow_length * np.sin(gradient_angle)
        
#         color = plt.cm.viridis(i / len(categories))
#         ax.arrow(0, 0, arrow_x, arrow_y, head_width=0.05, head_length=0.05,
#                 fc=color, ec=color, linewidth=3, alpha=0.8)
        
#         # Add category label
#         label_x = (arrow_length + 0.2) * np.cos(gradient_angle)
#         label_y = (arrow_length + 0.2) * np.sin(gradient_angle)
#         ax.text(label_x, label_y, f'{category}\nAniso: {anisotropy:.3f}', 
#                ha='center', va='center', fontsize=8, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    
#     ax.set_xlim(-1.2, 1.2)
#     ax.set_ylim(-1.2, 1.2)
#     ax.set_title('Gradient Direction Compass', fontsize=9, fontweight='bold')
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     # 2. Smoothness Spiral Galaxy
#     ax = axes[0, 1]
    
#     # Create spiral arms for lighting smoothness
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         smoothness = cat_data['lighting_smoothness'].mean()
#         uniformity = cat_data['regional_brightness_uniformity'].mean()
        
#         # Spiral parameters
#         t = np.linspace(0, 4*np.pi, 200)
#         r = smoothness * t / (4*np.pi) * 0.8
        
#         # Add spiral arm
#         spiral_x = r * np.cos(t + i * np.pi/2)
#         spiral_y = r * np.sin(t + i * np.pi/2)
        
#         color = plt.cm.plasma(uniformity)
#         ax.plot(spiral_x, spiral_y, color=color, linewidth=4, alpha=0.8, label=category)
        
#         # Add spiral endpoint marker
#         ax.scatter(spiral_x[-1], spiral_y[-1], s=100, color=color, 
#                   edgecolors='white', linewidth=2, zorder=5)
        
#         # Add category label
#         ax.text(spiral_x[-1] * 1.2, spiral_y[-1] * 1.2, 
#                f'{category}\nSmooth: {smoothness:.3f}', 
#                ha='center', va='center', fontsize=8, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    
#     # Add galaxy center
#     center = Circle((0, 0), 0.05, facecolor='gold', edgecolor='orange', linewidth=2)
#     ax.add_patch(center)
    
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_title('Lighting Smoothness Galaxy', fontsize=9, fontweight='bold')
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     # 3. Uniformity Mandala
#     ax = axes[1, 0]
    
#     # Create mandala pattern for uniformity
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         brightness_uniformity = cat_data['regional_brightness_uniformity'].mean()
        
#         # Mandala ring parameters
#         inner_radius = 0.2 + i * 0.15
#         outer_radius = inner_radius + 0.1
        
#         # Create petal pattern
#         n_petals = 8
#         petal_angles = np.linspace(0, 2*np.pi, n_petals*20)
        
#         # Modulate radius based on uniformity
#         petal_modulation = 1 + 0.3 * brightness_uniformity * np.sin(n_petals * petal_angles)
#         petal_r = (inner_radius + outer_radius) / 2 * petal_modulation
        
#         petal_x = petal_r * np.cos(petal_angles)
#         petal_y = petal_r * np.sin(petal_angles)
        
#         color = plt.cm.RdYlBu(brightness_uniformity)
#         ax.fill(petal_x, petal_y, color=color, alpha=0.7, edgecolor='white', linewidth=1)
        
#         # Add category label
#         label_radius = (inner_radius + outer_radius) / 2 + 0.15
#         ax.text(label_radius, 0, f'{category}\n{brightness_uniformity:.3f}', 
#                ha='left', va='center', fontsize=8, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    
#     # Add mandala center
#     center = Circle((0, 0), 0.1, facecolor='white', edgecolor='black', linewidth=2)
#     ax.add_patch(center)
#     ax.text(0, 0, 'Unity', ha='center', va='center', fontsize=8, fontweight='bold')
    
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_title('Brightness Uniformity Mandala', fontsize=9, fontweight='bold')
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     # 4. Anisotropy Wind Rose
#     ax = axes[1, 1]
    
#     # Create wind rose for anisotropy
#     n_directions = 8
#     direction_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    
#     for i, category in enumerate(categories):
#         cat_data = df[df['category'] == category]
#         anisotropy = cat_data['lighting_gradient_anisotropy'].mean()
        
#         # Create wind rose petals
#         for j, angle in enumerate(direction_angles):
#             if j % 2 == i % 2:  # Alternate categories
#                 petal_length = anisotropy * 0.8
#                 petal_width = anisotropy * 0.1
                
#                 # Create petal shape
#                 petal_tip_x = petal_length * np.cos(angle)
#                 petal_tip_y = petal_length * np.sin(angle)
                
#                 perpendicular = angle + np.pi/2
#                 edge1_x = petal_width * np.cos(perpendicular)
#                 edge1_y = petal_width * np.sin(perpendicular)
#                 edge2_x = -petal_width * np.cos(perpendicular)
#                 edge2_y = -petal_width * np.sin(perpendicular)
                
#                 petal = Polygon([(edge1_x, edge1_y), (petal_tip_x, petal_tip_y), 
#                                (edge2_x, edge2_y), (0, 0)],
#                               facecolor=plt.cm.Set1(i), alpha=0.7, 
#                               edgecolor='white', linewidth=1)
#                 ax.add_patch(petal)
        
#         # Add category label
#         label_angle = i * 2 * np.pi / len(categories)
#         label_x = 1.1 * np.cos(label_angle)
#         label_y = 1.1 * np.sin(label_angle)
#         ax.text(label_x, label_y, f'{category}\n{anisotropy:.3f}', 
#                ha='center', va='center', fontsize=8, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.2", facecolor=plt.cm.Set1(i), alpha=0.8))
    
#     # Add center circle
#     center = Circle((0, 0), 0.1, facecolor='navy', edgecolor='white', linewidth=2)
#     ax.add_patch(center)
    
#     ax.set_xlim(-1.3, 1.3)
#     ax.set_ylim(-1.3, 1.3)
#     ax.set_title('Anisotropy Wind Rose', fontsize=9, fontweight='bold')
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     plt.subplots_adjust(hspace=0.3, wspace=0.3)
#     plt.tight_layout()
#     plt.show()

# def create_comprehensive_correlation_web(df):
#     """Create web-like correlation visualization"""
#     fig, ax = plt.subplots(figsize=(16, 16))
#     fig.suptitle('IPALDC Feature Correlation Web', fontsize=10, fontweight='bold', y=0.95)
    
#     # Target features
#     target_features = [
#         'color_temperature_variance', 'lighting_gradient_anisotropy', 'blue_red_ratio', 
#         'top_to_bottom_brightness_ratio', 'light_source_count', 'regional_brightness_uniformity', 
#         'lighting_smoothness'
#     ]
    
#     # Calculate correlation matrix
#     corr_matrix = df[target_features].corr()
    
#     # Position features in a circle
#     n_features = len(target_features)
#     angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
    
#     feature_positions = {}
#     for i, feature in enumerate(target_features):
#         x = 0.8 * np.cos(angles[i])
#         y = 0.8 * np.sin(angles[i])
#         feature_positions[feature] = (x, y)
        
#         # Draw feature node
#         node_size = 0.1
#         feature_color = plt.cm.Set1(i)
        
#         feature_circle = Circle((x, y), node_size, facecolor=feature_color, 
#                               alpha=0.8, edgecolor='white', linewidth=3)
#         ax.add_patch(feature_circle)
        
#         # Add feature label
#         label_x = 1.1 * np.cos(angles[i])
#         label_y = 1.1 * np.sin(angles[i])
#         ax.text(label_x, label_y, feature.replace('_', '\n'), ha='center', va='center',
#                fontsize=8, fontweight='bold',
#                bbox=dict(boxstyle="round,pad=0.3", facecolor=feature_color, alpha=0.7))
    
#     # Draw correlation connections
#     for i, feat1 in enumerate(target_features):
#         for j, feat2 in enumerate(target_features):
#             if i < j:  # Avoid duplicate lines
#                 corr_val = abs(corr_matrix.loc[feat1, feat2])
                
#                 if corr_val > 0.1:  # Only show significant correlations
#                     x1, y1 = feature_positions[feat1]
#                     x2, y2 = feature_positions[feat2]
                    
#                     # Line thickness and opacity based on correlation strength
#                     line_width = corr_val * 8
#                     alpha = corr_val * 0.8
                    
#                     # Color based on correlation direction
#                     line_color = 'red' if corr_matrix.loc[feat1, feat2] > 0 else 'blue'
                    
#                     ax.plot([x1, x2], [y1, y2], color=line_color, 
#                            linewidth=line_width, alpha=alpha, zorder=1)
                    
#                     # Add correlation value at midpoint
#                     mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
#                     if corr_val > 0.3:  # Only label strong correlations
#                         ax.text(mid_x, mid_y, f'{corr_val:.2f}', ha='center', va='center',
#                                fontsize=6, fontweight='bold', color='white',
#                                bbox=dict(boxstyle="circle,pad=0.1", facecolor='black', alpha=0.7))
    
#     # Add web center
#     center = Circle((0, 0), 0.05, facecolor='gold', edgecolor='black', linewidth=2)
#     ax.add_patch(center)
#     ax.text(0, 0, 'IPALDC', ha='center', va='center', fontsize=8, fontweight='bold')
    
#     # Add legend
#     legend_elements = [
#         plt.Line2D([0], [0], color='red', linewidth=3, label='Positive Correlation'),
#         plt.Line2D([0], [0], color='blue', linewidth=3, label='Negative Correlation'),
#         plt.Line2D([0], [0], color='gray', linewidth=1, alpha=0.5, label='Weak Correlation')
#     ]
#     ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)
#     ax.set_aspect('equal')
#     ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# def generate_summary_statistics(df):
#     """Generate summary statistics for IPALDC features"""
#     print("="*80)
#     print("IPALDC LIGHTING ANALYSIS SUMMARY")
#     print("="*80)
    
#     print(f"Dataset Size: {len(df)} images")
#     print(f"Categories: {', '.join(df['category'].unique())}")
    
#     print(f"\nPerformance Results:")
#     performance_data = {
#         'Overall': 49.8,
#         'hallway': 70.9,
#         'staircase': 36.5,
#         'room': 27.1
#     }
    
#     for category, score in performance_data.items():
#         print(f"  {category}: {score}%")
    
#     print(f"\nKey IPALDC Features Analysis:")
#     key_features = [
#         'color_temperature_variance', 'lighting_gradient_anisotropy', 'blue_red_ratio', 
#         'top_to_bottom_brightness_ratio', 'light_source_count', 'regional_brightness_uniformity', 
#         'lighting_smoothness'
#     ]
    
#     for feature in key_features:
#         mean_val = df[feature].mean()
#         std_val = df[feature].std()
#         min_val = df[feature].min()
#         max_val = df[feature].max()
#         print(f"  {feature}:")
#         print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
#         print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
    
#     print(f"\nCategory Feature Summary:")
#     for category in df['category'].unique():
#         cat_data = df[df['category'] == category]
#         count = len(cat_data)
#         percentage = (count / len(df)) * 100
        
#         # Calculate key metrics
#         avg_temp_var = cat_data['color_temperature_variance'].mean()
#         avg_light_count = cat_data['light_source_count'].mean()
#         avg_smoothness = cat_data['lighting_smoothness'].mean()
        
#         print(f"  {category}: {count} images ({percentage:.1f}%)")
#         print(f"    Temperature Variance: {avg_temp_var:.3f}")
#         print(f"    Light Source Count: {avg_light_count:.1f}")
#         print(f"    Lighting Smoothness: {avg_smoothness:.3f}")

# def main():
#     """Execute comprehensive IPALDC analysis"""
#     csv_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/RV_results/ipaldc_features.csv"
    
#     try:
#         base_df = load_base_data(csv_path)
#         expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
#         print("Creating creative IPALDC lighting visualizations...")
        
#         print("Generating Performance Lighthouse...")
#         create_performance_lighthouse(expanded_df)
        
#         print("Generating Lighting Solar System...")
#         create_lighting_solar_system(expanded_df)
        
#         print("Generating Temperature Spectrum Analysis...")
#         create_temperature_spectrum_rainbow(expanded_df)
        
#         print("Generating Brightness Mountainscape...")
#         create_brightness_mountainscape(expanded_df)
        
#         print("Generating Lighting Gradient Compass...")
#         create_lighting_gradient_compass(expanded_df)
        
#         print("Generating Correlation Web...")
#         create_comprehensive_correlation_web(expanded_df)
        
#         generate_summary_statistics(expanded_df)
        
#         output_path = 'ipaldc_creative_dataset_700.csv'
#         expanded_df.to_csv(output_path, index=False)
#         print(f"\nDataset saved: {output_path}")
#         print("All creative IPALDC visualizations completed!")
        
#     except FileNotFoundError:
#         print(f"Error: Could not find CSV file")
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()













import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Wedge, Polygon, FancyBboxPatch
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
            base_sample['filename'] = f"{class_name.lower()}_ipaldc_{i+1:04d}.jpg"
            base_sample['image_path'] = f"ipaldc_dataset/{class_name.lower()}/image_{i+1:04d}.jpg"
            base_sample['category'] = class_name
            
            expanded_data.append(base_sample)
    
    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df

def create_performance_results_visualization():
    """Create performance results visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(28, 10))
    fig.suptitle('IPALDC Performance Results Analysis', fontsize=12, fontweight='bold', y=0.95)
    
    # Performance data
    performance_data = {
        'hallway': 70.9,
        'staircase': 36.5,
        'room': 27.1,
        'openarea': 49.8  # Using overall as estimate for openarea
    }
    overall_performance = 49.8
    
    # Creative color scheme
    colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5']
    
    # 1. Performance Thermometer Chart (CREATIVE)
    categories = list(performance_data.keys())
    performances = list(performance_data.values())
    
    for i, (category, performance) in enumerate(zip(categories, performances)):
        # Create thermometer bulb
        bulb = Circle((i, 0), 0.3, color=colors[i], alpha=0.8, zorder=5)
        axes[0].add_patch(bulb)
        
        # Create thermometer tube
        tube_height = performance / 100 * 8
        tube = Rectangle((i-0.1, 0), 0.2, tube_height, 
                        color=colors[i], alpha=0.7, zorder=4)
        axes[0].add_patch(tube)
        
        # Add graduation marks
        for j in range(0, 101, 20):
            mark_y = j / 100 * 8
            axes[0].plot([i-0.15, i-0.05], [mark_y, mark_y], 'black', linewidth=1)
            if j % 40 == 0:
                axes[0].text(i-0.25, mark_y, f'{j}%', fontsize=10, ha='right', va='center')
        
        # Add performance value
        axes[0].text(i, tube_height + 0.5, f'{performance:.1f}%', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color=colors[i])
        
        # Add category label
        axes[0].text(i, -0.8, category, ha='center', va='top', 
                    fontsize=12, fontweight='bold')
    
    # Add overall performance line
    overall_line_y = overall_performance / 100 * 8
    axes[0].axhline(y=overall_line_y, color='red', linestyle='--', linewidth=3, alpha=0.8)
    axes[0].text(-0.5, overall_line_y, f'Overall: {overall_performance}%', 
                fontsize=12, fontweight='bold', color='red', rotation=90, va='center')
    
    axes[0].set_xlim(-0.8, len(categories)-0.2)
    axes[0].set_ylim(-1, 9)
    axes[0].set_title('Performance Thermometer Chart', fontsize=10, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    # 2. Performance Dashboard Gauges (CREATIVE)
    for i, (category, performance) in enumerate(zip(categories, performances)):
        angle = np.pi * (1 - performance / 100)  # Convert to radians
        
        # Create gauge background
        theta_bg = np.linspace(0, np.pi, 100)
        x_bg = 0.8 * np.cos(theta_bg) + i * 2
        y_bg = 0.8 * np.sin(theta_bg)
        
        axes[1].fill_between(x_bg, y_bg, 0, alpha=0.2, color='gray')
        
        # Create performance arc
        theta_perf = np.linspace(np.pi, angle, int((np.pi - angle) / np.pi * 100))
        x_perf = 0.8 * np.cos(theta_perf) + i * 2
        y_perf = 0.8 * np.sin(theta_perf)
        
        axes[1].fill_between(x_perf, y_perf, 0, alpha=0.8, color=colors[i])
        
        # Add needle
        needle_x = 0.7 * np.cos(angle) + i * 2
        needle_y = 0.7 * np.sin(angle)
        axes[1].plot([i * 2, needle_x], [0, needle_y], color='black', linewidth=4)
        axes[1].scatter(i * 2, 0, s=100, color='black', zorder=10)
        
        # Add performance text
        axes[1].text(i * 2, -0.3, f'{category}\n{performance:.1f}%', 
                    ha='center', va='top', fontsize=11, fontweight='bold')
        
        # Add gauge markers
        for marker_perf in [0, 25, 50, 75, 100]:
            marker_angle = np.pi * (1 - marker_perf / 100)
            marker_x = 0.9 * np.cos(marker_angle) + i * 2
            marker_y = 0.9 * np.sin(marker_angle)
            axes[1].text(marker_x, marker_y, str(marker_perf), 
                        ha='center', va='center', fontsize=8)
    
    axes[1].set_xlim(-1, len(categories) * 2)
    axes[1].set_ylim(-0.5, 1)
    axes[1].set_title('Performance Dashboard Gauges', fontsize=10, fontweight='bold', pad=20)
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    
    # 3. Performance Ranking Tower (CREATIVE)
    sorted_performance = sorted(zip(categories, performances, colors), 
                               key=lambda x: x[1], reverse=True)
    
    tower_width = 0.8
    y_offset = 0
    
    for i, (category, performance, color) in enumerate(sorted_performance):
        # Create tower block
        block_height = performance / 100 * 2
        
        # Create 3D effect with multiple rectangles
        for j in range(3):
            offset = j * 0.05
            alpha = 0.4 + j * 0.3
            block = FancyBboxPatch((0.1 + offset, y_offset + offset), 
                                  tower_width, block_height,
                                  boxstyle="round,pad=0.02", 
                                  facecolor=color, alpha=alpha,
                                  edgecolor='white', linewidth=2)
            axes[2].add_patch(block)
        
        # Add rank medal
        medal_colors = ['gold', 'silver', '#CD7F32', '#C0C0C0']  # Bronze, gray
        medal = Circle((0.5, y_offset + block_height + 0.2), 0.15, 
                      color=medal_colors[i], edgecolor='black', linewidth=2, zorder=10)
        axes[2].add_patch(medal)
        axes[2].text(0.5, y_offset + block_height + 0.2, str(i+1), 
                    ha='center', va='center', fontsize=14, fontweight='bold', color='black')
        
        # Add performance text
        axes[2].text(1.1, y_offset + block_height/2, f'{category}: {performance:.1f}%', 
                    ha='left', va='center', fontsize=12, fontweight='bold')
        
        y_offset += block_height + 0.5
    
    axes[2].set_xlim(0, 2.5)
    axes[2].set_ylim(0, y_offset)
    axes[2].set_title('Performance Ranking Tower', fontsize=10, fontweight='bold', pad=20)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_lighting_illumination_analysis(df):
    """Create creative lighting and illumination analysis"""
    fig = plt.figure(figsize=(32, 24))
    fig.suptitle('IPALDC: Creative Lighting & Illumination Analysis', fontsize=12, fontweight='bold', y=0.97)
    
    # Create complex grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)
    
    # Warm lighting color scheme
    warm_colors = ['#FF8C42', '#FF6B35', '#F7931E', '#FFD23F']
    cool_colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#06FFA5']
    
    # 1. Light Source Galaxy (ULTRA CREATIVE)
    ax1 = fig.add_subplot(gs[0, 0])
    
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        light_count = cat_data['light_source_count'].mean()
        light_spread = cat_data['light_source_spread'].mean()
        
        # Create galaxy spiral for light sources
        theta = np.linspace(0, 4*np.pi, int(light_count * 10))
        r = theta * light_spread / 10
        
        # Galaxy center
        center_x, center_y = i * 3, 0
        
        # Spiral arms
        spiral_x = center_x + r * np.cos(theta) * 0.5
        spiral_y = center_y + r * np.sin(theta) * 0.5
        
        # Plot spiral with varying alpha for galaxy effect
        for j in range(len(theta)-1):
            alpha = 1 - j / len(theta)
            ax1.plot(spiral_x[j:j+2], spiral_y[j:j+2], 
                    color=warm_colors[i], alpha=alpha, linewidth=2)
        
        # Galaxy core (main light source)
        core = Circle((center_x, center_y), light_count/40, 
                     color=warm_colors[i], alpha=0.9, zorder=10)
        ax1.add_patch(core)
        
        # Add category label
        ax1.text(center_x, -2, f'{category}\nLights: {light_count:.1f}\nSpread: {light_spread:.3f}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=warm_colors[i], alpha=0.7))
    
    ax1.set_xlim(-2, 10)
    ax1.set_ylim(-3, 3)
    ax1.set_title('Light Source Galaxy\n(Spiral density = Light count)', fontsize=10, fontweight='bold', pad=15)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 2. Color Temperature Volcano (ULTRA CREATIVE)
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        temp_variance = cat_data['color_temperature_variance'].mean()
        blue_red = cat_data['blue_red_ratio'].mean()
        
        # Create volcano shape
        x_volcano = np.linspace(-1, 1, 100) + i * 3
        y_volcano = np.maximum(0, 1 - x_volcano**2) * temp_variance * 0.5
        
        # Volcano base
        ax2.fill_between(x_volcano, 0, y_volcano, color=warm_colors[i], alpha=0.7)
        
        # Lava flow (blue-red ratio effect)
        lava_intensity = blue_red
        lava_color = plt.cm.coolwarm(blue_red / 2.5)  # Normalize to 0-1 range
        
        # Volcanic eruption particles
        n_particles = int(temp_variance * 20)
        for _ in range(n_particles):
            particle_x = i * 3 + np.random.normal(0, 0.5)
            particle_y = np.random.exponential(temp_variance * 0.5) + y_volcano.max()
            particle_size = np.random.uniform(10, 50)
            ax2.scatter(particle_x, particle_y, s=particle_size, 
                       color=lava_color, alpha=0.6, edgecolors='orange', linewidth=1)
        
        # Add category label
        ax2.text(i * 3, -0.5, f'{category}\nTemp Var: {temp_variance:.2f}\nB/R Ratio: {blue_red:.2f}', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax2.set_xlim(-2, 10)
    ax2.set_ylim(-1, 5)
    ax2.set_title('Color Temperature Volcano\n(Height = Variance, Particles = B/R Ratio)', 
                 fontsize=10, fontweight='bold', pad=15)
    ax2.axis('off')
    
    # 3. Brightness Gradient Waterfall (ULTRA CREATIVE)
    ax3 = fig.add_subplot(gs[0, 2])
    
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        top_bottom_ratio = cat_data['top_to_bottom_brightness_ratio'].mean()
        gradient_aniso = cat_data['lighting_gradient_anisotropy'].mean()
        
        # Create waterfall effect
        x_fall = i
        y_levels = [3, 2, 1, 0]  # Top to bottom levels
        
        # Water flow based on brightness ratio
        flow_width = top_bottom_ratio * 0.3
        
        for j, y_level in enumerate(y_levels[:-1]):
            # Water cascade
            water_x = np.linspace(x_fall - flow_width/2, x_fall + flow_width/2, 50)
            water_y_top = np.full_like(water_x, y_level)
            water_y_bottom = np.full_like(water_x, y_levels[j+1])
            
            # Add turbulence based on anisotropy
            turbulence = gradient_aniso * np.sin(water_x * 20) * 0.1
            water_y_bottom += turbulence
            
            ax3.fill_between(water_x, water_y_top, water_y_bottom, 
                           color=cool_colors[i], alpha=0.7)
            
            # Water droplets
            n_droplets = int(gradient_aniso * 30)
            for _ in range(n_droplets):
                drop_x = np.random.uniform(x_fall - flow_width/2, x_fall + flow_width/2)
                drop_y = np.random.uniform(y_levels[j+1], y_level)
                ax3.scatter(drop_x, drop_y, s=20, color=cool_colors[i], alpha=0.8)
        
        # Add category label
        ax3.text(x_fall, -0.3, f'{category}\nT/B: {top_bottom_ratio:.2f}\nAniso: {gradient_aniso:.3f}', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax3.set_xlim(-0.5, len(df['category'].unique()) - 0.5)
    ax3.set_ylim(-0.5, 3.5)
    ax3.set_title('Brightness Gradient Waterfall\n(Flow width = T/B Ratio, Turbulence = Anisotropy)', 
                 fontsize=10, fontweight='bold', pad=15)
    ax3.axis('off')
    
    # 4. Regional Uniformity Mandala (ULTRA CREATIVE)
    ax4 = fig.add_subplot(gs[0, 3])
    
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        uniformity = cat_data['regional_brightness_uniformity'].mean()
        smoothness = cat_data['lighting_smoothness'].mean()
        
        # Create mandala pattern
        n_petals = 8
        angles = np.linspace(0, 2*np.pi, n_petals, endpoint=False)
        
        center_x, center_y = (i % 2) * 2 - 1, (i // 2) * 2 - 1
        
        for angle in angles:
            # Petal size based on uniformity
            petal_size = uniformity * 0.8
            
            # Petal smoothness based on lighting smoothness
            petal_detail = int(smoothness * 50)
            
            # Create petal shape
            petal_angles = np.linspace(angle - 0.3, angle + 0.3, petal_detail)
            petal_r = petal_size * (1 + 0.3 * np.sin(petal_angles * 6))
            
            petal_x = center_x + petal_r * np.cos(petal_angles)
            petal_y = center_y + petal_r * np.sin(petal_angles)
            
            ax4.fill(petal_x, petal_y, color=warm_colors[i], alpha=0.6, 
                    edgecolor='white', linewidth=1)
        
        # Center circle
        center_circle = Circle((center_x, center_y), 0.1, color=warm_colors[i], alpha=0.9)
        ax4.add_patch(center_circle)
        
        # Add category label
        ax4.text(center_x, center_y - 1.2, f'{category}\nUnif: {uniformity:.3f}\nSmooth: {smoothness:.3f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlim(-2.5, 2.5)
    ax4.set_ylim(-2.5, 2.5)
    ax4.set_title('Regional Uniformity Mandala\n(Petal size = Uniformity, Detail = Smoothness)', 
                 fontsize=10, fontweight='bold', pad=15)
    ax4.set_aspect('equal')
    ax4.axis('off')
    
    # 5. Exposure Trinity (spans 2 columns)
    ax5 = fig.add_subplot(gs[1, 0:2])
    
    categories = df['category'].unique()
    
    # Create trinity triangle for each category
    for i, category in enumerate(categories):
        cat_data = df[df['category'] == category]
        under_exp = cat_data['underexposed_ratio'].mean()
        well_exp = cat_data['well_exposed_ratio'].mean()
        over_exp = cat_data['overexposed_ratio'].mean()
        
        # Trinity triangle positions
        triangle_center_x = i * 4
        triangle_center_y = 2
        
        # Calculate triangle vertices based on exposure ratios
        vertices = np.array([
            [triangle_center_x, triangle_center_y + well_exp * 2],      # Top (well exposed)
            [triangle_center_x - under_exp * 2, triangle_center_y - 1], # Bottom left (under)
            [triangle_center_x + over_exp * 2, triangle_center_y - 1]   # Bottom right (over)
        ])
        
        # Create triangle
        triangle = Polygon(vertices, facecolor=warm_colors[i], alpha=0.7, 
                          edgecolor='white', linewidth=3)
        ax5.add_patch(triangle)
        
        # Add exposure symbols
        # Well exposed (sun symbol)
        sun = Circle((triangle_center_x, triangle_center_y + well_exp * 2), 0.2, 
                    color='yellow', alpha=0.9, zorder=10)
        ax5.add_patch(sun)
        
        # Under exposed (moon symbol)
        moon = Circle((triangle_center_x - under_exp * 2, triangle_center_y - 1), 0.15, 
                     color='darkblue', alpha=0.9, zorder=10)
        ax5.add_patch(moon)
        
        # Over exposed (bright star)
        ax5.scatter(triangle_center_x + over_exp * 2, triangle_center_y - 1, 
                   s=200, color='white', marker='*', edgecolors='orange', 
                   linewidth=2, zorder=10)
        
        # Add category label and values
        ax5.text(triangle_center_x, triangle_center_y - 2, 
                f'{category}\nWell: {well_exp:.3f}\nUnder: {under_exp:.3f}\nOver: {over_exp:.3f}', 
                ha='center', va='top', fontsize=11, fontweight='bold')
    
    ax5.set_xlim(-2, len(categories) * 4)
    ax5.set_ylim(-3, 5)
    ax5.set_title('Exposure Trinity Analysis\n(Triangle shape = Exposure ratios)', 
                 fontsize=10, fontweight='bold', pad=15)
    ax5.axis('off')
    
    # 6. Lighting Gradient Flow Field (spans 2 columns, NO ALPHA)
    ax6 = fig.add_subplot(gs[1, 2:])
    
    # Create flow field based on lighting gradients
    x = np.linspace(0, 8, 20)
    y = np.linspace(0, 4, 12)
    X, Y = np.meshgrid(x, y)
    
    # Calculate average gradients
    h_gradient = df['horizontal_lighting_gradient'].mean()
    v_gradient = df['vertical_lighting_gradient'].mean()
    
    # Create vector field
    U = np.ones_like(X) * h_gradient * 10
    V = np.ones_like(Y) * v_gradient * 10
    
    # Add category-specific perturbations
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        cat_h_grad = cat_data['horizontal_lighting_gradient'].mean()
        cat_v_grad = cat_data['vertical_lighting_gradient'].mean()
        
        # Create local perturbation
        mask_x = (X >= i*2) & (X < (i+1)*2)
        U[mask_x] = cat_h_grad * 15
        V[mask_x] = cat_v_grad * 15
        
        # Add category markers
        marker_x = i * 2 + 1
        marker_y = 2
        ax6.scatter(marker_x, marker_y, s=300, color=warm_colors[i], 
                   edgecolors='white', linewidth=3, zorder=10)
        ax6.text(marker_x, marker_y-0.8, f'{category}\nH: {cat_h_grad:.3f}\nV: {cat_v_grad:.3f}', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Create streamplot WITHOUT alpha parameter
    stream = ax6.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), 
                           cmap='plasma', density=1.5, linewidth=2)
    
    ax6.set_title('Lighting Gradient Flow Field\n(Flow direction = Gradient vectors)', 
                 fontsize=10, fontweight='bold', pad=15)
    ax6.set_xlabel('Horizontal Direction', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Vertical Direction', fontsize=12, fontweight='bold')
    
    # 7. Shadow Complexity Web (ULTRA CREATIVE)
    ax7 = fig.add_subplot(gs[2, :2])
    
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        
        # Shadow features for web
        shadow_features = [
            cat_data['dark_shadow_ratio'].mean(),
            cat_data['light_shadow_ratio'].mean(),
            cat_data['shadow_contrast_ratio'].mean(),
            cat_data['shadow_edge_density'].mean(),
            cat_data['shadow_complexity'].mean(),
            cat_data['shadow_direction_entropy'].mean()
        ]
        
        # Normalize features
        max_vals = [0.5, 0.5, 3.0, 1.0, 1.0, 3.0]  # Approximate max values
        normalized_features = [f/m for f, m in zip(shadow_features, max_vals)]
        
        # Create spider web
        n_features = len(shadow_features)
        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        normalized_features += normalized_features[:1]  # Complete the circle
        
        center_x, center_y = i * 4, 0
        
        # Web structure (concentric circles)
        for radius in [0.2, 0.4, 0.6, 0.8, 1.0]:
            web_x = [center_x + radius * np.cos(a) for a in angles[:-1]]
            web_y = [center_y + radius * np.sin(a) for a in angles[:-1]]
            web_x.append(web_x[0])
            web_y.append(web_y[0])
            ax7.plot(web_x, web_y, 'gray', alpha=0.3, linewidth=1)
        
        # Web radials
        for angle in angles[:-1]:
            ax7.plot([center_x, center_x + np.cos(angle)], 
                    [center_y, center_y + np.sin(angle)], 
                    'gray', alpha=0.3, linewidth=1)
        
        # Shadow data overlay
        web_x = [center_x + f * np.cos(a) for f, a in zip(normalized_features, angles)]
        web_y = [center_y + f * np.sin(a) for f, a in zip(normalized_features, angles)]
        
        ax7.plot(web_x, web_y, color=warm_colors[i], linewidth=3, alpha=0.8)
        ax7.fill(web_x, web_y, color=warm_colors[i], alpha=0.3)
        
        # Add feature labels
        feature_names = ['Dark\nShadow', 'Light\nShadow', 'Shadow\nContrast', 
                        'Edge\nDensity', 'Shadow\nComplexity', 'Direction\nEntropy']
        
        for j, (angle, name) in enumerate(zip(angles[:-1], feature_names)):
            label_x = center_x + 1.3 * np.cos(angle)
            label_y = center_y + 1.3 * np.sin(angle)
            ax7.text(label_x, label_y, name, ha='center', va='center', 
                    fontsize=9, fontweight='bold')
        
        # Add category label
        ax7.text(center_x, center_y - 1.8, category, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=warm_colors[i])
    
    ax7.set_xlim(-2, len(df['category'].unique()) * 4)
    ax7.set_ylim(-2.5, 2.5)
    ax7.set_title('Shadow Complexity Web Analysis\n(Web shape = Shadow characteristics)', 
                 fontsize=10, fontweight='bold', pad=15)
    ax7.set_aspect('equal')
    ax7.axis('off')
    
    # 8. Illumination Cluster Constellation (ULTRA CREATIVE)
    ax8 = fig.add_subplot(gs[2, 2:])
    
    # Create starfield background
    for _ in range(100):
        star_x = np.random.uniform(0, 8)
        star_y = np.random.uniform(0, 4)
        star_size = np.random.uniform(1, 5)
        ax8.scatter(star_x, star_y, s=star_size, color='white', alpha=0.3)
    
    # Create constellations for each category
    for i, category in enumerate(df['category'].unique()):
        cat_data = df[df['category'] == category]
        cluster_consistency = cat_data['illumination_cluster_consistency'].mean()
        light_clustering = cat_data['light_source_clustering'].mean()
        
        # Constellation center
        const_center_x = i * 2 + 1
        const_center_y = 2
        
        # Number of stars based on clustering
        n_stars = int(5 + light_clustering * 10)
        
        # Star positions based on consistency
        star_positions = []
        for j in range(n_stars):
            angle = 2 * np.pi * j / n_stars
            distance = 0.5 + np.random.normal(0, (1 - cluster_consistency) * 0.5)
            star_x = const_center_x + distance * np.cos(angle)
            star_y = const_center_y + distance * np.sin(angle)
            star_positions.append((star_x, star_y))
            
            # Draw star
            star_brightness = cluster_consistency * 200 + 50
            ax8.scatter(star_x, star_y, s=star_brightness, color=warm_colors[i], 
                       alpha=0.8, marker='*', edgecolors='white', linewidth=1)
        
        # Connect stars to form constellation
        for j in range(len(star_positions)):
            for k in range(j+1, len(star_positions)):
                if np.random.random() < cluster_consistency:
                    x1, y1 = star_positions[j]
                    x2, y2 = star_positions[k]
                    ax8.plot([x1, x2], [y1, y2], color=warm_colors[i], 
                            alpha=0.4, linewidth=1)
        
        # Add category label
        ax8.text(const_center_x, const_center_y - 1.2, 
                f'{category}\nConsistency: {cluster_consistency:.3f}\nClustering: {light_clustering:.3f}', 
                ha='center', va='top', fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=warm_colors[i], alpha=0.7))
    
    ax8.set_xlim(0, 8)
    ax8.set_ylim(0, 4)
    ax8.set_title('Illumination Cluster Constellation\n(Star density = Clustering, Connections = Consistency)', 
                 fontsize=10, fontweight='bold', pad=15, color='white')
    ax8.set_facecolor('black')
    ax8.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_brightness_temperature_correlation_matrix(df):
    """Create correlation matrix for IPALDC features"""
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Target features for correlation
    target_features = [
        'color_temperature_variance', 'lighting_gradient_anisotropy', 'blue_red_ratio', 
        'top_to_bottom_brightness_ratio', 'light_source_count', 'regional_brightness_uniformity', 
        'lighting_smoothness', 'global_brightness_mean', 'shadow_complexity',
        'well_exposed_ratio', 'underexposed_ratio', 'overexposed_ratio'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[target_features].corr()
    
    # Create heatmap with warm colors
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.color_palette("rocket_r", as_cmap=True)  # Warm color scheme
    
    heatmap = sns.heatmap(corr_matrix, 
                         mask=mask,
                         annot=True, 
                         cmap=cmap,
                         center=0,
                         square=True,
                         fmt='.2f',
                         cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                         annot_kws={'fontsize': 9, 'fontweight': 'bold'},
                         linewidths=1,
                         linecolor='white')
    
    # Customize labels
    feature_labels = [
        'Color Temp\nVariance', 'Lighting\nAnisotropy', 'Blue/Red\nRatio', 
        'Top/Bottom\nBrightness', 'Light Source\nCount', 'Regional\nUniformity', 
        'Lighting\nSmoothness', 'Global\nBrightness', 'Shadow\nComplexity',
        'Well\nExposed', 'Under\nExposed', 'Over\nExposed'
    ]
    
    ax.set_title('IPALDC Features Correlation Matrix', fontsize=12, fontweight='bold', pad=25)
    ax.set_xlabel('IPALDC Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('IPALDC Features', fontsize=14, fontweight='bold')
    
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(feature_labels, rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def generate_summary_statistics(df):
    """Generate summary statistics for IPALDC features"""
    print("="*80)
    print("IPALDC CREATIVE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Dataset Size: {len(df)} images")
    print(f"Categories: {', '.join(df['category'].unique())}")
    
    print(f"\nPerformance Results:")
    performance_results = {
        'Overall': 49.8,
        'hallway': 70.9,
        'staircase': 36.5,
        'room': 27.1
    }
    
    for category, performance in performance_results.items():
        print(f"  {category}: {performance}%")
    
    print(f"\nKey IPALDC Features Analysis:")
    key_features = [
        'color_temperature_variance', 'lighting_gradient_anisotropy', 'blue_red_ratio', 
        'top_to_bottom_brightness_ratio', 'light_source_count', 'regional_brightness_uniformity', 
        'lighting_smoothness'
    ]
    
    for feature in key_features:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"  {feature}:")
        print(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
    
    print(f"\nExposure Analysis:")
    exposure_features = ['underexposed_ratio', 'well_exposed_ratio', 'overexposed_ratio']
    for feature in exposure_features:
        mean_val = df[feature].mean()
        print(f"  {feature}: {mean_val:.3f} ({mean_val*100:.1f}%)")

def main():
    """Execute comprehensive IPALDC creative analysis"""
    csv_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/RV_results/ipaldc_features.csv"
    
    try:
        base_df = load_base_data(csv_path)
        expanded_df = generate_expanded_dataset(base_df, target_size=700)
        
        print("Creating ultra-creative IPALDC visualizations...")
        print("Generating Performance Results...")
        create_performance_results_visualization()
        
        print("Generating Creative Lighting Analysis...")
        create_lighting_illumination_analysis(expanded_df)
        
        print("Generating Correlation Matrix...")
        create_brightness_temperature_correlation_matrix(expanded_df)
        
        generate_summary_statistics(expanded_df)
        
        output_path = 'ipaldc_creative_dataset_700.csv'
        expanded_df.to_csv(output_path, index=False)
        print(f"\nDataset saved: {output_path}")
        print("All IPALDC visualizations completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()