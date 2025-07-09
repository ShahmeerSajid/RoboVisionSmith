


import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

class RegionalColorDistributionSystem:
    """
    This Regional Color Distribution Analysis System analyzes color patterns across different 
    image regions to classify indoor environments without using any machine learning algorithms. 
    I designed this system to distinguish between open areas with natural color transitions, 
    hallways with uniform artificial colors, staircases with geometric color boundaries, and 
    rooms with mixed color patterns from furniture. The approach focuses on statistical analysis 
    of color distributions in different image regions, examining properties like color variance, 
    temperature, natural versus artificial color scores, and gradient smoothness. This rule-based 
    methodology provides interpretable results while capturing the distinctive color characteristics 
    that differentiate various indoor environments.
    """
    
    def __init__(self):
        self.region_divisions = {
            'vertical_regions': 3,
            'horizontal_regions': 3
        }
        
        self.color_thresholds = {
            'sky_blue_range': {
                'h_min': 100, 'h_max': 130,
                's_min': 30, 's_max': 255,
                'v_min': 80, 'v_max': 255
            },
            'natural_green_range': {
                'h_min': 40, 'h_max': 80,
                's_min': 30, 's_max': 255,
                'v_min': 50, 'v_max': 255
            },
            'artificial_lighting': {
                'warm_threshold': 0.6,
                'uniformity_threshold': 0.8
            }
        }
    
    """
    In this function, I extract comprehensive color distribution features from an image by analyzing 
    multiple color spaces and regional patterns. I start by loading the image and converting it to 
    RGB, HSV, and LAB color spaces to capture different aspects of color information. After resizing 
    for consistent processing, I analyze color patterns across different regions of the image to 
    understand how colors are distributed spatially. This multi-faceted approach includes regional 
    color analysis to detect spatial patterns, global color analysis for overall characteristics, 
    and color relationship analysis to understand transitions between regions. By combining these 
    different perspectives, I create a comprehensive feature set that captures the distinctive color 
    signatures of different indoor environments.
    """
    def extract_color_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            height, width = img_rgb.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_width, new_height))
                img_hsv = cv2.resize(img_hsv, (new_width, new_height))
                img_lab = cv2.resize(img_lab, (new_width, new_height))
                height, width = img_rgb.shape[:2]
            
            features = self._analyze_regional_colors(img_rgb, img_hsv, img_lab)
            global_features = self._analyze_global_colors(img_rgb, img_hsv, img_lab)
            features.update(global_features)
            relationship_features = self._analyze_color_relationships(img_rgb, img_hsv, img_lab)
            features.update(relationship_features)
            
            return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return self._get_default_features()
    
    """
    Here I analyze color distributions across three distinct vertical regions of the image - top, 
    middle, and bottom sections - to capture spatial color patterns that characterize different 
    environments. For each region, I calculate comprehensive color statistics including mean and 
    standard deviation for RGB channels, color variance as a key discriminator, and HSV properties 
    like hue variation and saturation levels. I also compute color entropy to measure complexity, 
    analyze dominant colors to understand the color palette, and detect specific colors like blue 
    and green that are indicative of outdoor elements. This regional analysis is crucial because 
    different environments show characteristic spatial color patterns - for example, open areas 
    often have blue skies in the top region and green vegetation in lower regions.
    """
    def _analyze_regional_colors(self, img_rgb, img_hsv, img_lab):
        height, width = img_rgb.shape[:2]
        
        top_region = img_rgb[:height//3, :]
        middle_region = img_rgb[height//3:2*height//3, :]
        bottom_region = img_rgb[2*height//3:, :]
        
        top_hsv = img_hsv[:height//3, :]
        middle_hsv = img_hsv[height//3:2*height//3, :]
        bottom_hsv = img_hsv[2*height//3:, :]
        
        features = {}
        
        regions = {
            'top': (top_region, top_hsv),
            'middle': (middle_region, middle_hsv),
            'bottom': (bottom_region, bottom_hsv)
        }
        
        for region_name, (rgb_region, hsv_region) in regions.items():
            features[f'{region_name}_mean_r'] = np.mean(rgb_region[:, :, 0])
            features[f'{region_name}_mean_g'] = np.mean(rgb_region[:, :, 1])
            features[f'{region_name}_mean_b'] = np.mean(rgb_region[:, :, 2])
            
            features[f'{region_name}_std_r'] = np.std(rgb_region[:, :, 0])
            features[f'{region_name}_std_g'] = np.std(rgb_region[:, :, 1])
            features[f'{region_name}_std_b'] = np.std(rgb_region[:, :, 2])
            
            features[f'{region_name}_color_variance'] = np.mean([
                np.var(rgb_region[:, :, 0]),
                np.var(rgb_region[:, :, 1]),
                np.var(rgb_region[:, :, 2])
            ])
            
            features[f'{region_name}_hue_std'] = np.std(hsv_region[:, :, 0])
            features[f'{region_name}_saturation_mean'] = np.mean(hsv_region[:, :, 1])
            features[f'{region_name}_value_mean'] = np.mean(hsv_region[:, :, 2])
            
            features[f'{region_name}_color_entropy'] = self._calculate_color_entropy(rgb_region)
            
            dominant_colors = self._get_dominant_colors(rgb_region)
            features[f'{region_name}_dominant_color_count'] = len(dominant_colors)
            features[f'{region_name}_dominant_color_variance'] = np.var([c[1] for c in dominant_colors])
            
            features[f'{region_name}_blue_percentage'] = self._calculate_blue_percentage(hsv_region)
            features[f'{region_name}_green_percentage'] = self._calculate_green_percentage(hsv_region)
        
        return features
    
    """
    This function analyzes global color properties that characterize the entire image rather than 
    specific regions. I examine overall color diversity through entropy calculations, assess color 
    variance across all channels, and analyze color temperature to distinguish between warm and 
    cool lighting conditions. The natural versus artificial color scoring helps me identify whether 
    an environment contains natural elements like sky and vegetation or artificial elements like 
    indoor lighting and painted walls. I also measure color uniformity to detect environments with 
    consistent lighting and calculate gradient smoothness in both vertical and horizontal directions. 
    These global metrics complement the regional analysis by providing context about the overall 
    color characteristics that define different environment types.
    """
    def _analyze_global_colors(self, img_rgb, img_hsv, img_lab):
        features = {}
        
        features['global_color_entropy'] = self._calculate_color_entropy(img_rgb)
        features['global_color_variance'] = np.mean([
            np.var(img_rgb[:, :, 0]),
            np.var(img_rgb[:, :, 1]),
            np.var(img_rgb[:, :, 2])
        ])
        
        features['color_temperature_score'] = self._analyze_color_temperature(img_rgb)
        features['natural_color_score'] = self._calculate_natural_color_score(img_hsv)
        features['artificial_color_score'] = self._calculate_artificial_color_score(img_rgb)
        features['global_color_uniformity'] = self._calculate_color_uniformity(img_rgb)
        features['vertical_gradient_smoothness'] = self._calculate_vertical_gradient(img_rgb)
        features['horizontal_gradient_smoothness'] = self._calculate_horizontal_gradient(img_rgb)
        
        return features
    
    """
    Here I analyze the relationships and transitions between different regions of the image to 
    understand how colors change spatially across the environment. I calculate correlations between 
    the mean colors of adjacent regions to detect smooth transitions versus abrupt changes, which 
    can indicate natural outdoor scenes versus structured indoor spaces. The color transition 
    measurements quantify how dramatically colors change between regions, while variance ratios 
    help identify regions with significantly different color complexity. These relationship features 
    are particularly valuable for distinguishing environments because they capture spatial color 
    patterns - for example, open areas often show smooth color transitions from sky to ground, 
    while indoor environments may have more abrupt color boundaries between walls, floors, and objects.
    """
    def _analyze_color_relationships(self, img_rgb, img_hsv, img_lab):
        height, width = img_rgb.shape[:2]
        
        top_mean = np.mean(img_rgb[:height//3, :], axis=(0, 1))
        middle_mean = np.mean(img_rgb[height//3:2*height//3, :], axis=(0, 1))
        bottom_mean = np.mean(img_rgb[2*height//3:, :], axis=(0, 1))
        
        features = {}
        
        features['top_middle_color_correlation'] = np.corrcoef(top_mean, middle_mean)[0, 1]
        features['middle_bottom_color_correlation'] = np.corrcoef(middle_mean, bottom_mean)[0, 1]
        features['top_bottom_color_correlation'] = np.corrcoef(top_mean, bottom_mean)[0, 1]
        
        features['top_to_middle_transition'] = np.linalg.norm(top_mean - middle_mean)
        features['middle_to_bottom_transition'] = np.linalg.norm(middle_mean - bottom_mean)
        features['overall_color_transition'] = np.linalg.norm(top_mean - bottom_mean)
        
        top_var = np.var(img_rgb[:height//3, :])
        middle_var = np.var(img_rgb[height//3:2*height//3, :])
        bottom_var = np.var(img_rgb[2*height//3:, :])
        
        features['top_middle_variance_ratio'] = top_var / (middle_var + 1e-6)
        features['middle_bottom_variance_ratio'] = middle_var / (bottom_var + 1e-6)
        features['max_min_variance_ratio'] = max(top_var, middle_var, bottom_var) / (min(top_var, middle_var, bottom_var) + 1e-6)
        
        return features
    
    """
    In this function, I calculate color entropy as a measure of color complexity and randomness 
    within an image region. I convert the region to grayscale to create a histogram of intensity 
    values, then normalize the histogram to create a probability distribution. Using Shannon entropy 
    formula, I quantify how evenly distributed the color intensities are across the region. Higher 
    entropy indicates a complex color pattern with many different intensity levels distributed 
    relatively evenly, which is typical of cluttered environments like rooms with furniture and 
    decorations. Lower entropy suggests simpler, more uniform color patterns found in structured 
    environments like hallways with consistent lighting and minimal color variation. This entropy 
    measure serves as a key discriminator for environment classification.
    """
    def _calculate_color_entropy(self, img_region):
        if len(img_region.shape) == 3:
            gray = cv2.cvtColor(img_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_region
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / np.sum(hist)
        
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    """
    Here I identify dominant colors in an image region using histogram analysis without machine 
    learning clustering algorithms. I reshape the image to a list of pixels and sample every 10th 
    pixel for computational efficiency. To reduce complexity, I quantize the RGB values by dividing 
    by 32, which groups similar colors together. I then create a 3D histogram in RGB space by 
    counting occurrences of each quantized color combination. Finally, I sort the colors by frequency 
    and return the top k most dominant colors along with their occurrence counts. This approach 
    provides insight into the color palette of each region, helping distinguish environments with 
    simple color schemes (like hallways) from those with complex, varied color patterns (like rooms).
    """
    def _get_dominant_colors(self, img_region, k=5):
        pixels = img_region.reshape(-1, 3)
        
        hist_3d = {}
        for pixel in pixels[::10]:
            quantized = (pixel[0]//32, pixel[1]//32, pixel[2]//32)
            hist_3d[quantized] = hist_3d.get(quantized, 0) + 1
        
        sorted_colors = sorted(hist_3d.items(), key=lambda x: x[1], reverse=True)
        return sorted_colors[:k]
    
    """
    This function calculates the percentage of blue pixels in a given image region using HSV color 
    space for more accurate color detection. I create a mask that identifies pixels falling within 
    the blue hue range (100-130 degrees), with sufficient saturation (30-255) and value (80-255) 
    to exclude very dark or very light pixels that might not appear distinctly blue. By counting 
    the pixels that match this blue criterion and dividing by the total number of pixels, I get 
    a percentage that indicates how much of the region contains blue colors. This is particularly 
    useful for detecting outdoor scenes where blue sky is prominent in the top region, helping 
    distinguish open areas from indoor environments that typically lack significant blue content.
    """
    def _calculate_blue_percentage(self, hsv_region):
        blue_mask = cv2.inRange(hsv_region,
                               np.array([100, 30, 80]),
                               np.array([130, 255, 255]))
        blue_pixels = np.sum(blue_mask > 0)
        
        total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
        
        return blue_pixels / total_pixels
    
    """
    Similar to blue detection, this function calculates the percentage of green pixels in a region 
    using HSV color space thresholds. I define the green hue range as 40-80 degrees with moderate 
    saturation and value requirements to capture natural green colors like vegetation while excluding 
    very dark or pale greens that might not be visually significant. The percentage of green pixels 
    helps identify outdoor environments where vegetation is present, particularly in the middle and 
    bottom regions of images. This green percentage serves as another indicator for distinguishing 
    open areas with natural elements from indoor environments that typically have minimal green 
    content except for artificial objects or decorations. The combination of blue and green percentages 
    provides strong evidence for natural outdoor scenes.
    """
    def _calculate_green_percentage(self, hsv_region):
        green_mask = cv2.inRange(hsv_region,
                                np.array([40, 30, 50]),
                                np.array([80, 255, 255]))
        green_pixels = np.sum(green_mask > 0)
        total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
        return green_pixels / total_pixels
    
    """
    In this function, I analyze color temperature to distinguish between warm and cool lighting 
    conditions, which can indicate different types of environments. I identify warm colors by 
    finding pixels where both red and green values exceed blue values, and cool colors where 
    blue dominates over both red and green. By calculating the ratio of warm to cool pixels, 
    I create a color temperature score that ranges from -1 (very cool) to +1 (very warm). 
    This analysis helps distinguish between outdoor environments with natural daylight (often 
    cooler color temperature) and indoor environments with artificial lighting (which can be 
    either warm incandescent lighting or cool fluorescent lighting). The color temperature score 
    provides additional context for environment classification beyond basic color detection.
    """
    def _analyze_color_temperature(self, img_rgb):
        warm_colors = np.sum((img_rgb[:, :, 0] > img_rgb[:, :, 2]) &
                            (img_rgb[:, :, 1] > img_rgb[:, :, 2]))
        cool_colors = np.sum((img_rgb[:, :, 2] > img_rgb[:, :, 0]) &
                            (img_rgb[:, :, 2] > img_rgb[:, :, 1]))
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        
        warm_ratio = warm_colors / total_pixels
        cool_ratio = cool_colors / total_pixels
        
        return (warm_ratio - cool_ratio)
    
    """
    Here I calculate how natural the color palette appears by detecting colors typically associated 
    with natural environments. I combine the blue percentage (indicating sky), green percentage 
    (indicating vegetation), and brown/earth tone detection for natural ground or tree colors. 
    For brown detection, I use a specific HSV range that captures earth tones with hues between 
    10-25 degrees and moderate saturation levels. The total natural color score represents the 
    combined presence of these natural elements in the image. Higher scores indicate environments 
    with significant natural content, which is characteristic of open areas and outdoor scenes. 
    This natural color scoring helps distinguish outdoor environments from indoor spaces that 
    typically contain more artificial colors from painted walls, furniture, and lighting.
    """
    def _calculate_natural_color_score(self, img_hsv):
        blue_score = self._calculate_blue_percentage(img_hsv)
        green_score = self._calculate_green_percentage(img_hsv)
        
        brown_mask = cv2.inRange(img_hsv,
                                np.array([10, 30, 30]),
                                np.array([25, 200, 200]))
        brown_score = np.sum(brown_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
        
        return blue_score + green_score + brown_score
    
    """
    This function calculates how artificial the color palette appears by analyzing color uniformity 
    and deviation patterns typical of artificial lighting and indoor environments. I start by 
    computing the grayscale level for each pixel, then measure how much the actual color deviates 
    from this neutral gray level across all color channels. Low color deviation indicates uniform, 
    artificial lighting conditions where colors appear flat and consistent, which is common in 
    indoor environments with artificial lighting. I convert this deviation measurement to an 
    artificial score where higher values indicate more artificial/uniform lighting conditions. 
    This scoring helps identify indoor environments like hallways, rooms, and staircases that 
    typically have artificial lighting and painted surfaces with relatively uniform colors.
    """
    def _calculate_artificial_color_score(self, img_rgb):
        gray_level = np.mean(img_rgb, axis=2)
        color_deviation = np.std(img_rgb - gray_level[:, :, np.newaxis], axis=2)
        
        artificial_score = 1.0 - (np.mean(color_deviation) / 128.0)
        return max(0, min(1, artificial_score))
    
    """
    Here I calculate overall color uniformity across the entire image by measuring the coefficient 
    of variation for each color channel. I compute the mean and standard deviation for red, green, 
    and blue channels separately, then calculate the coefficient of variation (standard deviation 
    divided by mean) for each channel. This coefficient is then converted to a uniformity score 
    where higher values indicate more uniform color distribution. Uniform colors are characteristic 
    of indoor environments with consistent artificial lighting, painted walls, and minimal color 
    variation. In contrast, outdoor environments and complex indoor spaces like rooms typically 
    show less color uniformity due to natural lighting variations, shadows, and diverse objects. 
    This uniformity measure helps distinguish structured indoor environments from more varied spaces.
    """
    def _calculate_color_uniformity(self, img_rgb):
        uniformity_scores = []
        for channel in range(3):
            mean_val = np.mean(img_rgb[:, :, channel])
            std_val = np.std(img_rgb[:, :, channel])
            cv = std_val / (mean_val + 1e-6)
            uniformity_scores.append(1.0 / (1.0 + cv))
        
        return np.mean(uniformity_scores)
    
    """
    This function calculates the smoothness of vertical color transitions by analyzing how gradually 
    colors change from top to bottom in the image. I compute the mean color for each horizontal 
    row, then calculate the differences between consecutive rows to measure the gradient. The 
    variance of these gradients indicates how smooth or abrupt the vertical color transitions are. 
    Smooth transitions (low variance) are characteristic of natural scenes like open areas where 
    sky gradually transitions to horizon and ground. Abrupt transitions (high variance) are more 
    common in structured indoor environments where walls, floors, and objects create distinct 
    color boundaries. I convert the gradient variance to a smoothness score where higher values 
    indicate smoother, more natural transitions typical of outdoor environments.
    """
    def _calculate_vertical_gradient(self, img_rgb):
        row_means = np.mean(img_rgb, axis=(1, 2))
        gradients = np.diff(row_means)
        gradient_variance = np.var(gradients)
        
        smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
        return smoothness
    
    """
    Similar to vertical gradient analysis, this function measures the smoothness of horizontal 
    color transitions across the image from left to right. I calculate the mean color for each 
    vertical column, then analyze the gradients between adjacent columns to assess transition 
    smoothness. The gradient variance is converted to a smoothness score where higher values 
    indicate more gradual color changes horizontally. This horizontal gradient analysis helps 
    identify environments with consistent lighting and color patterns versus those with irregular 
    color variations. Indoor environments like hallways often show relatively smooth horizontal 
    transitions due to uniform lighting and symmetrical features, while complex environments 
    like rooms with diverse furniture and decorations may exhibit more irregular horizontal 
    color patterns.
    """
    def _calculate_horizontal_gradient(self, img_rgb):
        col_means = np.mean(img_rgb, axis=(0, 2))
        gradients = np.diff(col_means)
        gradient_variance = np.var(gradients)
        
        smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
        return smoothness
    
    """
    When image processing fails or encounters errors, I need to return a consistent set of default 
    features to maintain the integrity of my analysis pipeline. This function creates a comprehensive 
    dictionary containing all the feature names that my system normally extracts, but sets their 
    values to 0.0. The feature set includes regional RGB statistics for top, middle, and bottom 
    regions, color properties like variance and entropy, global features such as color temperature 
    and uniformity, and relationship features that measure transitions between regions. By providing 
    these default values, I ensure that failed image processing doesn't break the classification 
    system and that the feature structure remains consistent across all processed images, enabling 
    robust batch processing even when individual images fail.
    """
    def _get_default_features(self):
        feature_names = [
            'top_mean_r', 'top_mean_g', 'top_mean_b', 'top_std_r', 'top_std_g', 'top_std_b',
            'middle_mean_r', 'middle_mean_g', 'middle_mean_b', 'middle_std_r', 'middle_std_g', 'middle_std_b',
            'bottom_mean_r', 'bottom_mean_g', 'bottom_mean_b', 'bottom_std_r', 'bottom_std_g', 'bottom_std_b',
            'top_color_variance', 'middle_color_variance', 'bottom_color_variance',
            'top_hue_std', 'middle_hue_std', 'bottom_hue_std',
            'top_saturation_mean', 'middle_saturation_mean', 'bottom_saturation_mean',
            'top_value_mean', 'middle_value_mean', 'bottom_value_mean',
            'top_color_entropy', 'middle_color_entropy', 'bottom_color_entropy',
            'top_dominant_color_count', 'middle_dominant_color_count', 'bottom_dominant_color_count',
            'top_dominant_color_variance', 'middle_dominant_color_variance', 'bottom_dominant_color_variance',
            'top_blue_percentage', 'middle_blue_percentage', 'bottom_blue_percentage',
            'top_green_percentage', 'middle_green_percentage', 'bottom_green_percentage',
            'global_color_entropy', 'global_color_variance', 'color_temperature_score',
            'natural_color_score', 'artificial_color_score', 'global_color_uniformity',
            'vertical_gradient_smoothness', 'horizontal_gradient_smoothness',
            'top_middle_color_correlation', 'middle_bottom_color_correlation', 'top_bottom_color_correlation',
            'top_to_middle_transition', 'middle_to_bottom_transition', 'overall_color_transition',
            'top_middle_variance_ratio', 'middle_bottom_variance_ratio', 'max_min_variance_ratio'
        ]
        return {name: 0.0 for name in feature_names}
    
    """
    This function processes my entire training dataset by systematically extracting color features 
    from organized folders of images representing different indoor environments. I handle folder 
    structure dynamically, mapping environment types to their corresponding directories and adapting 
    to different naming conventions. For each image, I extract the comprehensive color feature set 
    that characterizes the environment type, including regional color analysis, global color properties, 
    and spatial color relationships. The results are compiled into a structured dataset with both 
    extracted features and metadata about each image. This creates a comprehensive training set 
    that I can use for statistical analysis and rule-based classification model development, providing 
    the foundation for understanding color-based characteristics of different indoor environments.
    """
    def process_training_dataset(self, base_path):
        folder_mapping = {
            'hallway': 'hallway_test_photos',
            'staircase': 'staircase_test_photos',
            'room': 'room_test_photos',
            'open_area': 'openarea_test_photos'
        }
        
        available_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        print(f"Available folders: {available_folders}")
        
        for env_type in folder_mapping.keys():
            exact_folder = folder_mapping[env_type]
            if exact_folder not in available_folders:
                similar_folders = [f for f in available_folders if env_type in f.lower()]
                if similar_folders:
                    folder_mapping[env_type] = similar_folders[0]
                    print(f"Using '{similar_folders[0]}' for {env_type}")
        
        results = []
        total_processed = 0
        
        for env_type, folder_name in folder_mapping.items():
            folder_path = os.path.join(base_path, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"Skipping {env_type}: folder {folder_path} not found")
                continue
            
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            image_files = [f for f in os.listdir(folder_path)
                          if f.lower().endswith(image_extensions)]
            
            print(f"Processing {len(image_files)} images from {env_type} ({folder_name})...")
            
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(folder_path, img_file)
                
                try:
                    features = self.extract_color_features(img_path)
                    
                    result = {
                        'image_path': img_path,
                        'image_name': img_file,
                        'true_class': env_type,
                        **features
                    }
                    
                    results.append(result)
                    total_processed += 1
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(image_files)} images from {env_type}")
                
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
        
        print(f"\nTotal images processed: {total_processed}")
        
        df = pd.DataFrame(results)
        
        results_dir = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/RV_results"
        os.makedirs(results_dir, exist_ok=True)
        
        csv_filename = f"color_distribution_dataset_700.csv"
        csv_filepath = os.path.join(results_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        
        print(f"Features saved to: {csv_filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Classes distribution:")
        print(df['true_class'].value_counts())
        
        self._generate_feature_statistics(df)
        
        return df, csv_filepath
    
    """
    Here I generate comprehensive statistics about the extracted color features to understand how 
    different environment types are characterized by their color patterns. I focus on key features 
    that are most likely to differentiate between environments, such as blue percentages in different 
    regions, color variance measures, global color entropy, and natural versus artificial color 
    scores. For each environment class, I calculate mean values and standard deviations to identify 
    the characteristic color signatures of hallways, staircases, rooms, and open areas. This 
    statistical analysis helps me understand which color features are most discriminative and 
    guides the development of classification rules. The analysis reveals patterns like higher 
    blue percentages in open areas or greater color uniformity in hallways with artificial lighting.
    """
    def _generate_feature_statistics(self, df):
        print("\n" + "="*60)
        print("COLOR FEATURE STATISTICS BY ENVIRONMENT")
        print("="*60)
        
        key_features = [
            'top_blue_percentage', 'top_color_variance', 'middle_color_variance', 'bottom_color_variance',
            'global_color_entropy', 'natural_color_score', 'artificial_color_score',
            'vertical_gradient_smoothness', 'color_temperature_score'
        ]
        
        for env_type in df['true_class'].unique():
            env_data = df[df['true_class'] == env_type]
            print(f"\n{env_type.upper()} (n={len(env_data)}):")
            print("-" * 40)
            
            for feature in key_features:
                if feature in env_data.columns:
                    mean_val = env_data[feature].mean()
                    std_val = env_data[feature].std()
                    print(f"  {feature:<30}: {mean_val:.4f} ± {std_val:.4f}")

"""
This main function orchestrates the entire color distribution analysis pipeline from initialization 
through feature extraction to dataset creation. I start by initializing the Regional Color Distribution 
System with all the necessary parameters and thresholds for color analysis. The function then processes 
the entire training dataset, extracting comprehensive color features from each image and organizing 
them into a structured format suitable for analysis and classification development. This includes 
handling multiple environment types, processing hundreds of images, and generating detailed statistics 
about color patterns in different environments. The resulting dataset provides the foundation for 
developing rule-based classification systems that can distinguish between indoor environments based 
on their distinctive color characteristics and spatial color distributions.
"""
def main():
    print("Regional Color Distribution Analysis System")
    print("="*50)
    
    system = RegionalColorDistributionSystem()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    
    print("Extracting color features from training dataset...")
    df, csv_filename = system.process_training_dataset(base_path)
    
    print(f"\nColor features saved to {csv_filename}")
    print("Ready for classification system development!")
    
    return system, csv_filename

if __name__ == "__main__":
    system, csv_file = main()