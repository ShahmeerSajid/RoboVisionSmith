

### IPALCD ----> DATA, GLOBAL FEATURE EXTRACTION + CSV FILE CREATION (TRICKY)


import cv2
import numpy as np
import datetime
import os
import pandas as pd
from pathlib import Path
import math
from scipy import ndimage
from scipy.stats import entropy
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SmartIPALDCExtractor:
    
    def __init__(self):
        self.target_width = 480
        self.target_height = 360
        self.grid_size = 8
        
        self.shadow_threshold_low = 0.3
        self.shadow_threshold_high = 0.7
        self.brightness_bins = 32
        
        self.regions = {
            'top': (0.0, 0.33),
            'middle': (0.33, 0.67),
            'bottom': (0.67, 1.0)
        }
    
    """
    In this function, I implement enhanced preprocessing specifically designed for comprehensive 
    lighting analysis across multiple color spaces and normalization techniques. I resize images 
    while maintaining aspect ratio to ensure consistent processing, then convert to various color 
    representations including BGR, grayscale, HSV, and LAB color spaces to capture different aspects 
    of illumination information. The histogram equalization on the grayscale image provides lighting-
    invariant analysis that reduces the effects of overall brightness variations while preserving 
    relative lighting patterns. This multi-representation approach allows me to analyze both absolute 
    lighting characteristics and relative illumination patterns, ensuring that the feature extraction 
    captures lighting properties that are robust across different imaging conditions while still 
    being sensitive to the distinctive illumination characteristics that differentiate various 
    environment types.
    """
    def preprocess_image(self, image):
        h, w = image.shape[:2]
        scale_w = self.target_width / w
        scale_h = self.target_height / h
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image
        
        bgr = resized
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        
        gray_norm = cv2.equalizeHist(gray)
        
        return {
            'bgr': bgr,
            'gray': gray,
            'gray_norm': gray_norm,
            'hsv': hsv,
            'lab': lab,
            'h': h, 'w': w
        }
    
    """
    Here I analyze brightness distribution patterns across global and regional scales to understand 
    how light is distributed throughout the image. I compute comprehensive brightness statistics 
    using histogram-equalized images to reduce time-of-day effects while preserving relative lighting 
    patterns. The analysis includes statistical measures like entropy, skewness, and kurtosis that 
    characterize the shape of the brightness distribution, along with peak detection to identify 
    artificial lighting modes. I examine brightness patterns across three vertical regions (top, 
    middle, bottom) to capture spatial lighting organization, calculating inter-regional relationships 
    that reveal characteristic patterns like sky gradients in outdoor scenes or uniform artificial 
    lighting in indoor environments. The brightness uniformity measures help distinguish between 
    environments with consistent artificial lighting versus those with more variable natural lighting 
    or complex illumination patterns.
    """
    def analyze_brightness_distribution(self, images):
        gray = images['gray']
        gray_norm = images['gray_norm']
        h, w = gray.shape
        
        features = {}
        
        mean_brightness = np.mean(gray_norm) / 255.0
        brightness_std = np.std(gray_norm) / 255.0
        brightness_range = (np.max(gray_norm) - np.min(gray_norm)) / 255.0
        
        hist, _ = np.histogram(gray_norm, bins=self.brightness_bins, range=(0, 255))
        hist_norm = hist / np.sum(hist)
        
        brightness_entropy = entropy(hist_norm + 1e-10)
        brightness_skewness = self.calculate_skewness(gray_norm)
        brightness_kurtosis = self.calculate_kurtosis(gray_norm)
        
        peaks = self.detect_brightness_peaks(hist_norm)
        dominant_brightness_modes = len(peaks)
        
        features.update({
            'global_brightness_mean': mean_brightness,
            'global_brightness_std': brightness_std,
            'global_brightness_range': brightness_range,
            'brightness_entropy': brightness_entropy,
            'brightness_skewness': brightness_skewness,
            'brightness_kurtosis': brightness_kurtosis,
            'dominant_brightness_modes': dominant_brightness_modes
        })
        
        for region_name, (start_ratio, end_ratio) in self.regions.items():
            start_y = int(h * start_ratio)
            end_y = int(h * end_ratio)
            region_roi = gray_norm[start_y:end_y, :]
            
            region_mean = np.mean(region_roi) / 255.0
            region_std = np.std(region_roi) / 255.0
            
            features[f'{region_name}_brightness_mean'] = region_mean
            features[f'{region_name}_brightness_std'] = region_std
        
        top_mean = features['top_brightness_mean']
        middle_mean = features['middle_brightness_mean']
        bottom_mean = features['bottom_brightness_mean']
        
        features['top_to_bottom_brightness_ratio'] = (top_mean + 0.001) / (bottom_mean + 0.001)
        features['middle_brightness_dominance'] = middle_mean / (top_mean + bottom_mean + 0.001)
        
        regional_means = [top_mean, middle_mean, bottom_mean]
        features['regional_brightness_uniformity'] = 1.0 - (np.std(regional_means) / (np.mean(regional_means) + 0.001))
        
        return features
    
    """
    This function performs advanced shadow detection and pattern analysis to characterize how shadows 
    are formed and distributed in different environments. I use multi-threshold shadow detection to 
    identify both dark and light shadow regions, providing coverage and contrast measures that 
    distinguish between environments with complex shadow patterns versus uniform lighting. The shadow 
    edge analysis examines the sharpness and density of shadow boundaries, which differs between 
    artificial lighting (creating hard shadows) and natural lighting (creating softer shadows). I 
    analyze shadow shape complexity through contour analysis to understand the geometric characteristics 
    of shadow patterns. The directional shadow analysis uses gradient computation to assess shadow 
    direction coherence, which is important for distinguishing natural lighting with consistent 
    directional shadows from artificial lighting with multiple point sources creating complex shadow 
    patterns.
    """
    def analyze_shadow_patterns(self, images):
        gray = images['gray']
        gray_norm = images['gray_norm']
        lab = images['lab']
        h, w = gray.shape
        
        features = {}
        
        shadow_mask_dark = (gray_norm < self.shadow_threshold_low * 255).astype(np.uint8)
        shadow_mask_light = (gray_norm < self.shadow_threshold_high * 255).astype(np.uint8)
        
        dark_shadow_ratio = np.sum(shadow_mask_dark) / (h * w)
        light_shadow_ratio = np.sum(shadow_mask_light) / (h * w)
        shadow_contrast_ratio = light_shadow_ratio - dark_shadow_ratio
        
        features.update({
            'dark_shadow_ratio': dark_shadow_ratio,
            'light_shadow_ratio': light_shadow_ratio,
            'shadow_contrast_ratio': shadow_contrast_ratio
        })
        
        if np.sum(shadow_mask_dark) > 100:
            shadow_edges = cv2.Canny(shadow_mask_dark * 255, 50, 150)
            shadow_edge_density = np.sum(shadow_edges > 0) / (h * w)
            
            contours, _ = cv2.findContours(shadow_mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
                total_area = sum(cv2.contourArea(contour) for contour in contours)
                shadow_complexity = total_perimeter / (total_area + 1) if total_area > 0 else 0
            else:
                shadow_complexity = 0
            
            features['shadow_edge_density'] = shadow_edge_density
            features['shadow_complexity'] = shadow_complexity
        else:
            features['shadow_edge_density'] = 0
            features['shadow_complexity'] = 0
        
        grad_x = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        if np.sum(gradient_magnitude > 10) > 100:
            strong_gradients = gradient_direction[gradient_magnitude > 10]
            direction_histogram, _ = np.histogram(strong_gradients, bins=16, range=(-np.pi, np.pi))
            direction_entropy = entropy(direction_histogram + 1)
            features['shadow_direction_entropy'] = direction_entropy
        else:
            features['shadow_direction_entropy'] = 0
        
        return features
    
    """
    In this function, I detect and analyze light source patterns to characterize the illumination 
    infrastructure of different environments. I identify bright spots that represent potential light 
    sources using adaptive thresholding, then analyze their distribution, clustering, and size 
    characteristics. The light source counting and spatial distribution analysis helps distinguish 
    between environments with single dominant light sources (like natural lighting) versus multiple 
    artificial light sources (like indoor environments). I examine light source clustering patterns 
    and size uniformity to understand whether lighting is organized or random. The color temperature 
    analysis uses blue-to-red channel ratios as a proxy for distinguishing between warm artificial 
    lighting and cooler natural daylight. The color temperature variance measures how consistent 
    the lighting color is across the image, which helps distinguish uniform artificial lighting 
    from variable natural lighting conditions.
    """
    def analyze_light_source_characteristics(self, images):
        gray = images['gray']
        gray_norm = images['gray_norm']
        hsv = images['hsv']
        h, w = gray.shape
        
        features = {}
        
        bright_threshold = np.percentile(gray_norm, 95)
        bright_mask = (gray_norm > bright_threshold).astype(np.uint8)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_light_sources = [c for c in contours if cv2.contourArea(c) > 20]
        light_source_count = len(significant_light_sources)
        
        if significant_light_sources:
            light_centers = []
            light_sizes = []
            
            for contour in significant_light_sources:
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    light_centers.append([cx, cy])
                    light_sizes.append(cv2.contourArea(contour))
            
            if light_centers:
                light_centers = np.array(light_centers)
                
                if len(light_centers) > 1:
                    light_spread = np.std(light_centers.flatten()) / (w + h)
                    light_clustering = self.calculate_clustering_metric(light_centers)
                else:
                    light_spread = 0
                    light_clustering = 1.0
                
                light_size_uniformity = 1.0 - (np.std(light_sizes) / (np.mean(light_sizes) + 1))
                
                features.update({
                    'light_source_count': light_source_count,
                    'light_source_spread': light_spread,
                    'light_source_clustering': light_clustering,
                    'light_size_uniformity': light_size_uniformity
                })
            else:
                features.update({
                    'light_source_count': 0,
                    'light_source_spread': 0,
                    'light_source_clustering': 0,
                    'light_size_uniformity': 0
                })
        else:
            features.update({
                'light_source_count': 0,
                'light_source_spread': 0,
                'light_source_clustering': 0,
                'light_size_uniformity': 0
            })
        
        bgr = images['bgr']
        blue_channel = bgr[:, :, 0].astype(np.float32)
        red_channel = bgr[:, :, 2].astype(np.float32)
        
        blue_red_ratio = np.mean(blue_channel) / (np.mean(red_channel) + 1)
        
        blue_red_per_pixel = blue_channel / (red_channel + 1)
        color_temp_variance = np.std(blue_red_per_pixel)
        
        features.update({
            'blue_red_ratio': blue_red_ratio,
            'color_temperature_variance': color_temp_variance
        })
        
        return features
    
    """
    Here I analyze lighting patterns using spatial grid analysis to understand how illumination 
    is organized spatially across the image. I divide the image into a regular grid and calculate 
    brightness and variance statistics for each cell, providing a detailed map of lighting 
    distribution. The spatial uniformity measures reveal whether lighting is consistent across 
    the environment or shows significant spatial variation. I compute lighting gradients in both 
    horizontal and vertical directions to detect directional lighting patterns, with the gradient 
    anisotropy measure indicating whether lighting has a dominant directional component. The 
    pattern entropy calculations assess the complexity and randomness of spatial lighting organization. 
    This spatial analysis is particularly effective for distinguishing between indoor environments 
    with uniform artificial lighting, outdoor environments with directional natural lighting, 
    and complex indoor spaces with mixed lighting sources creating irregular illumination patterns.
    """
    def analyze_spatial_lighting_patterns(self, images):
        gray_norm = images['gray_norm']
        h, w = gray_norm.shape
        
        features = {}
        
        grid_h = h // self.grid_size
        grid_w = w // self.grid_size
        
        grid_brightnesses = []
        grid_variances = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_y = i * grid_h
                end_y = min((i + 1) * grid_h, h)
                start_x = j * grid_w
                end_x = min((j + 1) * grid_w, w)
                
                grid_cell = gray_norm[start_y:end_y, start_x:end_x]
                cell_brightness = np.mean(grid_cell) / 255.0
                cell_variance = np.std(grid_cell) / 255.0
                
                grid_brightnesses.append(cell_brightness)
                grid_variances.append(cell_variance)
        
        grid_brightnesses = np.array(grid_brightnesses)
        grid_variances = np.array(grid_variances)
        
        spatial_brightness_uniformity = 1.0 - (np.std(grid_brightnesses) / (np.mean(grid_brightnesses) + 0.001))
        spatial_variance_uniformity = 1.0 - (np.std(grid_variances) / (np.mean(grid_variances) + 0.001))
        
        grid_2d = grid_brightnesses.reshape(self.grid_size, self.grid_size)
        horizontal_gradient = np.mean(np.abs(np.gradient(grid_2d, axis=1)))
        vertical_gradient = np.mean(np.abs(np.gradient(grid_2d, axis=0)))
        
        gradient_anisotropy = abs(horizontal_gradient - vertical_gradient) / (horizontal_gradient + vertical_gradient + 0.001)
        
        features.update({
            'spatial_brightness_uniformity': spatial_brightness_uniformity,
            'spatial_variance_uniformity': spatial_variance_uniformity,
            'horizontal_lighting_gradient': horizontal_gradient,
            'vertical_lighting_gradient': vertical_gradient,
            'lighting_gradient_anisotropy': gradient_anisotropy
        })
        
        brightness_pattern_entropy = entropy(grid_brightnesses + 0.001)
        variance_pattern_entropy = entropy(grid_variances + 0.001)
        
        features.update({
            'brightness_pattern_entropy': brightness_pattern_entropy,
            'variance_pattern_entropy': variance_pattern_entropy
        })
        
        return features
    
    """
    This function analyzes illumination quality and consistency indicators that characterize the 
    overall lighting conditions and their impact on image quality. I compute local contrast analysis 
    using neighborhood filtering to understand how lighting creates contrast variations across the 
    image, with contrast uniformity indicating whether lighting creates consistent or variable 
    contrast patterns. The exposure quality assessment identifies underexposed, overexposed, and 
    well-exposed regions, providing insight into lighting adequacy and camera exposure settings. 
    I calculate lighting smoothness using gradient analysis to distinguish between smooth artificial 
    lighting and more variable natural lighting conditions. The illumination consistency analysis 
    uses K-means clustering to group similar intensity regions and assess how consistent the 
    lighting is across similar areas of the image. These quality indicators help distinguish 
    between well-lit indoor environments, challenging lighting conditions, and natural outdoor 
    lighting scenarios.
    """
    def analyze_illumination_quality_indicators(self, images):
        gray = images['gray']
        gray_norm = images['gray_norm']
        lab = images['lab']
        h, w = gray.shape
        
        features = {}
        
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray_norm.astype(np.float32), -1, kernel)
        local_contrast = np.abs(gray_norm.astype(np.float32) - local_mean)
        
        mean_local_contrast = np.mean(local_contrast) / 255.0
        contrast_uniformity = 1.0 - (np.std(local_contrast) / (np.mean(local_contrast) + 1))
        
        features.update({
            'mean_local_contrast': mean_local_contrast,
            'contrast_uniformity': contrast_uniformity
        })
        
        underexposed_ratio = np.sum(gray_norm < 25) / (h * w)
        overexposed_ratio = np.sum(gray_norm > 230) / (h * w)
        well_exposed_ratio = 1.0 - underexposed_ratio - overexposed_ratio
        
        features.update({
            'underexposed_ratio': underexposed_ratio,
            'overexposed_ratio': overexposed_ratio,
            'well_exposed_ratio': well_exposed_ratio
        })
        
        grad_x = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        lighting_smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude) / 255.0)
        
        features['lighting_smoothness'] = lighting_smoothness
        
        try:
            pixel_data = gray_norm.reshape(-1, 1)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pixel_data)
            cluster_centers = kmeans.cluster_centers_.flatten()
            
            cluster_consistency = 1.0 - (np.std(cluster_centers) / (np.mean(cluster_centers) + 1))
            features['illumination_cluster_consistency'] = cluster_consistency
        except:
            features['illumination_cluster_consistency'] = 0.5
        
        return features
    
    """
    In this function, I calculate the skewness of the data distribution to understand the asymmetry 
    of brightness values in the image. Skewness provides important information about lighting 
    characteristics - positive skewness indicates a distribution with many dark pixels and fewer 
    bright pixels (typical of underlit scenes), while negative skewness suggests many bright pixels 
    with fewer dark pixels (typical of overlit or outdoor scenes). Zero skewness indicates a 
    symmetric distribution around the mean brightness. I normalize the skewness calculation by 
    the standard deviation to make it scale-invariant, and handle the edge case where standard 
    deviation is zero to avoid division errors. This statistical measure helps distinguish between 
    different lighting conditions and provides insight into the overall illumination characteristics 
    that distinguish various environment types based on their typical brightness distribution patterns.
    """
    def calculate_skewness(self, data):
        data_flat = data.flatten()
        mean = np.mean(data_flat)
        std = np.std(data_flat)
        if std == 0:
            return 0
        skewness = np.mean(((data_flat - mean) / std) ** 3)
        return skewness
    
    """
    Here I calculate the kurtosis of the data distribution, which measures the "tailedness" or 
    peakedness of the brightness distribution compared to a normal distribution. High kurtosis 
    indicates a distribution with heavy tails and a sharp peak, often seen in images with high 
    contrast or extreme lighting conditions. Low kurtosis suggests a flatter distribution with 
    lighter tails, typical of evenly lit scenes. I subtract 3 from the fourth moment to get 
    excess kurtosis, where zero indicates normal distribution characteristics. This statistical 
    measure helps characterize lighting patterns - artificial lighting often creates distributions 
    with specific kurtosis characteristics due to uniform illumination, while natural lighting 
    may show different kurtosis patterns due to varying lighting conditions. The kurtosis measure 
    complements skewness to provide a complete picture of brightness distribution characteristics 
    that distinguish different environment types.
    """
    def calculate_kurtosis(self, data):
        data_flat = data.flatten()
        mean = np.mean(data_flat)
        std = np.std(data_flat)
        if std == 0:
            return 0
        kurtosis = np.mean(((data_flat - mean) / std) ** 4) - 3
        return kurtosis
    
    """
    This function detects peaks in the brightness histogram to identify distinct brightness modes 
    that characterize different lighting conditions. I use simple peak detection by comparing 
    each histogram bin with its neighbors and identifying local maxima that exceed a significance 
    threshold. Multiple peaks often indicate mixed lighting conditions (like natural light combined 
    with artificial lighting), while single peaks suggest uniform lighting conditions. The number 
    and characteristics of brightness peaks help distinguish between environment types - outdoor 
    scenes may show bimodal distributions from sky and ground, indoor artificial lighting often 
    creates unimodal distributions, and complex indoor environments may show multiple peaks from 
    mixed lighting sources. This peak analysis provides important information about the lighting 
    infrastructure and helps characterize the illumination patterns that distinguish different 
    types of indoor and outdoor environments.
    """
    def detect_brightness_peaks(self, histogram):
        peaks = []
        for i in range(1, len(histogram) - 1):
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                if histogram[i] > 0.05:
                    peaks.append(i)
        return peaks
    
    """
    In this function, I calculate how clustered a set of points are in spatial coordinates, which 
    is important for understanding light source distribution patterns. I compute pairwise distances 
    between all light source points, then calculate the mean distance and normalize it by the 
    maximum possible distance in the image. The clustering metric ranges from 0 (maximally spread) 
    to 1 (highly clustered), providing insight into whether light sources are concentrated in 
    specific areas or distributed throughout the environment. High clustering suggests localized 
    lighting (like a single window or concentrated artificial lights), while low clustering indicates 
    distributed lighting sources. This spatial clustering analysis helps distinguish between different 
    types of environments based on their characteristic lighting arrangements - outdoor scenes 
    often have single dominant light sources, while indoor environments may show various clustering 
    patterns depending on the lighting infrastructure and architectural design.
    """
    def calculate_clustering_metric(self, points):
        if len(points) < 2:
            return 1.0
        
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        mean_distance = np.mean(distances)
        max_possible_distance = np.sqrt(2) * max(self.target_width, self.target_height)
        
        clustering = 1.0 - (mean_distance / max_possible_distance)
        return max(0, clustering)
    
    """
    This comprehensive feature extraction function orchestrates the entire IPALDC analysis pipeline 
    to extract illumination and lighting features that characterize different indoor environments. 
    I start by loading and preprocessing the image through multiple color space conversions and 
    normalization techniques, then systematically analyze five key categories of illumination 
    characteristics. The brightness distribution analysis captures global and regional lighting 
    patterns, shadow pattern analysis examines shadow characteristics and directionality, light 
    source analysis identifies and characterizes illumination infrastructure, spatial lighting 
    analysis reveals the spatial organization of illumination, and quality analysis assesses 
    lighting adequacy and consistency. This multi-faceted approach captures the full range of 
    illumination cues that distinguish outdoor areas with natural lighting, indoor spaces with 
    artificial illumination, and various types of indoor environments based on their characteristic 
    lighting patterns and spatial light distribution properties.
    """
    def extract_features(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            images = self.preprocess_image(image)
            
            features = {'image_path': image_path}
            
            brightness_features = self.analyze_brightness_distribution(images)
            features.update(brightness_features)
            
            shadow_features = self.analyze_shadow_patterns(images)
            features.update(shadow_features)
            
            light_source_features = self.analyze_light_source_characteristics(images)
            features.update(light_source_features)
            
            spatial_features = self.analyze_spatial_lighting_patterns(images)
            features.update(spatial_features)
            
            quality_features = self.analyze_illumination_quality_indicators(images)
            features.update(quality_features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

"""
This function processes all images from organized folders to extract comprehensive IPALDC features 
and compile them into a structured dataset for illumination-based environment classification analysis. 
I systematically iterate through folders containing different environment types, applying the complete 
IPALDC feature extraction pipeline to each image while handling various image formats and providing 
progress tracking for large datasets. After extracting illumination features from all images, I 
organize the results into a pandas DataFrame with proper column ordering and save the complete 
dataset to CSV format. This comprehensive dataset creation process provides the foundation for 
developing and evaluating classification algorithms based on illumination pattern analysis. The 
resulting dataset contains rich lighting and illumination features that capture the distinctive 
illumination characteristics of different indoor environments, enabling research into lighting-based 
environment classification for applications in robotics navigation, computer vision, and automated 
scene understanding systems.
"""
def process_images_to_csv():
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    folders = {
        'hallway': 'hallway_test_photos',
        'staircase': 'staircase_test_photos', 
        'room': 'room_test_photos',
        'openarea': 'openarea_test_photos'
    }
    
    extractor = SmartIPALDCExtractor()
    all_features = []
    
    print("Starting Smart IPALDC feature extraction...")
    
    for category, folder_name in folders.items():
        folder_path = os.path.join(base_path, folder_name)
        print(f"\nProcessing {category} images from {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist!")
            continue
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {category}")
        
        for i, image_path in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f"  Processing {category} image {i+1}/{len(image_files)}")
            
            features = extractor.extract_features(str(image_path))
            if features is not None:
                features['category'] = category
                features['filename'] = image_path.name
                all_features.append(features)
            else:
                print(f"    Failed to process {image_path.name}")
    
    if all_features:
        df = pd.DataFrame(all_features)
        
        column_order = ['filename', 'category', 'image_path'] + \
                      [col for col in df.columns if col not in ['filename', 'category', 'image_path']]
        df = df[column_order]
        
        output_file = os.path.join(base_path, 'ipaldc_features.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\nSmart IPALDC feature extraction completed!")
        print(f"Total images processed: {len(all_features)}")
        print(f"Features saved to: {output_file}")
        print(f"Features extracted per image: {len(df.columns) - 3}")
        
        print(f"\nCategory distribution:")
        print(df['category'].value_counts())
        
        feature_cols = [col for col in df.columns if col not in ['filename', 'category', 'image_path']]
        print(f"\nExtracted {len(feature_cols)} IPALDC features:")
        for i, col in enumerate(feature_cols[:10], 1):
            print(f"{i:2d}. {col}")
        if len(feature_cols) > 10:
            print(f"... and {len(feature_cols) - 10} more features")
        
        return df
    else:
        print("No features were extracted. Check your image paths and formats.")
        return None

if __name__ == "__main__":
    df = process_images_to_csv()
    if df is not None:
        print(f"\nSample of extracted features is as follows --> just for a better understanding though ... :")
        print(df.head())