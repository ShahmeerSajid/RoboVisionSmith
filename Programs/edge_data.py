
### Extracting relevant data and global features from 700 images, for SEOAEPR

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import datetime
from datetime import datetime

class EdgeOrientationSystem:
    """
    This class creates a comprehensive system for analyzing edge orientations in images to classify 
    different indoor environments. I designed it to extract meaningful features from edge patterns 
    that can distinguish between hallways, staircases, rooms, and open areas. The system works by 
    detecting edges in images, calculating their orientations, and then extracting statistical 
    features that capture the geometric characteristics of each environment type. This approach 
    allows me to build a classifier that can automatically identify indoor spaces based on their 
    structural edge patterns.
    """
    
    def __init__(self):
        self.orientation_bins = {
            'horizontal': (0, 22.5, 157.5, 180),     
            'vertical': (67.5, 112.5),                
            'diagonal_45': (22.5, 67.5),              
            'diagonal_135': (112.5, 157.5)            
        }
        
        self.classification_stats = None
        self.feature_thresholds = None
    
    """
    In this function, I extract edges from an input image using Canny edge detection with careful 
    preprocessing to ensure consistent results. I start by converting the image to grayscale and 
    resizing it if necessary to maintain processing efficiency. Then I apply Gaussian blur to reduce 
    noise that could interfere with edge detection. The Canny algorithm helps me identify the most 
    significant edges in the image by using two thresholds - a low threshold for edge linking and 
    a high threshold for strong edge detection. This preprocessing pipeline ensures that I get clean, 
    reliable edge maps that form the foundation for my orientation analysis.
    """
    def extract_edges(self, image_path, canny_low=50, canny_high=150):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            height, width = gray.shape
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height))
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            edges = cv2.Canny(blurred, canny_low, canny_high)
            
            return edges, gray.shape
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None
    
    """
    Here I calculate the orientation angles of the detected edges by computing gradients in both x and y 
    directions using Sobel operators. The gradients tell me how rapidly the pixel intensities change in 
    each direction, which allows me to determine the direction perpendicular to the edge. I use the 
    arctangent of the gradient ratio to get the actual angle, then convert it to a 0-180 degree range 
    for consistency. To focus on the most significant edges, I filter out weak edges by keeping only 
    those above a certain magnitude threshold. This gives me a collection of orientation angles that 
    represent the dominant directional patterns in the image.
    """
    def calculate_edge_orientations(self, edges):
        if edges is None:
            return None, None
        
        grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        orientation = np.where(orientation < 0, orientation + 180, orientation)
        
        if np.max(magnitude) > 0:
            edge_threshold = np.percentile(magnitude[magnitude > 0], 70)
            strong_edges_mask = magnitude > edge_threshold
            
            if np.sum(strong_edges_mask) > 0:
                return orientation[strong_edges_mask], magnitude[strong_edges_mask]
        
        return None, None
    
    """
    This function extracts a comprehensive set of features from the edge orientations that capture the 
    geometric characteristics of different indoor environments. I calculate ratios for horizontal, vertical, 
    and diagonal edges by grouping orientations into specific angle ranges and weighting them by their 
    edge strength. Beyond basic ratios, I compute statistical measures like entropy to quantify the 
    randomness of edge directions, and geometric regularity to measure how structured the environment is. 
    I also include metrics for edge density, strength distribution, and dominant orientation patterns. 
    These features work together to create a signature that can distinguish between different types of 
    indoor spaces based on their architectural and structural characteristics.
    """
    def extract_comprehensive_features(self, orientations, magnitudes, img_shape):
        if orientations is None or len(orientations) == 0:
            return self._get_empty_features()
        
        total_magnitude = np.sum(magnitudes)
        if total_magnitude == 0:
            return self._get_empty_features()
            
        weights = magnitudes / total_magnitude
        features = {}
        
        horizontal_mask = ((orientations <= 22.5) | (orientations >= 157.5))
        features['horizontal_ratio'] = np.sum(weights[horizontal_mask])
        features['horizontal_count'] = np.sum(horizontal_mask)
        
        vertical_mask = ((orientations >= 67.5) & (orientations <= 112.5))
        features['vertical_ratio'] = np.sum(weights[vertical_mask])
        features['vertical_count'] = np.sum(vertical_mask)
        
        diagonal_45_mask = ((orientations >= 22.5) & (orientations <= 67.5))
        features['diagonal_45_ratio'] = np.sum(weights[diagonal_45_mask])
        features['diagonal_45_count'] = np.sum(diagonal_45_mask)
        
        diagonal_135_mask = ((orientations >= 112.5) & (orientations <= 157.5))
        features['diagonal_135_ratio'] = np.sum(weights[diagonal_135_mask])
        features['diagonal_135_count'] = np.sum(diagonal_135_mask)
        
        features['total_diagonal_ratio'] = features['diagonal_45_ratio'] + features['diagonal_135_ratio']
        features['total_diagonal_count'] = features['diagonal_45_count'] + features['diagonal_135_count']
        
        features['mean_edge_strength'] = np.mean(magnitudes)
        features['max_edge_strength'] = np.max(magnitudes)
        features['std_edge_strength'] = np.std(magnitudes)
        features['edge_density'] = len(orientations) / (img_shape[0] * img_shape[1])
        
        features['orientation_entropy'] = self._calculate_entropy(orientations)
        features['orientation_std'] = np.std(orientations)
        features['orientation_range'] = np.max(orientations) - np.min(orientations)
        
        features['geometric_regularity'] = (features['horizontal_ratio'] + 
                                          features['vertical_ratio'] + 
                                          features['total_diagonal_ratio'])
        
        orientation_types = ['horizontal', 'vertical', 'diagonal_45', 'diagonal_135']
        ratios = [features[f'{ot}_ratio'] for ot in orientation_types]
        features['dominant_orientation_idx'] = np.argmax(ratios)
        features['dominant_orientation_strength'] = max(ratios)
        
        features['vertical_horizontal_ratio'] = (features['vertical_ratio'] / 
                                               (features['horizontal_ratio'] + 1e-6))
        features['diagonal_geometric_ratio'] = (features['total_diagonal_ratio'] / 
                                              (features['geometric_regularity'] + 1e-6))
        
        ratios_array = np.array([features['horizontal_ratio'], features['vertical_ratio'],
                               features['diagonal_45_ratio'], features['diagonal_135_ratio']])
        features['pattern_uniformity'] = 1.0 - np.std(ratios_array)
        
        hist, _ = np.histogram(orientations, bins=36, range=(0, 180))
        hist_normalized = hist / np.sum(hist)
        features['orientation_concentration'] = np.max(hist_normalized)
        features['orientation_spread'] = np.sum(hist_normalized > 0.02)
        
        return features
    
    """
    When image processing fails or no edges are detected, I need to return a consistent feature structure 
    to maintain the integrity of my analysis pipeline. This function creates a dictionary with all the 
    same feature names I would normally extract, but sets their values to zero. This ensures that my 
    classification system can still process the data without crashing, while clearly indicating that 
    no meaningful edge information was extracted from the image. It acts as a safety net that keeps 
    my batch processing running smoothly even when individual images fail to process correctly.
    """
    def _get_empty_features(self):
        feature_names = [
            'horizontal_ratio', 'horizontal_count', 'vertical_ratio', 'vertical_count',
            'diagonal_45_ratio', 'diagonal_45_count', 'diagonal_135_ratio', 'diagonal_135_count',
            'total_diagonal_ratio', 'total_diagonal_count', 'mean_edge_strength', 'max_edge_strength',
            'std_edge_strength', 'edge_density', 'orientation_entropy', 'orientation_std',
            'orientation_range', 'geometric_regularity', 'dominant_orientation_idx',
            'dominant_orientation_strength', 'vertical_horizontal_ratio', 'diagonal_geometric_ratio',
            'pattern_uniformity', 'orientation_concentration', 'orientation_spread'
        ]
        return {name: 0.0 for name in feature_names}
    
    """
    In this function, I calculate the entropy of the edge orientation distribution to quantify how random 
    or uniform the angles are. Entropy serves as a measure of disorder—higher values indicate that the 
    edge directions are spread out evenly, showing no clear pattern. On the other hand, lower entropy 
    means certain orientations dominate the image, suggesting structured or repetitive patterns. This 
    helps me understand the complexity of edge arrangements in the image, which can be useful for 
    distinguishing between textures, environments, or object types in classification tasks. I use a 
    histogram-based approach to calculate the probability distribution of orientations first.
    """
    def _calculate_entropy(self, orientations):
        hist, _ = np.histogram(orientations, bins=18, range=(0, 180))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    """
    This function processes my entire training dataset by iterating through organized folders of images 
    representing different indoor environments. I handle the folder structure dynamically, mapping 
    environment types to their corresponding photo directories and adapting to different naming conventions. 
    For each image, I extract edges, calculate orientations, and compute the full feature set that 
    characterizes that environment type. The results are compiled into a comprehensive dataset that 
    includes both the extracted features and metadata about each image. This creates a structured 
    training set that I can use to build classification models and analyze the distinguishing 
    characteristics of different indoor spaces.
    """
    def process_training_dataset(self, base_path):
        folder_mapping = {
            'hallway': 'hallway_photos',
            'staircase': 'staircase_photos', 
            'room': 'room_photos',
            'open_area': 'openarea_photos'
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
                else:
                    print(f"Warning: No folder found for {env_type}")
        
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
                    edges, img_shape = self.extract_edges(img_path)
                    if edges is None:
                        print(f"Failed to extract edges from {img_file}")
                        continue
                    
                    orientations, magnitudes = self.calculate_edge_orientations(edges)
                    features = self.extract_comprehensive_features(orientations, magnitudes, img_shape)
                    
                    result = {
                        'image_path': img_path,
                        'image_name': img_file,
                        'true_class': env_type,
                        'image_width': img_shape[1],
                        'image_height': img_shape[0],
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
        
        csv_filename = f"edge_orientation_features_700.csv"
        csv_filepath = os.path.join(results_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        
        print(f"Features saved to: {csv_filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Classes distribution:")
        print(df['true_class'].value_counts())
        
        self._generate_feature_statistics(df)
        
        return df, csv_filepath
    
    """
    Here I generate comprehensive statistics about the extracted features to understand how different 
    environment types are characterized by their edge patterns. I calculate mean values and standard 
    deviations for key features across each environment class, which helps me identify the distinguishing 
    characteristics of hallways, staircases, rooms, and open areas. Additionally, I perform class 
    separability analysis to determine which features are most effective at distinguishing between 
    different environment types. This statistical analysis guides me in understanding which features 
    are most important for classification and helps me validate that my feature extraction is capturing 
    meaningful differences between indoor environments.
    """
    def _generate_feature_statistics(self, df):
        print("\n" + "="*60)
        print("FEATURE STATISTICS BY ENVIRONMENT")
        print("="*60)
        
        feature_cols = [col for col in df.columns if col not in 
                       ['image_path', 'image_name', 'true_class', 'image_width', 'image_height']]
        
        for env_type in df['true_class'].unique():
            env_data = df[df['true_class'] == env_type]
            print(f"\n{env_type.upper()} (n={len(env_data)}):")
            print("-" * 40)
            
            key_features = ['horizontal_ratio', 'vertical_ratio', 'total_diagonal_ratio', 
                          'orientation_entropy', 'mean_edge_strength', 'geometric_regularity']
            
            for feature in key_features:
                if feature in env_data.columns:
                    mean_val = env_data[feature].mean()
                    std_val = env_data[feature].std()
                    print(f"  {feature:<25}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n" + "-"*40)
        print("CLASS SEPARABILITY ANALYSIS")
        print("-"*40)
        
        key_features = ['horizontal_ratio', 'vertical_ratio', 'total_diagonal_ratio', 'orientation_entropy']
        
        for feature in key_features:
            if feature in df.columns:
                class_means = df.groupby('true_class')[feature].mean()
                overall_std = df[feature].std()
                separation = (max(class_means) - min(class_means)) / overall_std
                print(f"{feature:<25}: separation ratio = {separation:.2f}")
    
    """
    In this function, I load the previously extracted features from a CSV file and use them to create 
    classification thresholds for each environment type. I calculate statistical measures like means, 
    standard deviations, and quartiles for key features within each environment class. These statistics 
    form the basis of my classification system by establishing the expected ranges and distributions 
    for different features in each environment type. The thresholds help me determine how well a new 
    image's features match the patterns I've learned from the training data. This approach creates 
    a robust foundation for classifying new images based on their similarity to the training examples.
    """
    def load_features_and_create_classifier(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded features from {csv_path}")
        print(f"Dataset shape: {df.shape}")
        self.feature_thresholds = self._calculate_optimal_thresholds(df)
        self.classification_stats = df.groupby('true_class').agg({
            'horizontal_ratio': ['mean', 'std'],
            'vertical_ratio': ['mean', 'std'],
            'total_diagonal_ratio': ['mean', 'std'],
            'orientation_entropy': ['mean', 'std'],
            'mean_edge_strength': ['mean', 'std'],
            'geometric_regularity': ['mean', 'std']
        }).round(4)
        
        print("Classification thresholds calculated!")
        return df
    
    """
    This function calculates the optimal classification thresholds for each environment type based on 
    the statistical distribution of features in my training data. I compute multiple statistical measures 
    including means, standard deviations, and quartile ranges for each key feature within each environment 
    class. These thresholds capture the typical range of values I expect to see for different features 
    in each environment type. By using quartiles, I can establish robust boundaries that aren't overly 
    sensitive to outliers in the training data. This statistical approach creates a comprehensive profile 
    for each environment type that I can use to evaluate how well new images match the learned patterns.
    """
    def _calculate_optimal_thresholds(self, df):
        thresholds = {}
        
        for env_type in df['true_class'].unique():
            env_data = df[df['true_class'] == env_type]
            thresholds[env_type] = {
                'horizontal_ratio': {
                    'mean': env_data['horizontal_ratio'].mean(),
                    'std': env_data['horizontal_ratio'].std(),
                    'min': env_data['horizontal_ratio'].quantile(0.25),
                    'max': env_data['horizontal_ratio'].quantile(0.75)
                },
                'vertical_ratio': {
                    'mean': env_data['vertical_ratio'].mean(),
                    'std': env_data['vertical_ratio'].std(),
                    'min': env_data['vertical_ratio'].quantile(0.25),
                    'max': env_data['vertical_ratio'].quantile(0.75)
                },
                'total_diagonal_ratio': {
                    'mean': env_data['total_diagonal_ratio'].mean(),
                    'std': env_data['total_diagonal_ratio'].std(),
                    'min': env_data['total_diagonal_ratio'].quantile(0.25),
                    'max': env_data['total_diagonal_ratio'].quantile(0.75)
                },
                'orientation_entropy': {
                    'mean': env_data['orientation_entropy'].mean(),
                    'std': env_data['orientation_entropy'].std(),
                    'min': env_data['orientation_entropy'].quantile(0.25),
                    'max': env_data['orientation_entropy'].quantile(0.75)
                },
                'geometric_regularity': {
                    'mean': env_data['geometric_regularity'].mean(),
                    'std': env_data['geometric_regularity'].std(),
                    'min': env_data['geometric_regularity'].quantile(0.25),
                    'max': env_data['geometric_regularity'].quantile(0.75)
                }
            }
        
        return thresholds
    
    """
    This function performs the actual classification of a new image by comparing its extracted features 
    against the learned thresholds from my training data. I extract the same comprehensive feature set 
    from the new image that I used during training, then calculate similarity scores for each environment 
    type. The scoring system uses weighted combinations of individual feature scores, where the weights 
    are tailored to each environment type based on their characteristic patterns. For example, I give 
    higher weight to horizontal and vertical features for hallways, but emphasize diagonal features for 
    staircases. The final classification is determined by the highest scoring environment type, along 
    with confidence measures and detailed scoring breakdowns.
    """
    def classify_new_image(self, image_path):
        if self.feature_thresholds is None:
            raise ValueError("Must load training features first using load_features_and_create_classifier()")
        
        edges, img_shape = self.extract_edges(image_path)
        if edges is None:
            return None
        
        orientations, magnitudes = self.calculate_edge_orientations(edges)
        features = self.extract_comprehensive_features(orientations, magnitudes, img_shape)
        
        scores = {}
        
        for env_type, thresholds in self.feature_thresholds.items():
            score = 0
            
            h_score = self._calculate_feature_score(features['horizontal_ratio'], 
                                                  thresholds['horizontal_ratio'])
            
            v_score = self._calculate_feature_score(features['vertical_ratio'],
                                                  thresholds['vertical_ratio'])
            
            d_score = self._calculate_feature_score(features['total_diagonal_ratio'],
                                                  thresholds['total_diagonal_ratio'])
            
            e_score = self._calculate_feature_score(features['orientation_entropy'],
                                                  thresholds['orientation_entropy'])
            
            g_score = self._calculate_feature_score(features['geometric_regularity'],
                                                  thresholds['geometric_regularity'])
            
            if env_type == 'hallway':
                score = h_score * 0.3 + v_score * 0.3 + d_score * 0.1 + e_score * 0.1 + g_score * 0.2
            elif env_type == 'staircase':
                score = h_score * 0.1 + v_score * 0.1 + d_score * 0.5 + e_score * 0.1 + g_score * 0.2
            elif env_type == 'room':
                score = h_score * 0.2 + v_score * 0.2 + d_score * 0.2 + e_score * 0.3 + g_score * 0.1
            elif env_type == 'open_area':
                score = h_score * 0.1 + v_score * 0.1 + d_score * 0.1 + e_score * 0.4 + g_score * 0.3
            
            scores[env_type] = score
        
        predicted_class = max(scores, key=scores.get)
        confidence = scores[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_scores': scores,
            'features': features
        }
    
    """
    Here I calculate how well a specific feature value matches the expected range for a particular 
    environment type. I compute the normalized distance between the observed feature value and the 
    expected mean value, using the standard deviation to scale this distance appropriately. This 
    normalization ensures that features with different scales and variabilities are compared fairly. 
    The distance is then converted to a similarity score where higher values indicate better matches. 
    This scoring approach allows me to quantify how typical or atypical a feature value is for each 
    environment type, which forms the foundation of my classification decision-making process.
    """
    def _calculate_feature_score(self, feature_value, threshold_dict):
        mean_val = threshold_dict['mean']
        std_val = threshold_dict['std']
        
        if std_val > 0:
            distance = abs(feature_value - mean_val) / std_val
            score = max(0, 1 - distance / 2)
        else:
            score = 1.0 if feature_value == mean_val else 0.0
        
        return score

"""
This main function orchestrates the entire edge orientation analysis pipeline from start to finish. 
I initialize the system, process the training dataset to extract features, and then create a 
classifier based on those features. The function handles the complete workflow including data 
loading, feature extraction, statistical analysis, and classifier preparation. It provides progress 
updates and saves results to files for future use. This comprehensive approach ensures that all 
components of the system work together seamlessly and creates a ready-to-use classification system 
for indoor environment recognition based on edge orientation patterns.
"""
def main():
    print("Edge Orientation Analysis System")
    print("="*50)
    
    system = EdgeOrientationSystem()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    
    print("Step 1: Extracting features from training dataset...")
    df, csv_filename = system.process_training_dataset(base_path)
    
    print(f"\nStep 2: Features saved to {csv_filename}")
    print("You can now use this CSV file for analysis!")
    
    print("\nStep 3: Creating classifier from extracted features...")
    system.load_features_and_create_classifier(csv_filename)
    
    print("\nSystem ready for classification!")
    print("Use system.classify_new_image('path/to/image.jpg') to classify new images.")
    
    return system, csv_filename

if __name__ == "__main__":
    system, csv_file = main()