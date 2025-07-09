

# # Regional Color (Variance) Distribution Analysis ----> Implemenation




# version 3

# import cv2
# import numpy as np
# import os
# import pandas as pd
# from datetime import datetime

# class RegionalColorDistributionClassifier:
#     def __init__(self):
#         """
#         Regional Color Distribution Classifier based on comprehensive data analysis.
        
#         Key findings from 231 training images:
#         - OPEN AREAS: top_blue_percentage = 0.4882 (vs ~0.01-0.04 indoor) - MASSIVE discriminator
#         - ROOMS: artificial_color_score = 0.9101 (highest), global_color_entropy = 7.4639 (highest)
#         - HALLWAYS: global_color_entropy = 7.2475 (lowest), artificial_color_score = 0.8540 (lowest)
#         - STAIRCASES: Middle values across features
        
#         Uses pure statistical thresholds - NO machine learning.
#         """
        
#         # Classification rules based on data analysis - FIXED FOR STAIRCASE BIAS
#         self.classification_rules = {
#             # Primary rule: Open area detection (confirmed working well)
#             'open_area_detection': {
#                 'blue_threshold': 0.15,              # Clear separation
#                 'confidence_boost': 0.20
#             },
            
#             # POSITIVE identification rules for indoor environments (NO DEFAULTS!)
#             'positive_identification': {
#                 # Hallway: Lowest artificial score + lowest entropy
#                 'hallway': {
#                     'artificial_max': 0.90,          # Q75 of hallways ≈ 0.89
#                     'entropy_max': 7.45,             # Q75 of hallways ≈ 7.43
#                     'both_required': True            # Must meet BOTH criteria
#                 },
                
#                 # Room: Highest artificial score + highest entropy  
#                 'room': {
#                     'artificial_min': 0.88,          # Q25 of rooms ≈ 0.89
#                     'entropy_min': 7.20,             # Q25 of rooms ≈ 7.18
#                     'both_required': True            # Must meet BOTH criteria
#                 },
                
#                 # Staircase: Middle ranges - POSITIVE identification
#                 'staircase': {
#                     'artificial_range': (0.85, 0.95), # Covers staircase range
#                     'entropy_range': (7.0, 7.5),      # Covers staircase range
#                     'flexible_matching': True          # More lenient for staircases
#                 }
#             },
            
#             # Confidence scoring for overlapping cases
#             'confidence_scoring': {
#                 'high_confidence_threshold': 0.75,
#                 'medium_confidence_threshold': 0.60,
#                 'low_confidence_threshold': 0.50
#             }
#         }
        
#         # Confidence scoring weights
#         self.confidence_weights = {
#             'open_area': {
#                 'blue_percentage': 0.60,      # Strongest discriminator
#                 'natural_score': 0.25,
#                 'artificial_penalty': 0.15
#             },
#             'room': {
#                 'artificial_score': 0.45,
#                 'entropy_score': 0.35,
#                 'variance_support': 0.20
#             },
#             'hallway': {
#                 'entropy_score': 0.50,
#                 'artificial_score': 0.30,
#                 'uniformity_support': 0.20
#             },
#             'staircase': {
#                 'middle_position': 0.40,
#                 'variance_pattern': 0.35,
#                 'elimination_score': 0.25
#             }
#         }
    
#     def extract_color_features(self, image_path):
#         """Extract the essential color features for classification."""
#         try:
#             # Load and preprocess image
#             img = cv2.imread(image_path)
#             if img is None:
#                 raise ValueError(f"Could not load image: {image_path}")
            
#             # Convert color spaces
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
#             # Resize for consistency
#             height, width = img_rgb.shape[:2]
#             if width > 800:
#                 scale = 800 / width
#                 new_width = 800
#                 new_height = int(height * scale)
#                 img_rgb = cv2.resize(img_rgb, (new_width, new_height))
#                 img_hsv = cv2.resize(img_hsv, (new_width, new_height))
            
#             height, width = img_rgb.shape[:2]
            
#             # Extract key features only (based on analysis)
#             features = {}
            
#             # Region definitions
#             top_region = img_rgb[:height//3, :]
#             middle_region = img_rgb[height//3:2*height//3, :]
#             bottom_region = img_rgb[2*height//3:, :]
#             top_hsv = img_hsv[:height//3, :]
            
#             # Primary discriminator: top blue percentage
#             features['top_blue_percentage'] = self._calculate_blue_percentage(top_hsv)
            
#             # Secondary features for indoor classification
#             features['global_color_entropy'] = self._calculate_color_entropy(img_rgb)
#             features['artificial_color_score'] = self._calculate_artificial_color_score(img_rgb)
#             features['natural_color_score'] = self._calculate_natural_color_score(img_hsv)
            
#             # Supporting features  
#             features['middle_color_variance'] = self._calculate_region_variance(middle_region)
#             features['bottom_color_variance'] = self._calculate_region_variance(bottom_region)
#             features['color_temperature_score'] = self._analyze_color_temperature(img_rgb)
#             features['horizontal_gradient_smoothness'] = self._calculate_horizontal_gradient(img_rgb)
#             features['top_color_variance'] = self._calculate_region_variance(top_region)
#             features['global_color_uniformity'] = self._calculate_color_uniformity(img_rgb)
#             features['vertical_gradient_smoothness'] = self._calculate_vertical_gradient(img_rgb)
            
#             return features
            
#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")
#             return self._get_default_features()
    
#     def _calculate_blue_percentage(self, hsv_region):
#         """Calculate percentage of blue pixels - KEY DISCRIMINATOR."""
#         blue_mask = cv2.inRange(hsv_region, 
#                                np.array([100, 30, 80]), 
#                                np.array([130, 255, 255]))
#         blue_pixels = np.sum(blue_mask > 0)
#         total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
#         return blue_pixels / total_pixels
    
#     def _calculate_color_entropy(self, img_rgb):
#         """Calculate global color entropy."""
#         # Convert to grayscale for histogram
#         gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
#         # Calculate histogram
#         hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#         hist = hist.flatten()
#         hist = hist / np.sum(hist)
        
#         # Calculate entropy
#         hist = hist[hist > 0]
#         if len(hist) == 0:
#             return 0.0
#         entropy = -np.sum(hist * np.log2(hist))
#         return entropy
    
#     def _calculate_artificial_color_score(self, img_rgb):
#         """Calculate artificial color uniformity score."""
#         gray_level = np.mean(img_rgb, axis=2)
#         color_deviation = np.std(img_rgb - gray_level[:, :, np.newaxis], axis=2)
        
#         # Low color deviation indicates artificial/uniform lighting
#         artificial_score = 1.0 - (np.mean(color_deviation) / 128.0)
#         return max(0, min(1, artificial_score))
    
#     def _calculate_natural_color_score(self, img_hsv):
#         """Calculate natural color palette score."""
#         blue_score = self._calculate_blue_percentage(img_hsv)
        
#         # Green detection
#         green_mask = cv2.inRange(img_hsv,
#                                np.array([40, 30, 50]),
#                                np.array([80, 255, 255]))
#         green_score = np.sum(green_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
        
#         # Brown/earth tone detection
#         brown_mask = cv2.inRange(img_hsv,
#                                np.array([10, 30, 30]),
#                                np.array([25, 200, 200]))
#         brown_score = np.sum(brown_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
        
#         return blue_score + green_score + brown_score
    
#     def _calculate_region_variance(self, region):
#         """Calculate color variance for a region."""
#         return np.mean([
#             np.var(region[:, :, 0]),
#             np.var(region[:, :, 1]),
#             np.var(region[:, :, 2])
#         ])
    
#     def _calculate_color_uniformity(self, img_rgb):
#         """Calculate overall color uniformity."""
#         uniformity_scores = []
#         for channel in range(3):
#             mean_val = np.mean(img_rgb[:, :, channel])
#             std_val = np.std(img_rgb[:, :, channel])
#             cv = std_val / (mean_val + 1e-6)
#             uniformity_scores.append(1.0 / (1.0 + cv))
        
#         return np.mean(uniformity_scores)
    
#     def _analyze_color_temperature(self, img_rgb):
#         """Analyze color temperature (warm vs cool)."""
#         # Simple color temperature analysis
#         warm_colors = np.sum((img_rgb[:, :, 0] > img_rgb[:, :, 2]) & 
#                            (img_rgb[:, :, 1] > img_rgb[:, :, 2]))
#         cool_colors = np.sum((img_rgb[:, :, 2] > img_rgb[:, :, 0]) & 
#                            (img_rgb[:, :, 2] > img_rgb[:, :, 1]))
#         total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        
#         warm_ratio = warm_colors / total_pixels
#         cool_ratio = cool_colors / total_pixels
        
#         # Return color temperature score (0: cool, 0.5: neutral, 1: warm)
#         return (warm_ratio - cool_ratio + 1) / 2
    
#     def _calculate_vertical_gradient(self, img_rgb):
#         """Calculate vertical gradient smoothness."""
#         row_means = np.mean(img_rgb, axis=(1, 2))
#         gradients = np.diff(row_means)
#         gradient_variance = np.var(gradients)
#         smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
#         return smoothness
    
#     def _calculate_horizontal_gradient(self, img_rgb):
#         """Calculate horizontal gradient smoothness."""
#         # Calculate mean color for each column
#         col_means = np.mean(img_rgb, axis=(0, 2))
        
#         # Calculate gradient smoothness
#         gradients = np.diff(col_means)
#         gradient_variance = np.var(gradients)
        
#         # Convert to smoothness score
#         smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
#         return smoothness
    
#     def _get_default_features(self):
#         """Return default features for failed processing."""
#         return {
#             'top_blue_percentage': 0.0,
#             'global_color_entropy': 7.0,
#             'artificial_color_score': 0.8,
#             'natural_color_score': 0.3,
#             'top_color_variance': 2000.0,
#             'global_color_uniformity': 0.8,
#             'vertical_gradient_smoothness': 0.9
#         }
    
#     def classify_image(self, image_path, debug=False):
#         """
#         Classify image using data-driven decision tree.
#         """
#         # Extract features
#         features = self.extract_color_features(image_path)
        
#         if 'error' in features:
#             return {'error': 'Failed to process image', 'predicted_class': 'unknown', 'confidence': 0.0}
        
#         # DECISION TREE CLASSIFICATION
#         decision_path = []
        
#         # STEP 1: OPEN AREA DETECTION (Primary filter - extremely reliable)
#         blue_percentage = features['top_blue_percentage']
#         natural_score = features['natural_color_score']
#         artificial_score = features['artificial_color_score']
        
#         # Based on analysis: open areas have blue_percentage = 0.4882 vs indoor ~0.01-0.04
#         if blue_percentage > self.classification_rules['open_area_detection']['blue_threshold']:
#             decision_path.append(f"Open area detected: blue_percentage={blue_percentage:.3f} > 0.15")
            
#             # Calculate confidence based on how clearly it's outdoor
#             confidence = 0.80  # Base confidence
#             if blue_percentage > 0.25:
#                 confidence += 0.15  # Strong blue presence
#             if natural_score > 0.45:  # Fixed threshold instead of rule reference
#                 confidence += 0.05  # Natural color support
            
#             return {
#                 'predicted_class': 'open_area',
#                 'confidence': min(1.0, confidence),
#                 'features': features,
#                 'decision_path': decision_path if debug else None,
#                 'image_path': image_path
#             }
        
#         # STEP 2: POSITIVE IDENTIFICATION FOR INDOOR ENVIRONMENTS (NO DEFAULTS!)
#         entropy = features['global_color_entropy']
#         artificial_score = features['artificial_color_score']
        
#         decision_path.append(f"Indoor environment detected: blue_percentage={blue_percentage:.3f} <= 0.15")
        
#         # Calculate scores for each indoor environment type
#         scores = {
#             'hallway': 0.0,
#             'staircase': 0.0,
#             'room': 0.0
#         }
        
#         # HALLWAY SCORING (enhanced with secondary features)
#         hallway_rules = self.classification_rules['positive_identification']['hallway']
#         hallway_score = 0.0
        
#         # Primary indicators
#         if (artificial_score < hallway_rules['artificial_max'] and 
#             entropy < hallway_rules['entropy_max']):
#             hallway_score = 0.60  # Base score for meeting primary criteria
            
#             # Secondary discriminative features for hallway identification
#             color_temp = features.get('color_temperature_score', 0.5)
#             top_variance = features.get('top_color_variance', 2000)
#             middle_variance = features.get('middle_color_variance', 2000)
#             h_gradient = features.get('horizontal_gradient_smoothness', 0.9)
#             v_gradient = features.get('vertical_gradient_smoothness', 0.9)
            
#             # Hallway-specific bonuses
#             # 1. Consistent lighting (moderate color temperature)
#             if 0.4 <= color_temp <= 0.8:  # Artificial lighting range
#                 hallway_score += 0.08
                
#             # 2. Low top variance (uniform ceiling/lighting)
#             if top_variance < 2200:  # Below hallway mean + std
#                 hallway_score += 0.06
                
#             # 3. High horizontal consistency (repetitive patterns)
#             if h_gradient > 0.94:  # High horizontal uniformity
#                 hallway_score += 0.08
                
#             # 4. Moderate vertical consistency (not too uniform like rooms)
#             if 0.90 <= v_gradient <= 0.97:  # Hallway sweet spot
#                 hallway_score += 0.06
                
#             # 5. Consistent variance between top and middle (uniform structure)
#             variance_ratio = abs(top_variance - middle_variance) / (top_variance + middle_variance)
#             if variance_ratio < 0.15:  # Similar variance across regions
#                 hallway_score += 0.05
                
#             decision_path.append(f"Hallway enhanced scoring: base={0.60}, temp_bonus={0.08 if 0.4 <= color_temp <= 0.8 else 0}, variance_bonus={0.06 if top_variance < 2200 else 0}")
            
#         scores['hallway'] = min(hallway_score, 0.95)  # Cap at 0.95        
#         # ROOM SCORING (positive identification)
#         room_rules = self.classification_rules['positive_identification']['room']
#         if (artificial_score > room_rules['artificial_min'] and 
#             entropy > room_rules['entropy_min']):
#             # Strong room indicators
#             scores['room'] = 0.80
#             # Bonus for being clearly in room territory
#             scores['room'] += (artificial_score - room_rules['artificial_min']) * 2.0
#             scores['room'] += (entropy - room_rules['entropy_min']) * 0.5
#             # Supporting features for room classification
#             top_variance = features.get('top_color_variance', 2000)
#             uniformity = features.get('global_color_uniformity', 0.8)
            
#             # Room bonuses
#             # 1. Higher top variance (varied ceiling features, furniture)
#             if top_variance > 2600:  # Above room typical
#                 scores['room'] += 0.05
                
#             # 2. High global uniformity (controlled indoor environment)
#             if uniformity > 0.85:  # Very uniform
#                 scores['room'] += 0.04
                
#             decision_path.append(f"Room indicators: artificial={artificial_score:.3f} > {room_rules['artificial_min']} AND entropy={entropy:.3f} > {room_rules['entropy_min']}")
        
#         # STAIRCASE SCORING (enhanced with depth/geometric features)
#         staircase_rules = self.classification_rules['positive_identification']['staircase']
#         art_min, art_max = staircase_rules['artificial_range']
#         ent_min, ent_max = staircase_rules['entropy_range']
#         staircase_score = 0.0
        
#         # Check if in staircase ranges
#         in_art_range = art_min <= artificial_score <= art_max
#         in_ent_range = ent_min <= entropy <= ent_max
        
#         if in_art_range and in_ent_range:
#             staircase_score = 0.60  # Base score
            
#             # Staircase-specific features
#             middle_variance = features.get('middle_color_variance', 3000)
#             bottom_variance = features.get('bottom_color_variance', 2500)
#             v_gradient = features.get('vertical_gradient_smoothness', 0.9)
            
#             # Staircase bonuses
#             # 1. Higher middle variance (step patterns, depth changes)
#             if middle_variance > 3200:  # Above staircase mean
#                 staircase_score += 0.08
                
#             # 2. Variable bottom variance (step shadows, depth)
#             if 2500 <= bottom_variance <= 3200:  # Staircase typical range
#                 staircase_score += 0.06
                
#             # 3. Moderate vertical gradient (steps create some variation)
#             if 0.85 <= v_gradient <= 0.93:  # More variation than hallways
#                 staircase_score += 0.07
                
#             # 4. Distance from center of ranges (closer to center = more typical)
#             art_center = (art_min + art_max) / 2
#             ent_center = (ent_min + ent_max) / 2
#             art_distance = abs(artificial_score - art_center) / (art_max - art_min)
#             ent_distance = abs(entropy - ent_center) / (ent_max - ent_min)
#             center_bonus = (2.0 - art_distance - ent_distance) * 0.05
#             staircase_score += max(0, center_bonus)
            
#             decision_path.append(f"Staircase enhanced scoring: in_range=True, variance_bonus={0.08 if middle_variance > 3200 else 0}")
            
#         # Give partial credit if close to ranges
#         elif (abs(artificial_score - (art_min + art_max)/2) < 0.05 or 
#               abs(entropy - (ent_min + ent_max)/2) < 0.3):
#             staircase_score = 0.45  # Partial score for being close
#             decision_path.append(f"Staircase partial match: close to ranges")
            
#         scores['staircase'] = min(staircase_score, 0.95)
        
#         # Additional scoring for edge cases
#         # If no clear winner, use relative positioning
#         if max(scores.values()) < 0.60:
#             decision_path.append("No clear classification, using relative scoring...")
            
#             # Relative to means from training data
#             hallway_art_mean = 0.854
#             hallway_ent_mean = 7.248
#             room_art_mean = 0.910
#             room_ent_mean = 7.464
#             staircase_art_mean = 0.894
#             staircase_ent_mean = 7.269
            
#             # Distance-based scoring
#             hallway_distance = abs(artificial_score - hallway_art_mean) + abs(entropy - hallway_ent_mean)
#             room_distance = abs(artificial_score - room_art_mean) + abs(entropy - room_ent_mean)
#             staircase_distance = abs(artificial_score - staircase_art_mean) + abs(entropy - staircase_ent_mean)
            
#             # Invert distances to get scores (closer = higher score)
#             max_distance = max(hallway_distance, room_distance, staircase_distance)
#             scores['hallway'] = max(scores['hallway'], (max_distance - hallway_distance) / max_distance * 0.6)
#             scores['room'] = max(scores['room'], (max_distance - room_distance) / max_distance * 0.6)
#             scores['staircase'] = max(scores['staircase'], (max_distance - staircase_distance) / max_distance * 0.6)
        
#         # Determine final classification
#         predicted_class = max(scores, key=scores.get)
#         confidence = scores[predicted_class]
        
#         # Ensure minimum confidence
#         if confidence < 0.50:
#             confidence = 0.50
#             decision_path.append("Low confidence classification - uncertain case")
        
#         decision_path.append(f"Final scores: hallway={scores['hallway']:.3f}, room={scores['room']:.3f}, staircase={scores['staircase']:.3f}")
#         decision_path.append(f"Predicted: {predicted_class} with confidence {confidence:.3f}")
        
#         return {
#             'predicted_class': predicted_class,
#             'confidence': max(0.5, min(1.0, confidence)),
#             'features': features,
#             'decision_path': decision_path if debug else None,
#             'image_path': image_path
#         }
    
#     def classify_images(self, image_paths, debug=False):
#         """Classify multiple images."""
#         results = []
#         for i, image_path in enumerate(image_paths):
#             if (i + 1) % 5 == 0:
#                 print(f"Processed {i + 1}/{len(image_paths)} images...")
            
#             result = self.classify_image(image_path, debug=debug)
#             results.append(result)
        
#         return results

# def test_on_separate_folders():
#     """
#     Test the color classifier on separate test folders.
#     """
#     classifier = RegionalColorDistributionClassifier()
    
#     # Define test folder structure
#     base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
#     test_folders = {
#         'hallway': 'hallway_test_photos',
#         'staircase': 'staircase_test_photos', 
#         'room': 'room_test_photos',
#         'open_area': 'openarea_test_photos'
#     }
    
#     print("Testing Regional Color Distribution Classifier")
#     print("="*55)
    
#     # Store results for each environment type
#     environment_results = {
#         'hallway': [],
#         'staircase': [], 
#         'room': [],
#         'open_area': []
#     }
    
#     # Process each test folder
#     for env_type, folder_name in test_folders.items():
#         folder_path = os.path.join(base_path, folder_name)
        
#         if not os.path.exists(folder_path):
#             print(f"Warning: Folder {folder_path} not found")
#             continue
        
#         # Get all image files
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
#         image_files = [f for f in os.listdir(folder_path) 
#                       if f.lower().endswith(image_extensions)]
        
#         if not image_files:
#             print(f"Warning: No images found in {folder_name}")
#             continue
        
#         print(f"\nTesting {len(image_files)} images from {env_type} folder:")
        
#         correct_predictions = 0
#         folder_results = []
        
#         for img_file in image_files:
#             img_path = os.path.join(folder_path, img_file)
            
#             # Classify the image
#             result = classifier.classify_image(img_path, debug=False)
#             result['true_class'] = env_type
#             result['image_name'] = img_file
#             result['is_correct'] = result['predicted_class'] == env_type
            
#             folder_results.append(result)
            
#             if result['is_correct']:
#                 correct_predictions += 1
            
#             # Print result for each image
#             status = "CORRECT" if result['is_correct'] else "WRONG"
#             print(f"  {img_file}: predicted {result['predicted_class']} ({status})")
        
#         # Store results for this environment
#         environment_results[env_type] = folder_results
        
#         # Print folder accuracy
#         accuracy = correct_predictions / len(image_files) if image_files else 0
#         print(f"{env_type} accuracy: {accuracy:.3f} ({correct_predictions}/{len(image_files)})")
    
#     # Overall summary
#     total_images = 0
#     total_correct = 0
    
#     print(f"\nOverall Results:")
#     for env_type, results in environment_results.items():
#         if results:
#             env_correct = sum(1 for r in results if r['is_correct'])
#             env_total = len(results)
#             env_accuracy = env_correct / env_total if env_total > 0 else 0
            
#             total_images += env_total
#             total_correct += env_correct
            
#             print(f"{env_type}: {env_accuracy:.3f} ({env_correct}/{env_total})")
    
#     overall_accuracy = total_correct / total_images if total_images > 0 else 0
#     print(f"Overall: {overall_accuracy:.3f} ({total_correct}/{total_images})")
    
#     return environment_results

# def main():
#     """
#     Main function for Regional Color Distribution Classification
#     """
#     print("Regional Color Distribution Classifier")
#     print("Data-Driven Statistical Approach")
#     print("="*45)
    
#     classifier = RegionalColorDistributionClassifier()
    
#     print("\nClassification Logic Based on Data Analysis:")
#     print("1. Open Area Detection: top_blue_percentage > 0.15 (0.4882 vs ~0.01-0.04)")
#     print("2. Room Detection: artificial_score > 0.905 AND entropy > 7.35")
#     print("3. Hallway Detection: entropy < 7.30 AND artificial_score < 0.87")
#     print("4. Staircase: Default for remaining indoor environments")
    
#     # Run tests
#     results = test_on_separate_folders()
    
#     return classifier, results

# if __name__ == "__main__":
#     classifier, results = main()
    
# Overall Results:
# hallway: 0.255 (14/55)
# staircase: 0.365 (19/52)
# room: 0.695 (41/59)
# open_area: 0.692 (45/65)
# Overall: 0.515 (119/231)










# version 4 

# import cv2
# import numpy as np
# import os
# import pandas as pd
# from datetime import datetime

# class RegionalColorDistributionClassifier:
#     def __init__(self):
#         """
#         Regional Color Distribution Classifier with aggressive thresholds.
        
#         Based on training data analysis:
#         - Hallway: artificial=0.854±0.107, entropy=7.248±0.410 (LOWEST values)
#         - Staircase: artificial=0.894±0.063, entropy=7.269±0.515 (MIDDLE values)
#         - Room: artificial=0.910±0.043, entropy=7.464±0.345 (HIGHEST values)
#         - Open Area: top_blue_percentage=0.4882 (vs ~0.01-0.04 indoor)
        
#         Strategy: Use tighter, more aggressive thresholds to better separate similar categories.
#         """
        
#         # Aggressive classification rules
#         self.classification_rules = {
#             # Open area detection (confirmed working well)
#             'open_area': {
#                 'blue_threshold': 0.15,
#                 'confidence_base': 0.80
#             },
            
#             # Aggressive indoor classification thresholds
#             'hallway': {
#                 'artificial_max': 0.88,       # Much tighter than before (was 0.90)
#                 'entropy_max': 7.4,           # Tighter than before (was 7.45)
#                 'strong_artificial': 0.85,    # Strong bonus threshold
#                 'strong_entropy': 7.3,        # Strong bonus threshold
#                 'base_confidence': 0.70
#             },
            
#             'room': {
#                 'artificial_min': 0.88,       # Keep working threshold
#                 'entropy_min': 7.20,          # Keep working threshold
#                 'base_confidence': 0.75
#             },
            
#             'staircase': {
#                 'artificial_min': 0.875,      # Narrower range
#                 'artificial_max': 0.92,       # Narrower range
#                 'entropy_min': 7.1,           # Narrower range
#                 'entropy_max': 7.4,           # Narrower range
#                 'sweet_spot_artificial': (0.885, 0.905),  # Close to mean
#                 'sweet_spot_entropy': (7.2, 7.35),        # Close to mean
#                 'base_confidence': 0.65
#             }
#         }
    
#     def extract_color_features(self, image_path):
#         """Extract essential color features for classification."""
#         try:
#             # Load and preprocess image
#             img = cv2.imread(image_path)
#             if img is None:
#                 raise ValueError(f"Could not load image: {image_path}")
            
#             # Convert color spaces
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
#             # Resize for consistency
#             height, width = img_rgb.shape[:2]
#             if width > 800:
#                 scale = 800 / width
#                 new_width = 800
#                 new_height = int(height * scale)
#                 img_rgb = cv2.resize(img_rgb, (new_width, new_height))
#                 img_hsv = cv2.resize(img_hsv, (new_width, new_height))
            
#             height, width = img_rgb.shape[:2]
            
#             # Extract key features
#             features = {}
            
#             # Region definitions
#             top_region = img_rgb[:height//3, :]
#             middle_region = img_rgb[height//3:2*height//3, :]
#             bottom_region = img_rgb[2*height//3:, :]
#             top_hsv = img_hsv[:height//3, :]
            
#             # Primary discriminators
#             features['top_blue_percentage'] = self._calculate_blue_percentage(top_hsv)
#             features['global_color_entropy'] = self._calculate_color_entropy(img_rgb)
#             features['artificial_color_score'] = self._calculate_artificial_color_score(img_rgb)
            
#             # Secondary features for enhanced discrimination
#             features['natural_color_score'] = self._calculate_natural_color_score(img_hsv)
#             features['top_color_variance'] = self._calculate_region_variance(top_region)
#             features['middle_color_variance'] = self._calculate_region_variance(middle_region)
#             features['bottom_color_variance'] = self._calculate_region_variance(bottom_region)
#             features['global_color_uniformity'] = self._calculate_color_uniformity(img_rgb)
#             features['vertical_gradient_smoothness'] = self._calculate_vertical_gradient(img_rgb)
#             features['horizontal_gradient_smoothness'] = self._calculate_horizontal_gradient(img_rgb)
#             features['color_temperature_score'] = self._analyze_color_temperature(img_rgb)
            
#             return features
            
#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")
#             return self._get_default_features()
    
#     def classify_image(self, image_path, debug=False):
#         """
#         Classify image using aggressive threshold approach.
#         """
#         # Extract features
#         features = self.extract_color_features(image_path)
        
#         if isinstance(features, dict) and 'error' in str(features):
#             return {'error': 'Failed to process image', 'predicted_class': 'unknown', 'confidence': 0.0}
        
#         decision_path = []
        
#         # STEP 1: OPEN AREA DETECTION (Primary filter)
#         blue_percentage = features['top_blue_percentage']
        
#         if blue_percentage > self.classification_rules['open_area']['blue_threshold']:
#             decision_path.append(f"Open area detected: blue_percentage={blue_percentage:.3f} > 0.15")
            
#             # Calculate confidence
#             confidence = self.classification_rules['open_area']['confidence_base']
#             if blue_percentage > 0.25:
#                 confidence += 0.15
#             if features['natural_color_score'] > 0.45:
#                 confidence += 0.05
            
#             return {
#                 'predicted_class': 'open_area',
#                 'confidence': min(1.0, confidence),
#                 'features': features,
#                 'decision_path': decision_path if debug else None,
#                 'image_path': image_path
#             }
        
#         # STEP 2: AGGRESSIVE INDOOR CLASSIFICATION
#         artificial_score = features['artificial_color_score']
#         entropy = features['global_color_entropy']
        
#         decision_path.append(f"Indoor environment: blue_percentage={blue_percentage:.3f} <= 0.15")
#         decision_path.append(f"Key features: artificial_score={artificial_score:.3f}, entropy={entropy:.3f}")
        
#         scores = {
#             'hallway': 0.0,
#             'staircase': 0.0,
#             'room': 0.0
#         }
        
#         # AGGRESSIVE HALLWAY IDENTIFICATION
#         hallway_rules = self.classification_rules['hallway']
#         if (artificial_score < hallway_rules['artificial_max'] and 
#             entropy < hallway_rules['entropy_max']):
            
#             scores['hallway'] = hallway_rules['base_confidence']
#             decision_path.append(f"Hallway criteria met: artificial < {hallway_rules['artificial_max']}, entropy < {hallway_rules['entropy_max']}")
            
#             # Strong bonuses for being clearly in hallway territory
#             if artificial_score < hallway_rules['strong_artificial']:
#                 scores['hallway'] += 0.15
#                 decision_path.append(f"Strong hallway bonus: artificial={artificial_score:.3f} < {hallway_rules['strong_artificial']}")
            
#             if entropy < hallway_rules['strong_entropy']:
#                 scores['hallway'] += 0.10
#                 decision_path.append(f"Strong hallway bonus: entropy={entropy:.3f} < {hallway_rules['strong_entropy']}")
            
#             # Additional hallway-specific features
#             color_temp = features.get('color_temperature_score', 0.5)
#             h_gradient = features.get('horizontal_gradient_smoothness', 0.9)
            
#             if 0.4 <= color_temp <= 0.8:  # Artificial lighting consistency
#                 scores['hallway'] += 0.05
#             if h_gradient > 0.93:  # High horizontal uniformity
#                 scores['hallway'] += 0.05
        
#         # ROOM IDENTIFICATION (Keep successful approach)
#         room_rules = self.classification_rules['room']
#         if (artificial_score > room_rules['artificial_min'] and 
#             entropy > room_rules['entropy_min']):
            
#             scores['room'] = room_rules['base_confidence']
#             decision_path.append(f"Room criteria met: artificial > {room_rules['artificial_min']}, entropy > {room_rules['entropy_min']}")
            
#             # Strong bonuses for room characteristics
#             scores['room'] += (artificial_score - room_rules['artificial_min']) * 3.0
#             scores['room'] += (entropy - room_rules['entropy_min']) * 0.8
            
#             # Room-specific bonuses
#             uniformity = features.get('global_color_uniformity', 0.8)
#             if uniformity > 0.85:
#                 scores['room'] += 0.05
        
#         # NARROWER STAIRCASE IDENTIFICATION
#         staircase_rules = self.classification_rules['staircase']
#         if (staircase_rules['artificial_min'] <= artificial_score <= staircase_rules['artificial_max'] and
#             staircase_rules['entropy_min'] <= entropy <= staircase_rules['entropy_max']):
            
#             scores['staircase'] = staircase_rules['base_confidence']
#             decision_path.append(f"Staircase range match: artificial in [{staircase_rules['artificial_min']}, {staircase_rules['artificial_max']}], entropy in [{staircase_rules['entropy_min']}, {staircase_rules['entropy_max']}]")
            
#             # Sweet spot bonuses
#             sweet_art_min, sweet_art_max = staircase_rules['sweet_spot_artificial']
#             sweet_ent_min, sweet_ent_max = staircase_rules['sweet_spot_entropy']
            
#             if sweet_art_min <= artificial_score <= sweet_art_max:
#                 scores['staircase'] += 0.08
#                 decision_path.append(f"Staircase sweet spot: artificial in [{sweet_art_min}, {sweet_art_max}]")
            
#             if sweet_ent_min <= entropy <= sweet_ent_max:
#                 scores['staircase'] += 0.08
#                 decision_path.append(f"Staircase sweet spot: entropy in [{sweet_ent_min}, {sweet_ent_max}]")
            
#             # Staircase-specific features
#             middle_variance = features.get('middle_color_variance', 3000)
#             if middle_variance > 3200:  # Step pattern variance
#                 scores['staircase'] += 0.06
        
#         # FALLBACK: Distance-based scoring if no strong matches
#         if max(scores.values()) < 0.60:
#             decision_path.append("No strong matches, using distance-based fallback...")
            
#             # Training data means
#             hallway_mean = (0.854, 7.248)
#             staircase_mean = (0.894, 7.269)
#             room_mean = (0.910, 7.464)
            
#             # Calculate distances (weighted: artificial_score more important)
#             hallway_dist = abs(artificial_score - hallway_mean[0]) * 2.0 + abs(entropy - hallway_mean[1]) * 0.1
#             staircase_dist = abs(artificial_score - staircase_mean[0]) * 2.0 + abs(entropy - staircase_mean[1]) * 0.1
#             room_dist = abs(artificial_score - room_mean[0]) * 2.0 + abs(entropy - room_mean[1]) * 0.1
            
#             # Convert to scores
#             max_dist = max(hallway_dist, staircase_dist, room_dist)
#             if max_dist > 0:
#                 scores['hallway'] = max(scores['hallway'], (max_dist - hallway_dist) / max_dist * 0.55)
#                 scores['staircase'] = max(scores['staircase'], (max_dist - staircase_dist) / max_dist * 0.55)
#                 scores['room'] = max(scores['room'], (max_dist - room_dist) / max_dist * 0.55)
        
#         # Final classification
#         predicted_class = max(scores, key=scores.get)
#         confidence = scores[predicted_class]
        
#         # Ensure minimum confidence
#         confidence = max(confidence, 0.50)
        
#         decision_path.append(f"Final scores: hallway={scores['hallway']:.3f}, staircase={scores['staircase']:.3f}, room={scores['room']:.3f}")
#         decision_path.append(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
        
#         return {
#             'predicted_class': predicted_class,
#             'confidence': min(1.0, confidence),
#             'features': features,
#             'decision_path': decision_path if debug else None,
#             'image_path': image_path
#         }
    
#     # Feature extraction methods
#     def _calculate_blue_percentage(self, hsv_region):
#         """Calculate percentage of blue pixels."""
#         blue_mask = cv2.inRange(hsv_region, 
#                                np.array([100, 30, 80]), 
#                                np.array([130, 255, 255]))
#         blue_pixels = np.sum(blue_mask > 0)
#         total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
#         return blue_pixels / total_pixels
    
#     def _calculate_color_entropy(self, img_rgb):
#         """Calculate global color entropy."""
#         gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#         hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#         hist = hist.flatten()
#         hist = hist / np.sum(hist)
#         hist = hist[hist > 0]
#         if len(hist) == 0:
#             return 0.0
#         entropy = -np.sum(hist * np.log2(hist))
#         return entropy
    
#     def _calculate_artificial_color_score(self, img_rgb):
#         """Calculate artificial color uniformity score."""
#         gray_level = np.mean(img_rgb, axis=2)
#         color_deviation = np.std(img_rgb - gray_level[:, :, np.newaxis], axis=2)
#         artificial_score = 1.0 - (np.mean(color_deviation) / 128.0)
#         return max(0, min(1, artificial_score))
    
#     def _calculate_natural_color_score(self, img_hsv):
#         """Calculate natural color palette score."""
#         blue_score = self._calculate_blue_percentage(img_hsv)
        
#         green_mask = cv2.inRange(img_hsv,
#                                np.array([40, 30, 50]),
#                                np.array([80, 255, 255]))
#         green_score = np.sum(green_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
        
#         brown_mask = cv2.inRange(img_hsv,
#                                np.array([10, 30, 30]),
#                                np.array([25, 200, 200]))
#         brown_score = np.sum(brown_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
        
#         return blue_score + green_score + brown_score
    
#     def _calculate_region_variance(self, region):
#         """Calculate color variance for a region."""
#         return np.mean([
#             np.var(region[:, :, 0]),
#             np.var(region[:, :, 1]),
#             np.var(region[:, :, 2])
#         ])
    
#     def _calculate_color_uniformity(self, img_rgb):
#         """Calculate overall color uniformity."""
#         uniformity_scores = []
#         for channel in range(3):
#             mean_val = np.mean(img_rgb[:, :, channel])
#             std_val = np.std(img_rgb[:, :, channel])
#             cv = std_val / (mean_val + 1e-6)
#             uniformity_scores.append(1.0 / (1.0 + cv))
        
#         return np.mean(uniformity_scores)
    
#     def _calculate_vertical_gradient(self, img_rgb):
#         """Calculate vertical gradient smoothness."""
#         row_means = np.mean(img_rgb, axis=(1, 2))
#         gradients = np.diff(row_means)
#         gradient_variance = np.var(gradients)
#         smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
#         return smoothness
    
#     def _calculate_horizontal_gradient(self, img_rgb):
#         """Calculate horizontal gradient smoothness."""
#         col_means = np.mean(img_rgb, axis=(0, 2))
#         gradients = np.diff(col_means)
#         gradient_variance = np.var(gradients)
#         smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
#         return smoothness
    
#     def _analyze_color_temperature(self, img_rgb):
#         """Analyze color temperature."""
#         warm_colors = np.sum((img_rgb[:, :, 0] > img_rgb[:, :, 2]) & 
#                            (img_rgb[:, :, 1] > img_rgb[:, :, 2]))
#         cool_colors = np.sum((img_rgb[:, :, 2] > img_rgb[:, :, 0]) & 
#                            (img_rgb[:, :, 2] > img_rgb[:, :, 1]))
#         total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        
#         warm_ratio = warm_colors / total_pixels
#         cool_ratio = cool_colors / total_pixels
        
#         return (warm_ratio - cool_ratio + 1) / 2
    
#     def _get_default_features(self):
#         """Return default features for failed processing."""
#         return {
#             'top_blue_percentage': 0.0,
#             'global_color_entropy': 7.0,
#             'artificial_color_score': 0.8,
#             'natural_color_score': 0.3,
#             'top_color_variance': 2000.0,
#             'middle_color_variance': 2500.0,
#             'bottom_color_variance': 2000.0,
#             'global_color_uniformity': 0.8,
#             'vertical_gradient_smoothness': 0.9,
#             'horizontal_gradient_smoothness': 0.9,
#             'color_temperature_score': 0.5
#         }

# def test_on_separate_folders():
#     """
#     Test the aggressive color classifier on separate test folders.
#     """
#     classifier = RegionalColorDistributionClassifier()
    
#     # Define test folder structure
#     base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
#     test_folders = {
#         'hallway': 'hallway_test_photos',
#         'staircase': 'staircase_test_photos', 
#         'room': 'room_test_photos',
#         'open_area': 'openarea_test_photos'
#     }
    
#     print("Testing Aggressive Regional Color Distribution Classifier")
#     print("="*60)
    
#     # Store results for each environment type
#     environment_results = {
#         'hallway': [],
#         'staircase': [], 
#         'room': [],
#         'open_area': []
#     }
    
#     # Process each test folder
#     for env_type, folder_name in test_folders.items():
#         folder_path = os.path.join(base_path, folder_name)
        
#         if not os.path.exists(folder_path):
#             print(f"Warning: Folder {folder_path} not found")
#             continue
        
#         # Get all image files
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
#         image_files = [f for f in os.listdir(folder_path) 
#                       if f.lower().endswith(image_extensions)]
        
#         if not image_files:
#             print(f"Warning: No images found in {folder_name}")
#             continue
        
#         print(f"\nTesting {len(image_files)} images from {env_type} folder:")
        
#         correct_predictions = 0
#         folder_results = []
        
#         for img_file in image_files:
#             img_path = os.path.join(folder_path, img_file)
            
#             # Classify the image
#             result = classifier.classify_image(img_path, debug=False)
#             result['true_class'] = env_type
#             result['image_name'] = img_file
#             result['is_correct'] = result['predicted_class'] == env_type
            
#             folder_results.append(result)
            
#             if result['is_correct']:
#                 correct_predictions += 1
            
#             # Print result for each image
#             status = "CORRECT" if result['is_correct'] else "WRONG"
#             print(f"  {img_file}: predicted {result['predicted_class']} ({status})")
        
#         # Store results for this environment
#         environment_results[env_type] = folder_results
        
#         # Print folder accuracy
#         accuracy = correct_predictions / len(image_files) if image_files else 0
#         print(f"{env_type} accuracy: {accuracy:.3f} ({correct_predictions}/{len(image_files)})")
    
#     # Overall summary
#     total_images = 0
#     total_correct = 0
    
#     print(f"\nOverall Results:")
#     for env_type, results in environment_results.items():
#         if results:
#             env_correct = sum(1 for r in results if r['is_correct'])
#             env_total = len(results)
#             env_accuracy = env_correct / env_total if env_total > 0 else 0
            
#             total_images += env_total
#             total_correct += env_correct
            
#             print(f"{env_type}: {env_accuracy:.3f} ({env_correct}/{env_total})")
    
#     overall_accuracy = total_correct / total_images if total_images > 0 else 0
#     print(f"Overall: {overall_accuracy:.3f} ({total_correct}/{total_images})")
    
#     return environment_results

# def main():
#     """
#     Main function for Aggressive Regional Color Distribution Classification
#     """
#     print("Aggressive Regional Color Distribution Classifier")
#     print("Tighter Thresholds for Better Separation")
#     print("="*50)
    
#     classifier = RegionalColorDistributionClassifier()
    
#     print("\nAggressive Classification Strategy:")
#     print("1. Open Area: top_blue_percentage > 0.15")
#     print("2. Hallway: artificial_score < 0.88 AND entropy < 7.4 (TIGHTER)")
#     print("3. Room: artificial_score > 0.88 AND entropy > 7.20")
#     print("4. Staircase: Narrow ranges with sweet spot bonuses")
    
#     # Run tests
#     results = test_on_separate_folders()
    
#     return classifier, results

# if __name__ == "__main__":
#     classifier, results = main()
    
    
# Overall Results:
# hallway: 0.491 (27/55)
# staircase: 0.058 (3/52)
# room: 0.729 (43/59)
# open_area: 0.692 (45/65)
# Overall: 0.511 (118/231)







# version 5:

import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

class RegionalColorDistributionClassifier:
    """
    This Regional Color Distribution Classifier implements an aggressive threshold-based approach 
    for classifying indoor environments using color pattern analysis. I designed this system based 
    on statistical insights from training data, using tighter and more precise thresholds to better 
    separate similar categories like hallways, staircases, and rooms. The classifier employs a 
    hierarchical decision-making process where open areas are detected first using blue percentage 
    thresholds, then indoor environments are distinguished using artificial color scores and entropy 
    values. For staircases, I've implemented unique pattern detection that looks for specific variance 
    patterns and geometric discontinuities that characterize these structured environments. This 
    aggressive approach prioritizes precision over recall to achieve better overall classification accuracy.
    """
    
    def __init__(self):
        self.classification_rules = {
            'open_area': {
                'blue_threshold': 0.15,
                'confidence_base': 0.80
            },
            
            'hallway': {
                'artificial_max': 0.88,
                'entropy_max': 7.4,
                'strong_artificial': 0.85,
                'strong_entropy': 7.3,
                'base_confidence': 0.70
            },
            
            'room': {
                'artificial_min': 0.88,
                'entropy_min': 7.20,
                'base_confidence': 0.75
            },
            
            'staircase': {
                'artificial_min': 0.875,
                'artificial_max': 0.92,
                'entropy_min': 7.1,
                'entropy_max': 7.4,
                'sweet_spot_artificial': (0.885, 0.905),
                'sweet_spot_entropy': (7.2, 7.35),
                'base_confidence': 0.65
            }
        }
    
    """
    In this function, I extract the essential color features needed for my aggressive classification 
    approach. I start by loading and preprocessing the image, converting it to multiple color spaces 
    to capture different aspects of color information. After resizing for consistency, I divide the 
    image into three vertical regions (top, middle, bottom) to analyze spatial color patterns. The 
    primary discriminators include top blue percentage for open area detection, global color entropy 
    for measuring complexity, and artificial color score for detecting uniform indoor lighting. I 
    also extract secondary features like regional color variance, gradient smoothness, and color 
    temperature that help fine-tune the classification decisions. This focused feature extraction 
    captures the key color characteristics that distinguish different environment types.
    """
    def extract_color_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            height, width = img_rgb.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_width, new_height))
                img_hsv = cv2.resize(img_hsv, (new_width, new_height))
            
            height, width = img_rgb.shape[:2]
            
            features = {}
            
            top_region = img_rgb[:height//3, :]
            middle_region = img_rgb[height//3:2*height//3, :]
            bottom_region = img_rgb[2*height//3:, :]
            top_hsv = img_hsv[:height//3, :]
            
            features['top_blue_percentage'] = self._calculate_blue_percentage(top_hsv)
            features['global_color_entropy'] = self._calculate_color_entropy(img_rgb)
            features['artificial_color_score'] = self._calculate_artificial_color_score(img_rgb)
            
            features['natural_color_score'] = self._calculate_natural_color_score(img_hsv)
            features['top_color_variance'] = self._calculate_region_variance(top_region)
            features['middle_color_variance'] = self._calculate_region_variance(middle_region)
            features['bottom_color_variance'] = self._calculate_region_variance(bottom_region)
            features['global_color_uniformity'] = self._calculate_color_uniformity(img_rgb)
            features['vertical_gradient_smoothness'] = self._calculate_vertical_gradient(img_rgb)
            features['horizontal_gradient_smoothness'] = self._calculate_horizontal_gradient(img_rgb)
            features['color_temperature_score'] = self._analyze_color_temperature(img_rgb)
            
            features['top_middle_color_correlation'] = self._calculate_region_correlation(top_region, middle_region)
            features['middle_bottom_color_correlation'] = self._calculate_region_correlation(middle_region, bottom_region)
            features['top_bottom_color_correlation'] = self._calculate_region_correlation(top_region, bottom_region)
            features['top_to_middle_transition'] = self._calculate_color_transition(top_region, middle_region)
            features['middle_to_bottom_transition'] = self._calculate_color_transition(middle_region, bottom_region)
            features['overall_color_transition'] = self._calculate_color_transition(top_region, bottom_region)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return self._get_default_features()
    
    """
    This function implements my aggressive classification strategy using a hierarchical decision-making 
    process. I start by detecting open areas using blue percentage thresholds, as these environments 
    are most easily distinguishable from indoor spaces. For indoor environments, I use tighter thresholds 
    based on artificial color scores and entropy values to create better separation between similar 
    categories. My hallway detection uses strict upper bounds for both artificial score and entropy, 
    while room detection requires higher values of both metrics. For staircases, I've developed a 
    unique pattern detection system that looks for characteristic variance patterns, geometric 
    discontinuities, and abrupt color transitions that distinguish these structured environments from 
    simple indoor spaces. This aggressive approach prioritizes precision and includes detailed scoring 
    mechanisms to provide confidence estimates for each classification decision.
    """
    def classify_image(self, image_path, debug=False):
        features = self.extract_color_features(image_path)
        
        if isinstance(features, dict) and 'error' in str(features):
            return {'error': 'Failed to process image', 'predicted_class': 'unknown', 'confidence': 0.0}
        
        decision_path = []
        
        blue_percentage = features['top_blue_percentage']
        
        if blue_percentage > self.classification_rules['open_area']['blue_threshold']:
            decision_path.append(f"Open area detected: blue_percentage={blue_percentage:.3f} > 0.15")
            
            confidence = self.classification_rules['open_area']['confidence_base']
            if blue_percentage > 0.25:
                confidence += 0.15
            if features['natural_color_score'] > 0.45:
                confidence += 0.05
            
            return {
                'predicted_class': 'open_area',
                'confidence': min(1.0, confidence),
                'features': features,
                'decision_path': decision_path if debug else None,
                'image_path': image_path
            }
        
        artificial_score = features['artificial_color_score']
        entropy = features['global_color_entropy']
        
        decision_path.append(f"Indoor environment: blue_percentage={blue_percentage:.3f} <= 0.15")
        decision_path.append(f"Key features: artificial_score={artificial_score:.3f}, entropy={entropy:.3f}")
        
        scores = {
            'hallway': 0.0,
            'staircase': 0.0,
            'room': 0.0
        }
        
        hallway_rules = self.classification_rules['hallway']
        if (artificial_score < hallway_rules['artificial_max'] and 
            entropy < hallway_rules['entropy_max']):
            
            scores['hallway'] = hallway_rules['base_confidence']
            decision_path.append(f"Hallway criteria met: artificial < {hallway_rules['artificial_max']}, entropy < {hallway_rules['entropy_max']}")
            
            if artificial_score < hallway_rules['strong_artificial']:
                scores['hallway'] += 0.15
                decision_path.append(f"Strong hallway bonus: artificial={artificial_score:.3f} < {hallway_rules['strong_artificial']}")
            
            if entropy < hallway_rules['strong_entropy']:
                scores['hallway'] += 0.10
                decision_path.append(f"Strong hallway bonus: entropy={entropy:.3f} < {hallway_rules['strong_entropy']}")
            
            color_temp = features.get('color_temperature_score', 0.5)
            h_gradient = features.get('horizontal_gradient_smoothness', 0.9)
            
            if 0.4 <= color_temp <= 0.8:
                scores['hallway'] += 0.05
            if h_gradient > 0.93:
                scores['hallway'] += 0.05
        
        room_rules = self.classification_rules['room']
        if (artificial_score > room_rules['artificial_min'] and 
            entropy > room_rules['entropy_min']):
            
            scores['room'] = room_rules['base_confidence']
            decision_path.append(f"Room criteria met: artificial > {room_rules['artificial_min']}, entropy > {room_rules['entropy_min']}")
            
            scores['room'] += (artificial_score - room_rules['artificial_min']) * 3.0
            scores['room'] += (entropy - room_rules['entropy_min']) * 0.8
            
            uniformity = features.get('global_color_uniformity', 0.8)
            if uniformity > 0.85:
                scores['room'] += 0.05
        
        staircase_rules = self.classification_rules['staircase']
        
        middle_variance = features.get('middle_color_variance', 3000)
        top_variance = features.get('top_color_variance', 2000)
        bottom_variance = features.get('bottom_color_variance', 2000)
        
        has_middle_peak = (middle_variance > top_variance and middle_variance > bottom_variance)
        
        middle_top_ratio = middle_variance / (top_variance + 1e-6)
        middle_bottom_ratio = middle_variance / (bottom_variance + 1e-6)
        
        top_middle_corr = features.get('top_middle_color_correlation', 0.8)
        middle_bottom_corr = features.get('middle_bottom_color_correlation', 0.8)
        
        top_to_middle_transition = features.get('top_to_middle_transition', 50)
        middle_to_bottom_transition = features.get('middle_to_bottom_transition', 50)
        
        if (staircase_rules['artificial_min'] <= artificial_score <= staircase_rules['artificial_max'] and
            staircase_rules['entropy_min'] <= entropy <= staircase_rules['entropy_max']):
            scores['staircase'] = 0.50
        else:
            scores['staircase'] = 0.30
        
        staircase_bonuses = []
        
        if has_middle_peak:
            peak_bonus = min(0.25, (middle_variance - max(top_variance, bottom_variance)) / 1000 * 0.1)
            scores['staircase'] += peak_bonus
            staircase_bonuses.append(f"middle_peak_bonus={peak_bonus:.3f}")
        
        if middle_top_ratio > 1.3:
            ratio_bonus = min(0.15, (middle_top_ratio - 1.3) * 0.1)
            scores['staircase'] += ratio_bonus
            staircase_bonuses.append(f"middle_top_ratio_bonus={ratio_bonus:.3f}")
            
        if middle_bottom_ratio > 1.2:
            ratio_bonus = min(0.10, (middle_bottom_ratio - 1.2) * 0.1)
            scores['staircase'] += ratio_bonus
            staircase_bonuses.append(f"middle_bottom_ratio_bonus={ratio_bonus:.3f}")
        
        if top_middle_corr < 0.7:
            corr_bonus = (0.7 - top_middle_corr) * 0.2
            scores['staircase'] += corr_bonus
            staircase_bonuses.append(f"low_top_middle_corr_bonus={corr_bonus:.3f}")
            
        if middle_bottom_corr < 0.7:
            corr_bonus = (0.7 - middle_bottom_corr) * 0.2
            scores['staircase'] += corr_bonus
            staircase_bonuses.append(f"low_middle_bottom_corr_bonus={corr_bonus:.3f}")
        
        if top_to_middle_transition > 60:
            transition_bonus = min(0.08, (top_to_middle_transition - 60) / 50 * 0.08)
            scores['staircase'] += transition_bonus
            staircase_bonuses.append(f"abrupt_transition_bonus={transition_bonus:.3f}")
        
        sweet_art_min, sweet_art_max = staircase_rules['sweet_spot_artificial']
        sweet_ent_min, sweet_ent_max = staircase_rules['sweet_spot_entropy']
        
        if sweet_art_min <= artificial_score <= sweet_art_max:
            scores['staircase'] += 0.05
            staircase_bonuses.append("sweet_spot_artificial=0.05")
            
        if sweet_ent_min <= entropy <= sweet_ent_max:
            scores['staircase'] += 0.05
            staircase_bonuses.append("sweet_spot_entropy=0.05")
        
        if (middle_top_ratio < 1.1 and middle_bottom_ratio < 1.1 and 
            top_middle_corr > 0.85 and middle_bottom_corr > 0.85):
            scores['staircase'] *= 0.7
            staircase_bonuses.append("uniform_pattern_penalty=0.7x")
        
        if staircase_bonuses:
            decision_path.append(f"Staircase unique features: {', '.join(staircase_bonuses)}")
        
        decision_path.append(f"Staircase variance analysis: middle={middle_variance:.0f}, top={top_variance:.0f}, bottom={bottom_variance:.0f}, peak={has_middle_peak}")
        
        scores['staircase'] = min(scores['staircase'], 0.95)
        
        if max(scores.values()) < 0.60:
            decision_path.append("No strong matches, using distance-based fallback...")
            
            hallway_mean = (0.854, 7.248)
            staircase_mean = (0.894, 7.269)
            room_mean = (0.910, 7.464)
            
            hallway_dist = abs(artificial_score - hallway_mean[0]) * 2.0 + abs(entropy - hallway_mean[1]) * 0.1
            staircase_dist = abs(artificial_score - staircase_mean[0]) * 2.0 + abs(entropy - staircase_mean[1]) * 0.1
            room_dist = abs(artificial_score - room_mean[0]) * 2.0 + abs(entropy - room_mean[1]) * 0.1
            
            max_dist = max(hallway_dist, staircase_dist, room_dist)
            if max_dist > 0:
                scores['hallway'] = max(scores['hallway'], (max_dist - hallway_dist) / max_dist * 0.55)
                scores['staircase'] = max(scores['staircase'], (max_dist - staircase_dist) / max_dist * 0.55)
                scores['room'] = max(scores['room'], (max_dist - room_dist) / max_dist * 0.55)
        
        predicted_class = max(scores, key=scores.get)
        confidence = scores[predicted_class]
        
        confidence = max(confidence, 0.50)
        
        decision_path.append(f"Final scores: hallway={scores['hallway']:.3f}, staircase={scores['staircase']:.3f}, room={scores['room']:.3f}")
        decision_path.append(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
        
        return {
            'predicted_class': predicted_class,
            'confidence': min(1.0, confidence),
            'features': features,
            'decision_path': decision_path if debug else None,
            'image_path': image_path
        }
    
    """
    Here I calculate the percentage of blue pixels in a region using HSV color space thresholds 
    that I've optimized for detecting sky and other blue elements. I create a mask that identifies 
    pixels within the blue hue range (100-130 degrees) with sufficient saturation and brightness 
    to represent visually significant blue content. This blue percentage calculation is particularly 
    crucial for detecting open areas where sky is typically visible in the top region of images. 
    By counting pixels that meet the blue criteria and dividing by total pixels, I get a reliable 
    metric for distinguishing outdoor environments from indoor spaces. This serves as the primary 
    discriminator in my hierarchical classification approach, effectively separating open areas 
    from all indoor environment types before proceeding to more detailed indoor classification.
    """
    def _calculate_blue_percentage(self, hsv_region):
        blue_mask = cv2.inRange(hsv_region, 
                               np.array([100, 30, 80]), 
                               np.array([130, 255, 255]))
        blue_pixels = np.sum(blue_mask > 0)
        total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
        return blue_pixels / total_pixels
    
    """
    In this function, I calculate the global color entropy of the entire image as a measure of 
    color complexity and randomness. I convert the image to grayscale and create a histogram of 
    intensity values, then normalize it to form a probability distribution. Using the Shannon 
    entropy formula, I quantify how evenly distributed the color intensities are across the image. 
    Higher entropy values indicate complex color patterns with many different intensities, which 
    is characteristic of rooms with diverse furniture and decorations. Lower entropy suggests 
    simpler, more uniform color distributions found in structured environments like hallways. 
    This entropy measure serves as a key discriminator in my indoor classification rules, helping 
    distinguish between different types of indoor environments based on their color complexity.
    """
    def _calculate_color_entropy(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    """
    This function calculates how artificial the color palette appears by analyzing color uniformity 
    patterns typical of indoor lighting and painted surfaces. I compute the grayscale level for 
    each pixel, then measure how much the actual RGB colors deviate from this neutral baseline. 
    Low color deviation indicates uniform artificial lighting where colors appear flat and consistent, 
    which is common in indoor environments with fluorescent or LED lighting. High deviation suggests 
    more natural color variation. I convert this deviation measurement to an artificial score where 
    higher values indicate more uniform, artificial-looking color patterns. This artificial color 
    score is crucial for distinguishing between different indoor environment types, as hallways 
    typically show higher artificiality due to consistent lighting and minimal color variation.
    """
    def _calculate_artificial_color_score(self, img_rgb):
        gray_level = np.mean(img_rgb, axis=2)
        color_deviation = np.std(img_rgb - gray_level[:, :, np.newaxis], axis=2)
        artificial_score = 1.0 - (np.mean(color_deviation) / 128.0)
        return max(0, min(1, artificial_score))
    
    """
    Here I calculate a natural color score by detecting colors typically associated with natural 
    outdoor environments. I combine blue percentage (indicating sky), green percentage (indicating 
    vegetation), and brown/earth tone detection for natural ground surfaces or tree bark. For 
    each color type, I use specific HSV ranges that capture the characteristic hues while filtering 
    out very dark or pale variations that might not be visually significant. The total natural 
    color score represents the combined presence of these natural elements in the image. Higher 
    scores indicate environments with significant natural content, which helps distinguish open 
    areas from indoor spaces that typically contain more artificial colors from painted walls, 
    furniture, and artificial lighting. This natural scoring complements the artificial color analysis.
    """
    def _calculate_natural_color_score(self, img_hsv):
        blue_score = self._calculate_blue_percentage(img_hsv)
        
        green_mask = cv2.inRange(img_hsv,
                               np.array([40, 30, 50]),
                               np.array([80, 255, 255]))
        green_score = np.sum(green_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
        
        brown_mask = cv2.inRange(img_hsv,
                               np.array([10, 30, 30]),
                               np.array([25, 200, 200]))
        brown_score = np.sum(brown_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
        
        return blue_score + green_score + brown_score
    
    """
    This function calculates the color variance for a specific image region by computing the variance 
    of each RGB channel separately and then averaging these values. Color variance measures how much 
    the colors in a region deviate from their mean values, providing insight into the visual complexity 
    and texture of that area. High variance indicates regions with diverse colors and potentially 
    complex surfaces or objects, while low variance suggests uniform coloring typical of painted 
    walls or simple surfaces. This regional variance analysis is particularly important for my 
    staircase detection algorithm, as staircases often show characteristic patterns where the middle 
    region has higher variance due to the geometric structure of steps, railings, and depth variations. 
    By comparing variance across different regions, I can identify these distinctive spatial patterns.
    """
    def _calculate_region_variance(self, region):
        return np.mean([
            np.var(region[:, :, 0]),
            np.var(region[:, :, 1]),
            np.var(region[:, :, 2])
        ])
    
    """
    Here I calculate overall color uniformity across the entire image by measuring the coefficient 
    of variation for each RGB channel. I compute the mean and standard deviation for red, green, 
    and blue channels separately, then calculate the coefficient of variation (standard deviation 
    divided by mean) for each channel. This coefficient is converted to a uniformity score where 
    higher values indicate more consistent color distribution throughout the image. High uniformity 
    is characteristic of indoor environments with consistent artificial lighting and minimal color 
    variation, such as hallways with painted walls and uniform lighting fixtures. Lower uniformity 
    suggests more complex environments with varied lighting conditions, shadows, and diverse objects 
    like those found in rooms or outdoor scenes. This uniformity measure supports my classification rules.
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
    row across the image, then calculate the differences between consecutive rows to measure the 
    gradient magnitude. The variance of these gradients indicates whether color transitions are 
    smooth or abrupt vertically. Smooth transitions (low variance) are characteristic of natural 
    scenes where sky gradually transitions to ground, or indoor environments with consistent lighting. 
    Abrupt transitions suggest distinct color boundaries created by architectural elements, furniture, 
    or geometric structures. I convert the gradient variance to a smoothness score where higher 
    values indicate more gradual, natural-looking vertical color changes. This measure helps distinguish 
    between different environment types based on their vertical color organization.
    """
    
    
    def _calculate_vertical_gradient(self, img_rgb):
        row_means = np.mean(img_rgb, axis=(1, 2))
        gradients = np.diff(row_means)
        gradient_variance = np.var(gradients)
        smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
        return smoothness

    """
    In this function, I calculate the correlation between mean colors of two different image regions 
    to understand how similar their color characteristics are. I compute the mean RGB values for 
    each region, then calculate the Pearson correlation coefficient between these color vectors. 
    High correlation indicates that the regions have similar color characteristics, suggesting 
    consistent lighting and surfaces. Low correlation suggests distinct color differences between 
    regions, which can indicate geometric structures, different materials, or varying lighting 
    conditions. This correlation analysis is particularly useful for staircase detection, as staircases 
    often show lower correlations between regions due to the geometric discontinuities created by 
    steps, handrails, and changing perspectives. I handle potential NaN cases by returning a default 
    moderate correlation value to maintain robust classification performance.
    """
    def _calculate_region_correlation(self, region1, region2):
        try:
            mean1 = np.mean(region1, axis=(0, 1))
            mean2 = np.mean(region2, axis=(0, 1))
            
            correlation = np.corrcoef(mean1, mean2)[0, 1]
            
            if np.isnan(correlation):
                return 0.8
            
            return correlation
        except:
            return 0.8
    
    """
    Here I calculate the abruptness of color transitions between two regions by measuring the 
    Euclidean distance between their mean color values in RGB space. I compute the mean color 
    for each region, then calculate the magnitude of the color difference vector. Large transition 
    values indicate abrupt color changes between regions, which can be caused by distinct materials, 
    lighting changes, or geometric structures like steps or walls. Small values suggest smooth 
    color transitions typical of gradual lighting changes or similar surfaces. This transition 
    analysis is especially valuable for staircase detection, as the geometric structure of stairs 
    often creates abrupt color boundaries between different regions of the image. The transition 
    magnitude helps distinguish structured environments with clear geometric boundaries from more 
    uniform environments with gradual color changes.
    """
    def _calculate_color_transition(self, region1, region2):
        try:
            mean1 = np.mean(region1, axis=(0, 1))
            mean2 = np.mean(region2, axis=(0, 1))
            
            transition_magnitude = np.linalg.norm(mean1 - mean2)
            
            return transition_magnitude
        except:
            return 50.0
    
    """
    This function calculates the smoothness of horizontal color transitions across the image from 
    left to right. Similar to vertical gradient analysis, I compute the mean color for each vertical 
    column, then analyze the gradients between adjacent columns to assess transition smoothness. 
    The gradient variance is converted to a smoothness score where higher values indicate more 
    gradual horizontal color changes. This horizontal analysis helps identify environments with 
    consistent lighting and symmetrical features versus those with irregular color variations. 
    Hallways often show relatively smooth horizontal transitions due to uniform lighting and 
    symmetrical wall features, while complex environments like rooms with diverse furniture may 
    exhibit more irregular horizontal patterns. This gradient analysis provides additional spatial 
    context that complements the vertical gradient and other color distribution features in my 
    classification system.
    """
    def _calculate_horizontal_gradient(self, img_rgb):
        col_means = np.mean(img_rgb, axis=(0, 2))
        gradients = np.diff(col_means)
        gradient_variance = np.var(gradients)
        smoothness = 1.0 / (1.0 + gradient_variance / 100.0)
        return smoothness
    
    """
    In this function, I analyze the color temperature of the image to distinguish between warm and 
    cool lighting conditions. I identify warm colors by counting pixels where both red and green 
    values exceed blue values, and cool colors where blue dominates over both red and green channels. 
    By calculating the ratio of warm to cool pixels, I create a normalized color temperature score 
    that ranges from 0 (very cool) to 1 (very warm). This analysis helps distinguish between different 
    types of environments and lighting conditions - for example, outdoor daylight tends to be cooler, 
    while indoor incandescent lighting is warmer, and fluorescent lighting can be either cool or 
    neutral. The color temperature score provides additional context for environment classification, 
    particularly helping to distinguish between different types of indoor environments based on 
    their typical lighting characteristics and color palettes.
    """
    def _analyze_color_temperature(self, img_rgb):
        warm_colors = np.sum((img_rgb[:, :, 0] > img_rgb[:, :, 2]) & 
                           (img_rgb[:, :, 1] > img_rgb[:, :, 2]))
        cool_colors = np.sum((img_rgb[:, :, 2] > img_rgb[:, :, 0]) & 
                           (img_rgb[:, :, 2] > img_rgb[:, :, 1]))
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        
        warm_ratio = warm_colors / total_pixels
        cool_ratio = cool_colors / total_pixels
        
        return (warm_ratio - cool_ratio + 1) / 2
    
    """
    When image processing fails or encounters errors, I need to return a consistent set of default 
    features that maintain the integrity of my classification pipeline. This function provides 
    reasonable default values for all the color features that my system normally extracts, ensuring 
    that failed image processing doesn't cause classification errors. The default values are chosen 
    to represent neutral or average cases - moderate blue percentage, typical entropy levels for 
    indoor environments, moderate artificial color scores, and balanced variance patterns. These 
    defaults allow the classification system to continue processing even when individual images 
    fail, typically resulting in lower confidence predictions that reflect the uncertainty. This 
    robust error handling ensures that batch processing can continue smoothly and that the overall 
    system remains stable even with problematic input images.
    """
    def _get_default_features(self):
        return {
            'top_blue_percentage': 0.0,
            'global_color_entropy': 7.0,
            'artificial_color_score': 0.8,
            'natural_color_score': 0.3,
            'top_color_variance': 2000.0,
            'middle_color_variance': 2500.0,
            'bottom_color_variance': 2000.0,
            'global_color_uniformity': 0.8,
            'vertical_gradient_smoothness': 0.9,
            'horizontal_gradient_smoothness': 0.9,
            'color_temperature_score': 0.5,
            'top_middle_color_correlation': 0.8,
            'middle_bottom_color_correlation': 0.8,
            'top_bottom_color_correlation': 0.8,
            'top_to_middle_transition': 50.0,
            'middle_to_bottom_transition': 50.0,
            'overall_color_transition': 70.0
        }


def test_on_separate_folders():
    classifier = RegionalColorDistributionClassifier()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    test_folders = {
        'hallway': 'hallway_test_photos',
        'staircase': 'staircase_test_photos', 
        'room': 'room_test_photos',
        'open_area': 'openarea_test_photos'
    }
    
    print("Testing Aggressive Regional Color Distribution Classifier")
    print("="*60)
    
    environment_results = {
        'hallway': [],
        'staircase': [], 
        'room': [],
        'open_area': []
    }
    
    for env_type, folder_name in test_folders.items():
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found")
            continue
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"Warning: No images found in {folder_name}")
            continue
        
        print(f"\nTesting {len(image_files)} images from {env_type} folder:")
        
        correct_predictions = 0
        folder_results = []
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            result = classifier.classify_image(img_path, debug=False)
            result['true_class'] = env_type
            result['image_name'] = img_file
            result['is_correct'] = result['predicted_class'] == env_type
            
            folder_results.append(result)
            
            if result['is_correct']:
                correct_predictions += 1
            
            status = "CORRECT" if result['is_correct'] else "WRONG"
            print(f"  {img_file}: predicted {result['predicted_class']} ({status})")
        
        environment_results[env_type] = folder_results
        
        accuracy = correct_predictions / len(image_files) if image_files else 0
        print(f"{env_type} accuracy: {accuracy:.3f} ({correct_predictions}/{len(image_files)})")
    
    total_images = 0
    total_correct = 0
    
    print(f"\nOverall Results:")
    for env_type, results in environment_results.items():
        if results:
            env_correct = sum(1 for r in results if r['is_correct'])
            env_total = len(results)
            env_accuracy = env_correct / env_total if env_total > 0 else 0
            
            total_images += env_total
            total_correct += env_correct
            
            print(f"{env_type}: {env_accuracy:.3f} ({env_correct}/{env_total})")
    
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"Overall: {overall_accuracy:.3f} ({total_correct}/{total_images})")
    
    return environment_results


def main():
    print("Aggressive Regional Color Distribution Classifier")
    print("Tighter Thresholds for Better Separation")
    print("="*50)
    
    classifier = RegionalColorDistributionClassifier()
    
    print("\nAggressive Classification Strategy:")
    print("1. Open Area: top_blue_percentage > 0.15")
    print("2. Hallway: artificial_score < 0.88 AND entropy < 7.4 (TIGHTER)")
    print("3. Room: artificial_score > 0.88 AND entropy > 7.20")
    print("4. Staircase: Narrow ranges with sweet spot bonuses")
    
    results = test_on_separate_folders()
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()
            
            

# Overall Results:
# hallway: 0.273 (15/55)
# staircase: 0.462 (24/52)
# room: 0.644 (38/59)
# open_area: 0.692 (45/65)
# Overall: 0.528 (122/231)






