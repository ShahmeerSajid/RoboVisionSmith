

# SIDTM classifier 

# version 1

import cv2
import numpy as np
import os
import skimage
import sklearn
import seaborn
import pandas as pd
from datetime import datetime

class SpatialDepthTransitionClassifier:
    def __init__(self):
        self.classification_rules = {
            'open_area_detection': {
                'high_freq_strong': 1.8,
                'sharpness_strong': 16.0,
                'high_freq_moderate': 1.5,
                'sharpness_moderate': 14.0,
                'min_confidence': 0.70
            },
            
            'indoor_classification': {
                'hallway_center_strong': 2.8,
                'hallway_center_fallback': 2.0,
                
                'room_discontinuity_strong': 0.45,
                'room_discontinuity_moderate': 0.4,
                'room_sharpness_min': 12.0,
                'room_center_max': 1.8,
                'room_freq_max': 1.4,
                'room_center_moderate': 2.0,
                
                'staircase_freq_max': 1.3,
                'staircase_sharpness_max': 11.5
            }
        }
    
    """
    In this function, I extract the key depth features that have proven most discriminative for 
    environment classification based on extensive analysis of training data feature overlaps. Rather 
    than extracting hundreds of features, I focus on six essential depth characteristics that provide 
    the best separation between environment types while maintaining computational efficiency. The 
    high-frequency ratio and sharpness gradient serve as primary indicators for open area detection, 
    while center-periphery ratio excels at identifying hallways with their characteristic focus 
    patterns. Discontinuity density captures the geometric complexity that distinguishes rooms, and 
    additional edge magnitude and focus variation features provide backup discrimination. This focused 
    approach ensures robust classification while maintaining the lightweight computational requirements 
    necessary for embedded robotic systems.
    """
    def extract_depth_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            height, width = img_gray.shape
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                img_gray = cv2.resize(img_gray, (new_width, new_height))
            
            features = {}
            
            features['high_freq_ratio'] = self._calculate_frequency_ratio(img_gray)
            features['sharpness_gradient_ratio'] = self._calculate_sharpness_gradient(img_gray)
            features['center_periphery_ratio'] = self._calculate_center_periphery_depth(img_gray)
            features['discontinuity_density'] = self._calculate_discontinuity_density(img_gray)
            
            features['edge_magnitude_mean'] = self._calculate_edge_magnitude(img_gray)
            features['focus_variation'] = self._calculate_focus_variation(img_gray)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return self._get_default_features()
    
    """
    Here I calculate the ratio of high-frequency to low-frequency content in the image using Fourier 
    analysis, which serves as a crucial indicator of image sharpness and detail preservation. I compute 
    the 2D FFT of the image and analyze energy distribution in frequency space, where high frequencies 
    correspond to fine details and sharp edges while low frequencies represent broad, smooth variations. 
    The ratio between these frequency bands reveals important depth characteristics - images with high 
    frequency ratios typically contain sharp, detailed foreground elements characteristic of open areas 
    with clear atmospheric perspective. Indoor environments generally show lower ratios due to artificial 
    lighting and less pronounced depth variations. This frequency analysis provides a robust depth 
    indicator that remains stable across different lighting conditions and camera settings.
    """
    def _calculate_frequency_ratio(self, img_gray):
        try:
            fft = np.fft.fft2(img_gray)
            fft_shifted = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shifted)
            
            height, width = magnitude_spectrum.shape
            center_y, center_x = height // 2, width // 2
            
            y, x = np.ogrid[:height, :width]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = min(center_x, center_y)
            
            if max_distance > 0:
                high_freq_mask = distances > 0.6 * max_distance
                low_freq_mask = distances < 0.3 * max_distance
                
                high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
                low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask])
                
                return high_freq_energy / (low_freq_energy + 1e-6)
            else:
                return 1.0
        except Exception:
            return 1.0
    
    """
    This function calculates the sharpness gradient ratio by comparing image sharpness at different 
    scales, providing insight into how depth affects image clarity across spatial frequencies. I 
    compute the Laplacian variance at the original image resolution, then downsample the image and 
    recalculate sharpness to assess how fine details are preserved across scales. The ratio between 
    these measurements reveals depth-related characteristics - images with strong foreground-background 
    separation show higher gradient ratios because fine details remain sharp at the original scale 
    but are lost during downsampling. This measure is particularly effective for distinguishing open 
    areas with clear depth hierarchies from indoor environments with more uniform depth distributions. 
    The sharpness gradient provides complementary information to frequency analysis, enhancing the 
    robustness of depth-based environment classification.
    """
    def _calculate_sharpness_gradient(self, img_gray):
        try:
            scale_1 = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            
            if img_gray.shape[1] > 4 and img_gray.shape[0] > 4:
                img_half = cv2.resize(img_gray, (img_gray.shape[1]//2, img_gray.shape[0]//2))
                scale_2 = cv2.Laplacian(img_half, cv2.CV_64F).var()
                
                return scale_1 / (scale_2 + 1e-6)
            else:
                return 10.0
        except Exception:
            return 10.0
    
    """
    In this function, I calculate the center-to-periphery depth ratio which has proven to be the 
    most effective discriminator for hallway detection based on training data analysis. I divide 
    the image into a central region encompassing the middle 50% of the image and four peripheral 
    regions representing the top, bottom, left, and right edges. Using Laplacian variance as a 
    depth proxy, I compute the focus characteristics of each region and calculate the ratio between 
    center and periphery depth measures. Hallways characteristically show very high center-periphery 
    ratios because the vanishing point and central corridor create sharp focus in the center while 
    walls and peripheral elements appear less detailed. This geometric property of hallway perspective 
    makes the center-periphery ratio an exceptionally reliable feature for hallway identification 
    in my classification system.
    """
    def _calculate_center_periphery_depth(self, img_gray):
        try:
            height, width = img_gray.shape
            
            center_y1, center_y2 = height // 4, 3 * height // 4
            center_x1, center_x2 = width // 4, 3 * width // 4
            center_region = img_gray[center_y1:center_y2, center_x1:center_x2]
            
            if center_region.size > 0:
                center_depth = cv2.Laplacian(center_region, cv2.CV_64F).var()
            else:
                center_depth = 1
            
            periphery_regions = [
                img_gray[:center_y1, :],
                img_gray[center_y2:, :],
                img_gray[center_y1:center_y2, :center_x1],
                img_gray[center_y1:center_y2, center_x2:]
            ]
            
            periphery_depths = []
            for region in periphery_regions:
                if region.size > 0:
                    depth = cv2.Laplacian(region, cv2.CV_64F).var()
                    periphery_depths.append(depth)
            
            periphery_depth = np.mean(periphery_depths) if periphery_depths else 1
            return center_depth / (periphery_depth + 1e-6)
        except Exception:
            return 1.0
    
    """
    Here I calculate the density of depth discontinuities in the image, which serves as a key 
    indicator of geometric complexity and structural boundaries. I use Canny edge detection to 
    identify edges that likely correspond to depth boundaries rather than just texture variations, 
    then compute the ratio of edge pixels to total pixels in the image. This discontinuity density 
    measure effectively captures the geometric complexity that distinguishes different environment 
    types - rooms typically show high discontinuity density due to furniture edges, wall boundaries, 
    and complex object arrangements. In contrast, hallways and open areas generally exhibit lower 
    discontinuity densities due to their simpler geometric structures. Staircases show moderate 
    density with characteristic step patterns. This measure provides crucial information about the 
    structural organization of environments, complementing the other depth analysis techniques.
    """
    def _calculate_discontinuity_density(self, img_gray):
        try:
            edges = cv2.Canny(img_gray, 50, 150)
            total_edge_pixels = np.sum(edges > 0)
            total_pixels = img_gray.shape[0] * img_gray.shape[1]
            return total_edge_pixels / total_pixels if total_pixels > 0 else 0
        except Exception:
            return 0.3
    
    """
    This function calculates the mean edge magnitude across the image using Sobel operators to 
    assess overall edge strength, which correlates with depth and distance characteristics. I 
    compute gradients in both horizontal and vertical directions, then calculate the magnitude 
    of the combined gradient vector for each pixel. The mean of these magnitudes provides an 
    indicator of how sharp and pronounced the edges are throughout the image. Strong edge magnitudes 
    typically indicate nearby objects with clear boundaries, while weaker magnitudes suggest more 
    distant or diffuse elements. This measure serves as a backup discriminator in my classification 
    system, providing additional depth information that helps distinguish between environments when 
    primary features show ambiguous results. The edge magnitude complements other depth measures 
    by capturing the overall clarity and definition of structural elements.
    """
    def _calculate_edge_magnitude(self, img_gray):
        try:
            sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return np.mean(edge_magnitude)
        except Exception:
            return 50.0
    
    """
    In this function, I calculate focus variation across different regions of the image to understand 
    how depth-related focus changes spatially throughout the scene. I divide the image into six 
    regions including top, middle, bottom, left, center, and right sections, then compute the 
    Laplacian variance for each region as a measure of local focus quality. The standard deviation 
    of these focus values across regions indicates how much focus varies spatially, which provides 
    important depth information. High focus variation suggests complex depth arrangements with some 
    regions in sharp focus and others blurred, typical of rooms with multiple objects at different 
    distances. Low focus variation indicates more uniform depth characteristics found in simpler 
    environments like hallways. This regional focus analysis complements the other depth measures 
    by capturing the spatial distribution of focus quality throughout the environment.
    """
    def _calculate_focus_variation(self, img_gray):
        try:
            height, width = img_gray.shape
            
            regions = [
                img_gray[:height//3, :],
                img_gray[height//3:2*height//3, :],
                img_gray[2*height//3:, :],
                img_gray[:, :width//3],
                img_gray[:, width//3:2*width//3],
                img_gray[:, 2*width//3:]
            ]
            
            focus_values = []
            for region in regions:
                if region.size > 0:
                    laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
                    focus_values.append(laplacian_var)
            
            return np.std(focus_values) if focus_values else 0
        except Exception:
            return 500.0
    
    """
    When image processing fails or encounters errors, I need to return a consistent set of default 
    features that maintain the integrity of my classification pipeline while avoiding system crashes. 
    This function provides reasonable default values for all six key depth features that my classifier 
    requires, chosen to represent neutral or average cases that won't bias the classification toward 
    any particular environment type. The default values are based on typical ranges observed in the 
    training data and are designed to result in lower confidence predictions when used, appropriately 
    reflecting the uncertainty introduced by processing failures. This robust error handling ensures 
    that the classification system can continue operating even when individual images fail to process 
    correctly, maintaining system stability and reliability in real-world robotic applications where 
    occasional processing failures are inevitable.
    """
    def _get_default_features(self):
        return {
            'high_freq_ratio': 1.0,
            'sharpness_gradient_ratio': 10.0,
            'center_periphery_ratio': 1.0,
            'discontinuity_density': 0.3,
            'edge_magnitude_mean': 50.0,
            'focus_variation': 500.0
        }
    
    """
    This function implements my corrected conservative combination rule approach for environment 
    classification, designed to handle the significant feature overlap discovered in training data 
    analysis. I use a hierarchical decision-making process starting with open area detection using 
    conservative thresholds on frequency ratio and sharpness gradient, either individually for strong 
    indicators or in combination for moderate values. For indoor environments, I apply specialized 
    rules where hallways are identified by very high center-periphery ratios, rooms are detected 
    through multiple indicators including high discontinuity density and complexity patterns, and 
    staircases are recognized by consistently low complexity across multiple features. This conservative 
    approach with multiple fallback strategies ensures reliable classification even when individual 
    features show ambiguous values, providing the robust performance needed for real-world robotic 
    navigation applications.
    """
    def classify_image(self, image_path, debug=False):
        features = self.extract_depth_features(image_path)
        
        if isinstance(features, dict) and 'error' in str(features):
            return {'error': 'Failed to process image', 'predicted_class': 'unknown', 'confidence': 0.0}
        
        decision_path = []
        
        high_freq_ratio = features.get('high_freq_ratio', 1.0)
        sharpness_gradient_ratio = features.get('sharpness_gradient_ratio', 10.0)
        
        open_rules = self.classification_rules['open_area_detection']
        
        if ((high_freq_ratio >= open_rules['high_freq_strong']) or
            (sharpness_gradient_ratio >= open_rules['sharpness_strong']) or
            (high_freq_ratio >= open_rules['high_freq_moderate'] and 
             sharpness_gradient_ratio >= open_rules['sharpness_moderate'])):
            
            decision_path.append(f"Open area detected: high_freq={high_freq_ratio:.3f} >= {open_rules['high_freq_strong']} OR sharpness={sharpness_gradient_ratio:.1f} >= {open_rules['sharpness_strong']} OR (freq>={open_rules['high_freq_moderate']} AND sharp>={open_rules['sharpness_moderate']})")
            
            confidence = open_rules['min_confidence']
            if high_freq_ratio >= open_rules['high_freq_strong']:
                confidence += 0.15
            if sharpness_gradient_ratio >= open_rules['sharpness_strong']:
                confidence += 0.15
            if (high_freq_ratio >= open_rules['high_freq_moderate'] and 
                sharpness_gradient_ratio >= open_rules['sharpness_moderate']):
                confidence += 0.10
            
            return {
                'predicted_class': 'open_area',
                'confidence': min(1.0, confidence),
                'features': features,
                'decision_path': decision_path if debug else None,
                'image_path': image_path
            }
        
        center_periphery_ratio = features.get('center_periphery_ratio', 1.0)
        discontinuity_density = features.get('discontinuity_density', 0.3)
        
        decision_path.append(f"Indoor environment: high_freq={high_freq_ratio:.3f}, sharpness={sharpness_gradient_ratio:.1f}, center_ratio={center_periphery_ratio:.2f}")
        
        indoor_rules = self.classification_rules['indoor_classification']
        
        if center_periphery_ratio >= indoor_rules['hallway_center_strong']:
            confidence = 0.85
            decision_path.append(f"Hallway detected: very high center_ratio={center_periphery_ratio:.2f} >= {indoor_rules['hallway_center_strong']}")
            
            return {
                'predicted_class': 'hallway',
                'confidence': confidence,
                'features': features,
                'decision_path': decision_path if debug else None,
                'image_path': image_path
            }
        
        room_score = 0.0
        room_reasons = []
        
        if discontinuity_density >= indoor_rules['room_discontinuity_strong']:
            room_score += 0.40
            room_reasons.append(f"high discontinuity={discontinuity_density:.3f}")
        
        if (sharpness_gradient_ratio >= indoor_rules['room_sharpness_min'] and 
            center_periphery_ratio <= indoor_rules['room_center_max']):
            room_score += 0.35
            room_reasons.append(f"complexity+low_center (sharp={sharpness_gradient_ratio:.1f}, center={center_periphery_ratio:.2f})")
        
        if (discontinuity_density >= indoor_rules['room_discontinuity_moderate'] and 
            center_periphery_ratio <= indoor_rules['room_center_moderate'] and
            high_freq_ratio <= indoor_rules['room_freq_max']):
            room_score += 0.30
            room_reasons.append(f"combined_room_pattern")
        
        stair_score = 0.0
        if (high_freq_ratio <= indoor_rules['staircase_freq_max'] and 
            sharpness_gradient_ratio <= indoor_rules['staircase_sharpness_max']):
            stair_score = 0.75
            decision_path.append(f"Staircase indicators: low complexity (freq={high_freq_ratio:.3f}, sharp={sharpness_gradient_ratio:.1f})")
        
        if room_score >= 0.35:
            predicted_class = 'room'
            confidence = min(0.95, 0.60 + room_score)
            decision_path.append(f"Room detected: {', '.join(room_reasons)} (score: {room_score:.2f})")
        
        elif stair_score >= 0.70:
            predicted_class = 'staircase'
            confidence = stair_score
        
        elif center_periphery_ratio >= indoor_rules['hallway_center_fallback']:
            predicted_class = 'hallway'
            confidence = 0.65
            decision_path.append(f"Hallway (fallback): moderate center_ratio={center_periphery_ratio:.2f} >= {indoor_rules['hallway_center_fallback']}")
        
        else:
            predicted_class = 'room'
            confidence = 0.55
            decision_path.append("Room (default fallback for uncertain indoor case)")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'features': features,
            'decision_path': decision_path if debug else None,
            'image_path': image_path
        }

"""
This main function provides comprehensive testing of my Spatial Depth Transition Mapping classifier 
on organized test image folders, demonstrating the system's performance across all environment types. 
I systematically process images from dedicated folders for hallways, staircases, rooms, and open 
areas, applying my classification algorithm and comparing predictions against known ground truth 
labels. The function tracks detailed results including per-image predictions, class-wise accuracy 
metrics, and overall system performance. This comprehensive evaluation framework allows me to assess 
how well my conservative combination rule approach performs in practice and provides clear feedback 
on classification accuracy for each environment type. The testing framework also showcases the 
system's resource-efficient design and interpretable decision-making process, making it suitable 
for deployment in resource-constrained robotic applications where computational efficiency and 
reliability are paramount.
"""
def main():
    print("SPATIAL DEPTH TRANSITION MAPPING CLASSIFIER")
    print("Environment Classification for Resource-Constrained Robotics")
    print("="*65)
    
    classifier = SpatialDepthTransitionClassifier()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    test_folders = {
        'hallway': 'hallway_test_photos',
        'staircase': 'staircase_test_photos', 
        'room': 'room_test_photos',
        'open_area': 'openarea_test_photos'
    }
    
    print(f"\nTesting SIDTM classifier on test images...")
    print(f"Base path: {base_path}")
    print("-" * 65)
    
    total_images = 0
    total_correct = 0
    class_results = {}
    
    for true_class, folder_name in test_folders.items():
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No images found in {folder_name}")
            continue
        
        print(f"\nProcessing {true_class.upper()} images from {folder_name}:")
        print(f"Found {len(image_files)} images")
        
        class_results[true_class] = {
            'total': len(image_files),
            'correct': 0,
            'predictions': {}
        }
        
        for image_file in sorted(image_files):
            image_path = os.path.join(folder_path, image_file)
            
            try:
                result = classifier.classify_image(image_path, debug=False)
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                
                if predicted_class not in class_results[true_class]['predictions']:
                    class_results[true_class]['predictions'][predicted_class] = 0
                class_results[true_class]['predictions'][predicted_class] += 1
                
                is_correct = (predicted_class == true_class)
                if is_correct:
                    class_results[true_class]['correct'] += 1
                    total_correct += 1
                
                total_images += 1
                
                status = "CORRECT" if is_correct else "INCORRECT"
                print(f"  {image_file}: actual={true_class}, predicted={predicted_class} (conf: {confidence:.3f}) - {status}")
                
            except Exception as e:
                print(f"  Error processing {image_file}: {e}")
    
    print("\n" + "="*80)
    print("SIDTM CLASSIFICATION RESULTS SUMMARY")
    print("="*80)
    
    if total_images > 0:
        overall_accuracy = total_correct / total_images
        
        print("# Overall Results:")
        for true_class, results in class_results.items():
            if results['total'] > 0:
                class_accuracy = results['correct'] / results['total']
                print(f"# {true_class}: {class_accuracy:.3f} ({results['correct']}/{results['total']})")
        
        print(f"# Overall: {overall_accuracy:.3f} ({total_correct}/{total_images})")
        
        print("\n" + "-" * 50)
        print("Algorithm Summary:")
        print("- Resource-efficient rule-based approach")
        print("- No machine learning required") 
        print("- Optimized for embedded robotic systems")
        print("- Based on spatial depth transition analysis")
        print("- Uses conservative combination rules for overlapping features")
        
    else:
        print("No images were processed. Please check:")
        print("1. Base path exists and is accessible")
        print("2. Folder names match exactly")
        print("3. Folders contain valid image files")
    
    return classifier

if __name__ == "__main__":
    classifier = main()
    
    
# Overall Results obtained from this classifer are as follows:
# hallway: 0.563 
# staircase: 0.519 
# room: 0.293
# open_area: 0.615 
# Overall: 0.497  --> (maximum achievable currently)




