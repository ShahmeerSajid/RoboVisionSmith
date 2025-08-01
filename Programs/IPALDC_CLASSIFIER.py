

### IPALDC CLASSIFIER ALGORITHM --> classifying test images based on illumination

import pathlib
import seaborn
import matplotlib
import cv2
import datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
import math
from scipy import ndimage
from scipy.stats import entropy
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class IPALDCClassifier:
    
    def __init__(self):
        self.set_hardcoded_thresholds()
        
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
    In this function, I establish hardcoded classification thresholds derived from comprehensive 
    discriminative analysis of lighting features that show the highest separability between 
    environment types. I prioritize color temperature variance as the most powerful discriminator, 
    which captures the natural lighting variation in outdoor scenes versus the consistent artificial 
    lighting in indoor environments. The lighting gradient anisotropy thresholds distinguish 
    directional natural lighting from uniform artificial illumination. Blue-red ratio thresholds 
    separate cool natural daylight from warm artificial lighting. The top-to-bottom brightness 
    ratios detect sky gradients characteristic of outdoor scenes. Light source count thresholds 
    distinguish between environments with different lighting infrastructure complexity. Regional 
    brightness uniformity and lighting smoothness measures separate uniform indoor lighting from 
    variable outdoor conditions. These thresholds are optimized based on coefficient of variation 
    analysis to maximize discrimination between environment types.
    """
    def set_hardcoded_thresholds(self):
        self.color_temp_var_openarea_min = 3.0
        self.color_temp_var_room_min = 1.2
        self.color_temp_var_indoor_max = 0.5
        
        self.gradient_aniso_openarea_min = 0.20
        self.gradient_aniso_hallway_min = 0.15
        self.gradient_aniso_room_max = 0.12
        
        self.blue_red_openarea_min = 0.95
        self.blue_red_room_min = 0.85
        self.blue_red_hallway_max = 0.80
        
        self.top_bottom_openarea_min = 1.60
        self.top_bottom_indoor_max = 1.40
        
        self.light_count_room_min = 14
        self.light_count_hallway_min = 13
        self.light_count_staircase_max = 13
        
        self.brightness_uniform_hallway_min = 0.81
        self.brightness_uniform_openarea_max = 0.75
        
        self.lighting_smooth_indoor_min = 0.75
        self.lighting_smooth_openarea_max = 0.72
    
    """
    Here I implement enhanced preprocessing specifically optimized for lighting analysis across 
    multiple color spaces that capture different aspects of illumination information. I resize 
    images while maintaining aspect ratio to ensure consistent processing, then convert to various 
    color representations including BGR for color temperature analysis, grayscale for brightness 
    distribution, HSV for illumination characteristics, and LAB for perceptual color analysis. 
    The histogram equalization on the grayscale image provides lighting-invariant normalization 
    that reduces overall brightness variations while preserving relative lighting patterns. This 
    multi-representation approach allows me to analyze both absolute lighting characteristics and 
    relative illumination patterns, ensuring robust feature extraction across different imaging 
    conditions while maintaining sensitivity to the distinctive illumination characteristics that 
    differentiate various environment types based on their lighting patterns.
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
    This function extracts only the most discriminative IPALDC features that have proven most 
    effective for environment classification, focusing on computational efficiency while maintaining 
    classification accuracy. I compute color temperature variance as the primary discriminator by 
    analyzing blue-to-red channel ratio variations across pixels. The regional brightness analysis 
    examines top, middle, and bottom regions to detect characteristic patterns like sky gradients 
    in outdoor scenes or uniform lighting in indoor environments. The spatial lighting gradient 
    analysis uses grid-based computation to measure lighting directionality and anisotropy. Light 
    source counting identifies bright spots representing illumination infrastructure, while lighting 
    smoothness measures gradient variations that distinguish natural from artificial lighting. This 
    focused feature extraction approach ensures computational efficiency for real-time robotic 
    applications while capturing the essential illumination characteristics needed for reliable 
    environment classification.
    """
    def extract_key_ipaldc_features(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            images = self.preprocess_image(image)
            gray_norm = images['gray_norm']
            bgr = images['bgr']
            h, w = gray_norm.shape
            
            features = {}
            
            blue_channel = bgr[:, :, 0].astype(np.float32)
            red_channel = bgr[:, :, 2].astype(np.float32)
            blue_red_per_pixel = blue_channel / (red_channel + 1)
            features['color_temperature_variance'] = np.std(blue_red_per_pixel)
            
            features['blue_red_ratio'] = np.mean(blue_channel) / (np.mean(red_channel) + 1)
            
            top_region = gray_norm[:int(h*0.33), :]
            middle_region = gray_norm[int(h*0.33):int(h*0.67), :]
            bottom_region = gray_norm[int(h*0.67):, :]
            
            top_mean = np.mean(top_region) / 255.0
            middle_mean = np.mean(middle_region) / 255.0
            bottom_mean = np.mean(bottom_region) / 255.0
            
            features['top_to_bottom_brightness_ratio'] = (top_mean + 0.001) / (bottom_mean + 0.001)
            
            regional_means = [top_mean, middle_mean, bottom_mean]
            features['regional_brightness_uniformity'] = 1.0 - (np.std(regional_means) / (np.mean(regional_means) + 0.001))
            
            grid_h = h // self.grid_size
            grid_w = w // self.grid_size
            
            grid_brightnesses = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    start_y = i * grid_h
                    end_y = min((i + 1) * grid_h, h)
                    start_x = j * grid_w
                    end_x = min((j + 1) * grid_w, w)
                    
                    grid_cell = gray_norm[start_y:end_y, start_x:end_x]
                    cell_brightness = np.mean(grid_cell) / 255.0
                    grid_brightnesses.append(cell_brightness)
            
            grid_2d = np.array(grid_brightnesses).reshape(self.grid_size, self.grid_size)
            horizontal_gradient = np.mean(np.abs(np.gradient(grid_2d, axis=1)))
            vertical_gradient = np.mean(np.abs(np.gradient(grid_2d, axis=0)))
            
            features['lighting_gradient_anisotropy'] = abs(horizontal_gradient - vertical_gradient) / (horizontal_gradient + vertical_gradient + 0.001)
            
            bright_threshold = np.percentile(gray_norm, 95)
            bright_mask = (gray_norm > bright_threshold).astype(np.uint8)
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            significant_light_sources = [c for c in contours if cv2.contourArea(c) > 20]
            features['light_source_count'] = len(significant_light_sources)
            
            grad_x = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features['lighting_smoothness'] = 1.0 / (1.0 + np.mean(gradient_magnitude) / 255.0)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    """
    This function implements a sophisticated multi-stage IPALDC classification system that uses 
    lighting characteristics to distinguish between environment types through hierarchical decision-making. 
    I start with strong open area detection using natural lighting signatures like high color 
    temperature variance and cool blue-red ratios. Secondary open area indicators include sky 
    gradients and directional lighting patterns. Room detection focuses on high light source counts 
    combined with moderate color variation. Hallway detection emphasizes uniform artificial lighting 
    with warm color temperatures and smooth illumination gradients. The refined scoring system 
    combines multiple indicators with weighted contributions, and includes validation logic to 
    handle close scores using the strongest single indicators as tiebreakers. This approach leverages 
    the distinctive illumination patterns that characterize different environment types, providing 
    reliable classification based on lighting infrastructure and natural versus artificial lighting 
    characteristics.
    """
    def classify_environment_ipaldc(self, features):
        if features is None:
            return 'unknown'
        
        color_temp_var = features['color_temperature_variance']
        blue_red_ratio = features['blue_red_ratio']
        top_bottom_ratio = features['top_to_bottom_brightness_ratio']
        brightness_uniform = features['regional_brightness_uniformity']
        gradient_aniso = features['lighting_gradient_anisotropy']
        light_count = features['light_source_count']
        lighting_smooth = features['lighting_smoothness']
        
        if color_temp_var > self.color_temp_var_openarea_min:
            if blue_red_ratio > self.blue_red_openarea_min:
                return 'openarea'
        
        if (top_bottom_ratio > self.top_bottom_openarea_min and 
            gradient_aniso > self.gradient_aniso_openarea_min):
            if brightness_uniform < self.brightness_uniform_openarea_max:
                return 'openarea'
        
        if light_count > self.light_count_room_min:
            if color_temp_var > self.color_temp_var_room_min:
                if blue_red_ratio > self.blue_red_room_min:
                    return 'room'
        
        if brightness_uniform > self.brightness_uniform_hallway_min:
            if blue_red_ratio < self.blue_red_hallway_max:
                if lighting_smooth > self.lighting_smooth_indoor_min:
                    return 'hallway'
        
        scores = {'hallway': 0, 'staircase': 0, 'room': 0, 'openarea': 0}
        
        if color_temp_var > 2.0:
            scores['openarea'] += 3
        if blue_red_ratio > 0.9:
            scores['openarea'] += 2
        if gradient_aniso > 0.18:
            scores['openarea'] += 2
        if top_bottom_ratio > 1.55:
            scores['openarea'] += 1
        
        if brightness_uniform > 0.8:
            scores['hallway'] += 2
        if blue_red_ratio < 0.8:
            scores['hallway'] += 2
        if lighting_smooth > 0.75:
            scores['hallway'] += 1
        if light_count > 12:
            scores['hallway'] += 1
        
        if light_count > 14:
            scores['room'] += 2
        if 0.8 < blue_red_ratio < 0.95:
            scores['room'] += 2
        if 1.0 < color_temp_var < 3.0:
            scores['room'] += 1
        if gradient_aniso < 0.15:
            scores['room'] += 1
        
        if color_temp_var < 0.5:
            scores['staircase'] += 2
        if light_count < 13:
            scores['staircase'] += 1
        if 0.75 < blue_red_ratio < 0.85:
            scores['staircase'] += 1
        if 1.4 < top_bottom_ratio < 1.6:
            scores['staircase'] += 1
        
        best_category = max(scores.keys(), key=lambda k: scores[k])
        
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] <= 1:
            if color_temp_var > 3.0:
                return 'openarea'
            elif brightness_uniform > 0.82:
                return 'hallway'
            elif light_count > 15:
                return 'room'
            else:
                return best_category
        
        return best_category
    

    def classify_image(self, image_path):
        features = self.extract_key_ipaldc_features(image_path)
        return self.classify_environment_ipaldc(features)


def evaluate_ipaldc_classifier():
    classifier = IPALDCClassifier()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    test_folders = {
        'hallway': 'hallway_test_photos',
        'staircase': 'staircase_test_photos',
        'room': 'room_test_photos',
        'openarea': 'openarea_test_photos'
    }
    
    print("IPALDC Environment Classification Results")
    print("=" * 45)
    
    total_images = 0
    total_correct = 0
    category_results = {}
    
    for true_category, folder_name in test_folders.items():
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            continue
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        test_images = []
        for ext in image_extensions:
            test_images.extend(Path(folder_path).glob(f"*{ext}"))
            test_images.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if len(test_images) == 0:
            continue
        
        print(f"\n{true_category.upper()}: {len(test_images)} images")
        
        correct_predictions = 0
        predictions = {}
        
        for image_path in test_images:
            predicted = classifier.classify_image(str(image_path))
            
            if predicted == true_category:
                correct_predictions += 1
            
            predictions[predicted] = predictions.get(predicted, 0) + 1
        
        accuracy = (correct_predictions / len(test_images)) * 100
        category_results[true_category] = {
            'total': len(test_images),
            'correct': correct_predictions,
            'accuracy': accuracy,
            'predictions': predictions
        }
        
        total_images += len(test_images)
        total_correct += correct_predictions
        
        print(f"Correct: {correct_predictions}/{len(test_images)} ({accuracy:.1f}%)")
        print(f"Predictions: {dict(predictions)}")
    
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    
    print("\n" + "=" * 45)
    print("FINAL RESULTS")
    print("=" * 45)
    print(f"Overall: {total_correct}/{total_images} ({overall_accuracy:.1f}%)")
    
    for category, results in category_results.items():
        print(f"{category.capitalize()}: {results['accuracy']:.1f}%")
    
    return category_results, overall_accuracy

if __name__ == "__main__":
    results, accuracy = evaluate_ipaldc_classifier()

    
# FINAL RESULTS
# =============================================
# Overall: 115/231 (49.8%)
# Hallway: 70.9%
# Staircase: 36.5%
# Room: 27.1%
# Openarea: 63.1%


















