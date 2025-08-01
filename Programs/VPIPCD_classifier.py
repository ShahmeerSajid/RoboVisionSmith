
# VPIPCD_ classifier  --> Actua classifier to categorize the image into an environment depending on the features extracted.


import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import math
import telnetlib
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class EnhancedVPIPCDClassifier:
    def __init__(self):
        self.set_hardcoded_thresholds()
        
        self.hough_threshold = 35
        self.min_line_length = 20
        self.max_line_gap = 8
        self.max_lines = 60
        self.vp_eps = 35
        self.vp_min_samples = 3
        self.target_width = 480
        self.target_height = 360
    """   
    In this function, I establish hardcoded classification thresholds derived from comprehensive 
    analysis of training data distributions and quantile analysis. I prioritize protecting the 
    classifications that already work well (hallways and open areas) while developing enhanced 
    criteria for challenging categories like staircases. The thresholds are based on statistical 
    analysis of feature distributions across different environment types, using percentile values 
    to establish reliable decision boundaries. For staircase detection, I focus on characteristics 
    like vanishing point spread, perspective distortion, convergence quality, and angle variance 
    that distinguish staircases from other environment types. The room detection criteria emphasize 
    shorter line lengths and specific centrality patterns, while hallway and open area thresholds 
    maintain the proven discrimination criteria that have shown good performance in previous testing.
    """
    
    
    def set_hardcoded_thresholds(self):
        self.hallway_v_ratio_min = 0.32
        self.openarea_h_ratio_min = 0.50
        self.openarea_length_min = 650
        
        self.staircase_vp_spread_min = 0.025
        self.staircase_perspective_min = 0.08
        self.staircase_convergence_max = 0.895
        self.staircase_angle_var_min = 650
        
        self.room_length_max = 580
        self.room_vp_centrality_min = 0.905
        
        self.hallway_convergence_min = 0.875
    
    """
    Here I implement enhanced preprocessing that optimizes images for perspective line detection 
    and geometric analysis. I resize images while maintaining aspect ratio to ensure consistent 
    processing across different input sizes, using area interpolation for quality preservation 
    during downsampling. The preprocessing pipeline includes adaptive histogram equalization using 
    CLAHE to improve contrast across different image regions, which is essential for detecting 
    architectural lines under varying lighting conditions. I apply Gaussian blur to reduce noise 
    while preserving important edge information, then use carefully tuned Canny edge detection 
    parameters that balance sensitivity and noise rejection. This preprocessing approach ensures 
    that perspective cues like wall edges, door frames, and structural elements are reliably 
    detected across different indoor lighting conditions and image qualities, providing a solid 
    foundation for subsequent geometric analysis.
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
            scale = 1.0
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120, apertureSize=3)
        
        return gray, edges, scale
    
    """
    This function implements robust line detection using dual Hough parameter sets to capture 
    both prominent architectural lines and shorter structural elements that contribute to perspective 
    analysis. I use standard parameters for detecting major features and relaxed parameters for 
    shorter lines, ensuring comprehensive coverage of perspective cues. The filtering process 
    applies multiple quality criteria including minimum length requirements relative to image 
    size, bounds checking that allows for perspective lines extending beyond the visible frame, 
    and angle filtering to focus on meaningful architectural orientations. When too many lines 
    are detected, I prioritize the longest lines since these typically represent the most significant 
    architectural features. This approach ensures that the line set used for vanishing point 
    analysis contains high-quality, geometrically significant features while maintaining computational 
    efficiency through reasonable line count limits.
    """
    def detect_and_filter_lines(self, edges, image_shape):
        lines1 = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                threshold=self.hough_threshold,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        
        lines2 = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=max(15, self.hough_threshold//2),
                                minLineLength=self.min_line_length//2,
                                maxLineGap=self.max_line_gap*2)
        
        all_lines = []
        if lines1 is not None:
            all_lines.extend(lines1.reshape(-1, 4))
        if lines2 is not None:
            all_lines.extend(lines2.reshape(-1, 4))
        
        if not all_lines:
            return np.array([]).reshape(0, 4)
        
        all_lines = np.array(all_lines)
        
        h, w = image_shape[:2]
        lengths = np.sqrt((all_lines[:, 2] - all_lines[:, 0])**2 + 
                         (all_lines[:, 3] - all_lines[:, 1])**2)
        
        min_length = max(15, min(w, h) * 0.05)
        length_mask = lengths >= min_length
        
        bounds_mask = ((all_lines[:, 0] >= -w*0.1) & (all_lines[:, 0] <= w*1.1) & 
                      (all_lines[:, 2] >= -w*0.1) & (all_lines[:, 2] <= w*1.1) &
                      (all_lines[:, 1] >= -h*0.1) & (all_lines[:, 1] <= h*1.1) & 
                      (all_lines[:, 3] >= -h*0.1) & (all_lines[:, 3] <= h*1.1))
        
        dx = all_lines[:, 2] - all_lines[:, 0]
        dy = all_lines[:, 3] - all_lines[:, 1]
        angles = np.abs(np.arctan2(dy, dx + 1e-6) * 180 / np.pi)
        angle_mask = (angles >= 10) | (angles <= 170)
        
        valid_mask = length_mask & bounds_mask & angle_mask
        filtered_lines = all_lines[valid_mask]
        
        if len(filtered_lines) > self.max_lines:
            lengths = np.sqrt((filtered_lines[:, 2] - filtered_lines[:, 0])**2 + 
                            (filtered_lines[:, 3] - filtered_lines[:, 1])**2)
            top_indices = np.argsort(lengths)[-self.max_lines:]
            filtered_lines = filtered_lines[top_indices]
        
        return filtered_lines
    
    """
    Here I implement enhanced vanishing point detection that computes quality-weighted line 
    intersections while applying stricter geometric constraints for more reliable results. I 
    calculate intersections between line pairs but filter them based on proximity to the image 
    center and require meaningful angular separation between contributing lines to avoid spurious 
    intersections from near-parallel lines. The weighting scheme considers both line lengths 
    and angular differences to emphasize intersections from significant, well-separated architectural 
    features. I use DBSCAN clustering to group nearby intersections into vanishing point candidates, 
    then compute comprehensive analysis metrics including horizontal and vertical spread patterns 
    and centrality scores. This enhanced approach provides more reliable vanishing point detection 
    while generating detailed perspective analysis that characterizes the geometric organization 
    of different environment types.
    """
    def find_vanishing_points_enhanced(self, lines, image_shape):
        if len(lines) < 3:
            return [], {}
        
        h, w = image_shape[:2]
        image_diagonal = np.sqrt(w**2 + h**2)
        image_center = np.array([w/2, h/2])
        
        intersections = []
        intersection_weights = []
        
        max_pairs = min(250, len(lines) * (len(lines) - 1) // 2)
        pair_count = 0
        
        for i in range(len(lines)):
            if pair_count >= max_pairs:
                break
            for j in range(i + 1, len(lines)):
                if pair_count >= max_pairs:
                    break
                
                x1, y1, x2, y2 = lines[i]
                x3, y3, x4, y4 = lines[j]
                
                denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                
                if abs(denom) > 1e-6:
                    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                    
                    distance_from_center = np.sqrt((px - w/2)**2 + (py - h/2)**2)
                    
                    if distance_from_center < image_diagonal:
                        len1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        len2 = np.sqrt((x4-x3)**2 + (y4-y3)**2)
                        
                        angle1 = np.arctan2(y2-y1, x2-x1)
                        angle2 = np.arctan2(y4-y3, x4-x3)
                        angle_diff = abs(angle1 - angle2)
                        angle_diff = min(angle_diff, np.pi - angle_diff)
                        
                        if angle_diff > 0.2:
                            weight = (len1 + len2) * angle_diff
                            intersections.append([px, py])
                            intersection_weights.append(weight)
                
                pair_count += 1
        
        if len(intersections) < 2:
            default_analysis = {
                'spread_metrics': {
                    'horizontal_spread': 0,
                    'vertical_spread': 0,
                    'total_spread': 0
                },
                'centrality_score': 0
            }
            return [], default_analysis
        
        try:
            intersections = np.array(intersections)
            clustering = DBSCAN(eps=25, min_samples=2)
            clusters = clustering.fit_predict(intersections)
            
            vanishing_points = []
            vp_analysis = {
                'spread_metrics': {
                    'horizontal_spread': 0,
                    'vertical_spread': 0,
                    'total_spread': 0
                },
                'centrality_score': 0
            }
            
            for cluster_id in set(clusters):
                if cluster_id != -1:
                    cluster_mask = clusters == cluster_id
                    cluster_points = intersections[cluster_mask]
                    vp = np.mean(cluster_points, axis=0)
                    vanishing_points.append(vp)
            
            if len(vanishing_points) > 1:
                vp_array = np.array(vanishing_points)
                vp_analysis['spread_metrics']['horizontal_spread'] = np.std(vp_array[:, 0]) / w
                vp_analysis['spread_metrics']['vertical_spread'] = np.std(vp_array[:, 1]) / h
                vp_analysis['spread_metrics']['total_spread'] = np.std(vp_array.flatten())
            
            if len(vanishing_points) > 0:
                distances = [np.linalg.norm(np.array(vp) - image_center) for vp in vanishing_points]
                vp_analysis['centrality_score'] = max(0, 1 - min(distances) / image_diagonal)
            
            return vanishing_points, vp_analysis
        except:
            default_analysis = {
                'spread_metrics': {
                    'horizontal_spread': 0,
                    'vertical_spread': 0,
                    'total_spread': 0
                },
                'centrality_score': 0
            }
            return [], default_analysis
    
    """
    This function extracts enhanced geometric and perspective features specifically designed for 
    environment classification using both basic line statistics and advanced perspective analysis. 
    I compute core geometric features including line orientation ratios for horizontal, vertical, 
    and diagonal directions, average line lengths, and angle variance measures. The enhanced 
    perspective analysis includes line length gradient calculation using correlation analysis to 
    detect perspective foreshortening effects, convergence quality assessment based on vanishing 
    point characteristics, and specialized staircase signature computation that combines multiple 
    geometric indicators. I also calculate vertical-to-horizontal ratios and vanishing point 
    spread metrics that help distinguish between different environment types. This comprehensive 
    feature set captures both the basic geometric properties and sophisticated perspective cues 
    that characterize the distinctive spatial organization of hallways, staircases, rooms, and 
    open areas.
    """
    def extract_enhanced_features(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            gray, edges, scale = self.preprocess_image(image)
            lines = self.detect_and_filter_lines(edges, gray.shape)
            
            if scale != 1.0:
                lines = lines / scale
            
            vanishing_points, vp_analysis = self.find_vanishing_points_enhanced(lines, image.shape)
            
            if len(lines) < 5:
                return None
            
            lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
            dx = lines[:, 2] - lines[:, 0]
            dy = lines[:, 3] - lines[:, 1]
            angles = np.abs(np.arctan2(dy, dx + 1e-6) * 180 / np.pi)
            
            horizontal_lines_ratio = np.sum((angles <= 30) | (angles >= 150)) / len(lines)
            vertical_lines_ratio = np.sum((angles >= 60) & (angles <= 120)) / len(lines)
            diagonal_lines_ratio = np.sum((angles > 30) & (angles < 60)) / len(lines)
            
            avg_line_length = np.mean(lengths)
            line_angle_variance = np.var(angles)
            vp_count = len(vanishing_points)
            
            h, w = image.shape[:2]
            center_y = (lines[:, 1] + lines[:, 3]) / 2
            
            if len(lines) > 3:
                try:
                    correlation = np.corrcoef(center_y, lengths)[0, 1]
                    line_length_gradient = abs(correlation) if not np.isnan(correlation) else 0
                except:
                    line_length_gradient = 0
            else:
                line_length_gradient = 0
            
            convergence_quality = min(1.0, vp_count * len(lines) / 200.0)
            
            vh_ratio = vertical_lines_ratio / (horizontal_lines_ratio + 0.001)
            
            staircase_signature = (line_length_gradient * 2 + 
                                 diagonal_lines_ratio * 3 + 
                                 vp_analysis['spread_metrics']['total_spread'] / 100)
            
            return {
                'horizontal_lines_ratio': horizontal_lines_ratio,
                'vertical_lines_ratio': vertical_lines_ratio,
                'diagonal_lines_ratio': diagonal_lines_ratio,
                'avg_line_length': avg_line_length,
                'line_angle_variance': line_angle_variance,
                'vp_count': vp_count,
                'vh_ratio': vh_ratio,
                'line_length_gradient': line_length_gradient,
                'convergence_quality': convergence_quality,
                'vp_horizontal_spread': vp_analysis['spread_metrics']['horizontal_spread'],
                'vp_vertical_spread': vp_analysis['spread_metrics']['vertical_spread'],
                'vp_centrality_score': vp_analysis['centrality_score'],
                'staircase_signature': staircase_signature
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    """
    This function implements a sophisticated five-stage hierarchical classification system that 
    prioritizes staircase detection while maintaining strong performance for other environment 
    types. I start by protecting existing good classifications for open areas and hallways using 
    proven criteria, then apply enhanced staircase detection through multiple geometric signatures 
    including perspective distortion patterns, vanishing point spread characteristics, and combined 
    diagonal-perspective indicators. The refined room detection uses shorter line lengths and 
    specific centrality patterns, while secondary classifications handle edge cases missed in 
    primary detection. Finally, I provide intelligent default assignment with preference for 
    staircase classification in ambiguous cases. This approach addresses the challenge of overlapping 
    feature distributions between environment types by using multiple complementary detection 
    strategies and prioritizing the most challenging classification categories through enhanced 
    signature analysis.
    """
    
    
    def classify_with_staircase_priority(self, features):
        if features is None:
            return 'unknown'
        
        h_ratio = features['horizontal_lines_ratio']
        v_ratio = features['vertical_lines_ratio']
        diag_ratio = features['diagonal_lines_ratio']
        avg_length = features['avg_line_length']
        angle_var = features['line_angle_variance']
        vh_ratio = features['vh_ratio']
        length_grad = features['line_length_gradient']
        convergence = features['convergence_quality']
        vp_h_spread = features['vp_horizontal_spread']
        vp_centrality = features['vp_centrality_score']
        staircase_sig = features['staircase_signature']
        
        if h_ratio > self.openarea_h_ratio_min and avg_length > self.openarea_length_min:
            return 'openarea'
        
        if v_ratio > self.hallway_v_ratio_min and vh_ratio > 1.5:
            if convergence > self.hallway_convergence_min:
                return 'hallway'
        
        if length_grad > self.staircase_perspective_min and vp_h_spread > self.staircase_vp_spread_min:
            if angle_var > self.staircase_angle_var_min:
                return 'staircase'
        
        if staircase_sig > 0.15 and diag_ratio > 0.20:
            if 0.8 < vh_ratio < 2.0:
                return 'staircase'
        
        if features['vp_count'] > 20 and convergence < self.staircase_convergence_max:
            if length_grad > 0.05 and angle_var > 600:
                return 'staircase'
        
        if avg_length < self.room_length_max:
            if vp_centrality > self.room_vp_centrality_min:
                if 0.35 < h_ratio < 0.55 and v_ratio < 0.35:
                    return 'room'
        
        if vh_ratio > 1.8 and v_ratio > 0.28:
            return 'hallway'
        
        if h_ratio > 0.45 and avg_length > 500:
            if angle_var < 600:
                return 'openarea'
        
        if avg_length < 550 and features['vp_count'] > 25:
            return 'room'
        
        if 0.6 < vh_ratio < 1.8 and diag_ratio > 0.15:
            return 'staircase'
        elif h_ratio > 0.40:
            return 'room'
        else:
            return 'staircase'
    
    """
    This wrapper function provides a simple interface for image classification by combining feature 
    extraction and classification into a single method call. I extract enhanced geometric and 
    perspective features from the input image, then apply the staircase-priority classification 
    algorithm to determine the environment type. This streamlined interface makes it easy to 
    classify individual images while maintaining access to the sophisticated feature extraction 
    and multi-stage classification logic. The function handles all the complex geometric analysis 
    internally and returns a simple classification result, making it suitable for integration 
    into larger systems or batch processing applications. If feature extraction fails, the 
    classification system gracefully handles the error and returns an appropriate result through 
    the robust error handling built into the feature extraction pipeline.
    """
    def classify_image(self, image_path):
        features = self.extract_enhanced_features(image_path)
        return self.classify_with_staircase_priority(features)



def evaluate_classifier():
    classifier = EnhancedVPIPCDClassifier()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    test_folders = {
        'hallway': 'hallway_test_photos',
        'staircase': 'staircase_test_photos',
        'room': 'room_test_photos',
        'openarea': 'openarea_test_photos'
    }
    
    print("Enhanced VPIPCD Results (Staircase-Focused)")
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
    results, accuracy = evaluate_classifier()
        
        
    
    
# FINAL RESULTS ---> from this classifer
# =============================================
# Overall: 110/231 (47.6%)
# Hallway: 54.5%
# Staircase: 42.3%
# Room: 27.1%
# Openarea: 64.6%  --> Nice!