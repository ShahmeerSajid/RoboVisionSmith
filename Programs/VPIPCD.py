



## VPIPCD ---> CSV file maker + global feature extraction from the 700 images' training dataset.

import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import math
from sklearn.cluster import DBSCAN, KMeans
import datetime
import skimage
import sklearn
import telnetlib
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# version 2 (has better accuracy than 46%)

class ImprovedVPIPCDExtractor:
    def __init__(self):
        self.hough_threshold = 35
        self.min_line_length = 20
        self.max_line_gap = 8
        self.max_lines = 60
        
        self.vp_eps = 35
        self.vp_min_samples = 3
        
        self.target_width = 480
        self.target_height = 360
    
    """
    In this function, I resize images while maintaining their aspect ratio to ensure consistent processing 
    across different input image sizes. I calculate scaling factors for both width and height dimensions, 
    then use the smaller factor to preserve the original proportions while fitting within target dimensions. 
    When scaling is required, I use area interpolation for downsampling to preserve image quality and 
    reduce aliasing artifacts that could interfere with line detection. This consistent sizing approach 
    ensures that subsequent line detection and vanishing point analysis operate on images with similar 
    spatial characteristics, improving the reliability of geometric feature extraction. The function 
    returns both the resized image and the scaling factor, allowing me to transform detected features 
    back to original image coordinates when needed for accurate spatial analysis.
    """
    def resize_image(self, image):
        h, w = image.shape[:2]
        scale_w = self.target_width / w
        scale_h = self.target_height / h
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale
        
        return image, 1.0
    
    """
    Here I implement enhanced preprocessing specifically optimized for perspective line detection in 
    indoor environments. I start by resizing the image for consistent processing, then apply adaptive 
    histogram equalization using CLAHE to improve contrast across different regions of the image, 
    which is crucial for detecting lines under varying lighting conditions. The Gaussian blur helps 
    reduce noise while preserving important edge information, and I use carefully tuned Canny edge 
    detection parameters that balance sensitivity and noise rejection. This preprocessing pipeline 
    is designed to enhance the visibility of architectural lines like wall edges, door frames, and 
    structural elements that are essential for vanishing point analysis. The enhanced contrast and 
    edge detection ensure that perspective cues are reliably detected across different indoor lighting 
    conditions and image qualities.
    """
    def preprocess_image(self, image):
        resized_img, scale = self.resize_image(image)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        edges = cv2.Canny(blurred, 40, 120, apertureSize=3)
        
        return gray, edges, scale
    
    """
    This function implements improved line detection using multiple Hough parameter sets to capture 
    different types of lines that contribute to perspective analysis. I use both standard and relaxed 
    parameters to detect prominent architectural lines as well as shorter structural elements that 
    might be missed with single-parameter detection. The multiple detection passes help capture the 
    full range of perspective cues present in indoor environments, from major wall edges to smaller 
    architectural details. After combining the line sets, I apply advanced filtering to remove noise 
    and select high-quality lines that contribute meaningfully to vanishing point analysis. The 
    diverse line selection ensures that I maintain good representation across different line orientations 
    and lengths, which is crucial for accurate perspective analysis and environment classification.
    """
    def detect_and_filter_lines(self, edges, image_shape):
        line_sets = []
        
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
        
        filtered_lines = self.advanced_line_filter(all_lines, image_shape)
        
        if len(filtered_lines) > self.max_lines:
            filtered_lines = self.select_diverse_lines(filtered_lines, self.max_lines)
        
        return filtered_lines
    
    """
    In this advanced line filtering function, I implement comprehensive quality criteria to ensure 
    that only meaningful lines contribute to perspective analysis. I filter lines based on minimum 
    length requirements relative to image size, ensuring that detected lines represent significant 
    architectural features rather than noise. The bounds checking allows for slight extension beyond 
    image boundaries to capture perspective lines that converge outside the visible frame, which is 
    important for vanishing point detection. I also filter out near-horizontal lines in most cases 
    since they rarely contribute to indoor perspective analysis, focusing instead on vertical and 
    meaningfully angled lines that characterize architectural structures. This multi-criteria filtering 
    approach ensures that the line set used for vanishing point detection contains high-quality, 
    geometrically significant features that enhance classification accuracy.
    """
    def advanced_line_filter(self, lines, image_shape):
        if len(lines) == 0:
            return np.array([]).reshape(0, 4)
        
        h, w = image_shape[:2]
        
        lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
        
        min_length = max(15, min(w, h) * 0.05)
        length_mask = lengths >= min_length
        
        bounds_mask = ((lines[:, 0] >= -w*0.1) & (lines[:, 0] <= w*1.1) & 
                      (lines[:, 2] >= -w*0.1) & (lines[:, 2] <= w*1.1) &
                      (lines[:, 1] >= -h*0.1) & (lines[:, 1] <= h*1.1) & 
                      (lines[:, 3] >= -h*0.1) & (lines[:, 3] <= h*1.1))
        
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.abs(np.arctan2(dy, dx + 1e-6) * 180 / np.pi)
        angle_mask = (angles >= 10) | (angles <= 170)
        
        valid_mask = length_mask & bounds_mask & angle_mask
        
        return lines[valid_mask]
    
    """
    Here I implement a sophisticated line selection algorithm that maintains diversity in line orientations 
    while limiting the total number of lines for computational efficiency. When too many lines are 
    detected, I use K-means clustering to group lines by their angles, ensuring representation across 
    different directional patterns in the image. From each cluster, I select the longest lines since 
    these typically represent the most significant architectural features. This approach prevents the 
    line set from being dominated by lines of a single orientation, which could bias vanishing point 
    detection. By maintaining angular diversity while prioritizing line quality through length selection, 
    I ensure that the final line set provides comprehensive coverage of the perspective cues present 
    in the image, leading to more accurate and robust vanishing point analysis for environment classification.
    """
    def select_diverse_lines(self, lines, max_count):
        if len(lines) <= max_count:
            return lines
        
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.arctan2(dy, dx + 1e-6)
        
        try:
            kmeans = KMeans(n_clusters=min(max_count//2, len(lines)), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(angles.reshape(-1, 1))
            
            selected_lines = []
            for cluster_id in range(kmeans.n_clusters):
                cluster_lines = lines[clusters == cluster_id]
                lengths = np.sqrt((cluster_lines[:, 2] - cluster_lines[:, 0])**2 + 
                                (cluster_lines[:, 3] - cluster_lines[:, 1])**2)
                n_select = min(2, len(cluster_lines))
                top_indices = np.argsort(lengths)[-n_select:]
                selected_lines.extend(cluster_lines[top_indices])
            
            if len(selected_lines) > max_count:
                lengths = np.sqrt((np.array(selected_lines)[:, 2] - np.array(selected_lines)[:, 0])**2 + 
                                (np.array(selected_lines)[:, 3] - np.array(selected_lines)[:, 1])**2)
                top_indices = np.argsort(lengths)[-max_count:]
                selected_lines = np.array(selected_lines)[top_indices]
            
            return np.array(selected_lines)
        except:
            lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
            top_indices = np.argsort(lengths)[-max_count:]
            return lines[top_indices]
    
    """
    This function implements robust vanishing point detection using quality-weighted line intersection 
    analysis. I compute intersections between all line pairs, but apply quality scoring based on line 
    lengths and angular separation to emphasize intersections from significant, well-separated lines. 
    The intersection points are filtered based on distance from the image center, allowing for vanishing 
    points that extend beyond the image boundaries while filtering out unrealistic intersections. I 
    weight each intersection by the combined length of the contributing lines and their angular separation, 
    ensuring that vanishing points are determined by the most significant architectural features. This 
    quality-based approach helps identify meaningful perspective convergence points that characterize 
    different environment types, while filtering out noise from minor lines or near-parallel intersections 
    that don't contribute to perspective understanding.
    """
    def find_vanishing_points_robust(self, lines, image_shape):
        if len(lines) < 3:
            return [], []
        
        h, w = image_shape[:2]
        image_diagonal = np.sqrt(w**2 + h**2)
        
        intersections = []
        intersection_weights = []
        
        max_pairs = min(800, len(lines) * (len(lines) - 1) // 2)
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
                    
                    if distance_from_center < image_diagonal * 2:
                        len1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        len2 = np.sqrt((x4-x3)**2 + (y4-y3)**2)
                        
                        angle1 = np.arctan2(y2-y1, x2-x1)
                        angle2 = np.arctan2(y4-y3, x4-x3)
                        angle_diff = abs(angle1 - angle2)
                        angle_diff = min(angle_diff, np.pi - angle_diff)
                        
                        weight = (len1 + len2) * angle_diff
                        
                        intersections.append([px, py])
                        intersection_weights.append(weight)
                
                pair_count += 1
        
        if len(intersections) < 3:
            return [], []
        
        intersections = np.array(intersections)
        intersection_weights = np.array(intersection_weights)
        
        return self.cluster_weighted_intersections(intersections, intersection_weights)
    
    """
    In this function, I use weighted clustering to identify meaningful vanishing points from the 
    collection of line intersections. I apply DBSCAN clustering to group nearby intersections while 
    incorporating quality weights to emphasize more reliable intersection points. The weighted centroid 
    calculation ensures that vanishing points are positioned based on the most significant contributing 
    intersections rather than being skewed by noise. I compute quality scores for each vanishing point 
    cluster based on both the number of contributing intersections and their average weights, providing 
    a measure of confidence for each detected vanishing point. This weighted clustering approach is 
    crucial for distinguishing between genuine perspective convergence points that characterize different 
    environment types and spurious intersections that result from noise or coincidental line arrangements. 
    The resulting vanishing points provide reliable perspective cues for environment classification.
    """
    def cluster_weighted_intersections(self, intersections, weights):
        try:
            clustering = DBSCAN(eps=self.vp_eps, min_samples=self.vp_min_samples)
            clusters = clustering.fit_predict(intersections)
            
            vanishing_points = []
            vp_qualities = []
            
            for cluster_id in set(clusters):
                if cluster_id == -1:
                    continue
                
                cluster_mask = clusters == cluster_id
                cluster_points = intersections[cluster_mask]
                cluster_weights = weights[cluster_mask]
                
                total_weight = np.sum(cluster_weights)
                if total_weight > 0:
                    weighted_centroid = np.sum(cluster_points * cluster_weights.reshape(-1, 1), axis=0) / total_weight
                    vanishing_points.append(weighted_centroid)
                    
                    quality = len(cluster_points) * np.mean(cluster_weights)
                    vp_qualities.append(quality)
            
            return vanishing_points, vp_qualities
        except:
            return [], []
    
    """
    Here I analyze the spatial distribution characteristics of detected vanishing points to extract 
    features that distinguish different environment types. I compute centrality scores based on how 
    close vanishing points are to the image center, which relates to the perspective characteristics 
    of different environments - hallways often have central vanishing points while other environments 
    may show more distributed patterns. The horizontal and vertical spread measurements capture how 
    vanishing points are spatially organized, providing insight into the geometric complexity of the 
    environment. I also classify the primary vanishing point region to understand where the main 
    perspective focus occurs in the image. These distribution characteristics help distinguish between 
    environments with strong central perspective (like hallways) versus those with more complex or 
    distributed perspective patterns (like rooms or staircases), providing valuable features for 
    environment classification.
    """
    def analyze_vanishing_point_distribution(self, vanishing_points, image_shape):
        if len(vanishing_points) == 0:
            return {
                'vp_count': 0,
                'vp_centrality_score': 0,
                'vp_horizontal_spread': 0,
                'vp_vertical_spread': 0,
                'primary_vp_region': 'none'
            }
        
        h, w = image_shape[:2]
        image_center = np.array([w/2, h/2])
        
        vp_array = np.array(vanishing_points)
        
        distances_to_center = [np.linalg.norm(vp - image_center) for vp in vanishing_points]
        min_distance = min(distances_to_center)
        image_diagonal = np.sqrt(w**2 + h**2)
        centrality_score = max(0, 1 - min_distance / image_diagonal)
        
        if len(vanishing_points) > 1:
            h_spread = np.std(vp_array[:, 0]) / w
            v_spread = np.std(vp_array[:, 1]) / h
        else:
            h_spread = 0
            v_spread = 0
        
        primary_vp = vanishing_points[np.argmin(distances_to_center)]
        px, py = primary_vp
        
        if px < w/3:
            h_region = 'left'
        elif px > 2*w/3:
            h_region = 'right'
        else:
            h_region = 'center'
        
        if py < h/3:
            v_region = 'top'
        elif py > 2*h/3:
            v_region = 'bottom'
        else:
            v_region = 'middle'
        
        primary_vp_region = f"{v_region}_{h_region}"
        
        return {
            'vp_count': len(vanishing_points),
            'vp_centrality_score': centrality_score,
            'vp_horizontal_spread': h_spread,
            'vp_vertical_spread': v_spread,
            'primary_vp_region': primary_vp_region
        }
    
    """
    This function analyzes how well lines converge toward detected vanishing points, providing insight 
    into the quality and consistency of perspective patterns in the image. For each vanishing point, 
    I calculate how many lines show strong directional alignment toward that point, using both direction 
    vectors and distance relationships. Lines that point toward vanishing points with high alignment 
    scores indicate strong perspective structure, while poor convergence suggests weaker perspective 
    cues. I compute the dominant convergence angle to understand the typical perspective characteristics, 
    and measure convergence consistency to assess how uniform the perspective pattern is across different 
    lines. These convergence measures help distinguish between environments with strong, consistent 
    perspective (like hallways) versus those with weaker or more variable perspective patterns (like 
    rooms), providing important discriminatory features for environment classification based on perspective 
    geometry.
    """
    def analyze_line_convergence_patterns(self, lines, vanishing_points):
        if len(vanishing_points) == 0 or len(lines) == 0:
            return {
                'convergence_quality': 0,
                'dominant_convergence_angle': 0,
                'convergence_consistency': 0
            }
        
        convergence_scores = []
        convergence_angles = []
        
        for vp in vanishing_points:
            vp_convergence_scores = []
            vp_angles = []
            
            for line in lines:
                x1, y1, x2, y2 = line
                
                line_vec = np.array([x2-x1, y2-y1])
                line_length = np.linalg.norm(line_vec)
                
                if line_length > 0:
                    line_vec_norm = line_vec / line_length
                    
                    to_vp_vec = np.array([vp[0]-x1, vp[1]-y1])
                    to_vp_length = np.linalg.norm(to_vp_vec)
                    
                    if to_vp_length > 0:
                        to_vp_vec_norm = to_vp_vec / to_vp_length
                        
                        alignment = abs(np.dot(line_vec_norm, to_vp_vec_norm))
                        
                        if alignment > 0.6:
                            vp_convergence_scores.append(alignment)
                            
                            angle = np.arccos(np.clip(alignment, 0, 1)) * 180 / np.pi
                            vp_angles.append(angle)
            
            if vp_convergence_scores:
                convergence_scores.extend(vp_convergence_scores)
                convergence_angles.extend(vp_angles)
        
        if not convergence_scores:
            return {
                'convergence_quality': 0,
                'dominant_convergence_angle': 0,
                'convergence_consistency': 0
            }
        
        quality = np.mean(convergence_scores)
        dominant_angle = np.median(convergence_angles)
        consistency = 1 - np.std(convergence_scores)
        
        return {
            'convergence_quality': quality,
            'dominant_convergence_angle': dominant_angle,
            'convergence_consistency': max(0, consistency)
        }
    
    """
    In this function, I analyze perspective distortion patterns that provide additional cues about 
    environment characteristics and viewing geometry. I examine how line lengths change with position 
    in the image, since perspective effects typically cause lines to appear shorter as they recede 
    toward the horizon. The length gradient correlation measures whether there's a systematic change 
    in line lengths based on vertical position, which is characteristic of strong perspective views. 
    I also analyze angular distortion by comparing line angle distributions in different image regions, 
    since perspective effects can cause apparent angular changes. The overall perspective distortion 
    score combines these measures to quantify how much the image exhibits classic perspective effects. 
    These distortion patterns help distinguish between environments with strong perspective geometry 
    (like hallways) versus those with more uniform or complex spatial arrangements (like rooms), 
    providing valuable supplementary features for environment classification.
    """
    def analyze_perspective_distortion(self, lines, image_shape):
        if len(lines) == 0:
            return {
                'perspective_distortion_score': 0,
                'line_length_gradient': 0,
                'angular_distortion': 0
            }
        
        h, w = image_shape[:2]
        
        lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
        
        center_x = (lines[:, 0] + lines[:, 2]) / 2
        center_y = (lines[:, 1] + lines[:, 3]) / 2
        
        if len(lines) > 3:
            try:
                correlation = np.corrcoef(center_y, lengths)[0, 1]
                length_gradient = abs(correlation) if not np.isnan(correlation) else 0
            except:
                length_gradient = 0
        else:
            length_gradient = 0
        
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.arctan2(dy, dx + 1e-6) * 180 / np.pi
        
        top_mask = center_y < h/2
        bottom_mask = center_y >= h/2
        
        if np.sum(top_mask) > 1 and np.sum(bottom_mask) > 1:
            top_angles = angles[top_mask]
            bottom_angles = angles[bottom_mask]
            
            top_std = np.std(top_angles)
            bottom_std = np.std(bottom_angles)
            
            angular_distortion = abs(top_std - bottom_std) / 90
        else:
            angular_distortion = 0
        
        perspective_score = (length_gradient + angular_distortion) / 2
        
        return {
            'perspective_distortion_score': perspective_score,
            'line_length_gradient': length_gradient,
            'angular_distortion': angular_distortion
        }
    
    """
    Here I analyze the spatial structure and organization of lines to understand the geometric complexity 
    and regularity of the environment. I examine line parallelism by counting pairs of lines with 
    similar orientations, since environments like hallways often contain many parallel architectural 
    elements while rooms may show more diverse line orientations. The structural complexity measure 
    uses entropy analysis of the angle distribution to quantify how varied or uniform the line 
    orientations are throughout the image. High complexity indicates many different line orientations, 
    while low complexity suggests dominant directional patterns. The spatial organization score combines 
    parallelism and complexity measures to characterize how structured versus chaotic the line patterns 
    are. These structural features help distinguish between highly organized environments with regular 
    geometric patterns (like hallways and staircases) versus more complex environments with irregular 
    line arrangements (like rooms with diverse furniture and features).
    """
    def analyze_spatial_structure(self, lines, image_shape):
        if len(lines) == 0:
            return {
                'spatial_organization_score': 0,
                'line_parallelism_score': 0,
                'structural_complexity': 0
            }
        
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.arctan2(dy, dx + 1e-6) * 180 / np.pi
        angles = angles % 180
        
        if len(lines) > 1:
            parallel_pairs = 0
            total_pairs = 0
            
            for i in range(len(angles)):
                for j in range(i+1, len(angles)):
                    angle_diff = abs(angles[i] - angles[j])
                    angle_diff = min(angle_diff, 180 - angle_diff)
                    
                    if angle_diff < 15:
                        parallel_pairs += 1
                    total_pairs += 1
            
            parallelism_score = parallel_pairs / total_pairs if total_pairs > 0 else 0
        else:
            parallelism_score = 0
        
        angle_hist, _ = np.histogram(angles, bins=18, range=(0, 180))
        angle_probs = angle_hist / np.sum(angle_hist)
        angle_probs = angle_probs[angle_probs > 0]
        
        if len(angle_probs) > 1:
            entropy = -np.sum(angle_probs * np.log2(angle_probs))
            max_entropy = np.log2(len(angle_probs))
            structural_complexity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            structural_complexity = 0
        
        organization_score = (parallelism_score + (1 - structural_complexity)) / 2
        
        return {
            'spatial_organization_score': organization_score,
            'line_parallelism_score': parallelism_score,
            'structural_complexity': structural_complexity
        }
    
    """
    This comprehensive feature extraction function orchestrates the entire VPIPCD analysis pipeline 
    to extract perspective and geometric features that characterize different indoor environments. 
    I start by preprocessing the image for optimal line detection, then detect and filter lines to 
    focus on architecturally significant features. After finding vanishing points through robust 
    intersection clustering, I extract five categories of features: basic line statistics including 
    counts and length distributions, vanishing point spatial distribution characteristics, line 
    convergence quality and consistency measures, perspective distortion patterns, and spatial 
    structure organization metrics. This multi-faceted analysis captures the full range of perspective 
    and geometric cues that distinguish hallways with their linear perspective, staircases with 
    geometric patterns, rooms with complex arrangements, and open areas with distinctive perspective 
    characteristics, providing a comprehensive feature set for environment classification.
    """
    def extract_features(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            gray, edges, scale = self.preprocess_image(image)
            
            lines = self.detect_and_filter_lines(edges, gray.shape)
            
            if scale != 1.0:
                lines = lines / scale
            
            vanishing_points, vp_qualities = self.find_vanishing_points_robust(lines, image.shape)
            
            features = {'image_path': image_path}
            
            features['total_lines'] = len(lines)
            if len(lines) > 0:
                lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
                features['avg_line_length'] = np.mean(lengths)
                features['line_length_std'] = np.std(lengths)
                
                dx = lines[:, 2] - lines[:, 0]
                dy = lines[:, 3] - lines[:, 1]
                angles = np.abs(np.arctan2(dy, dx + 1e-6) * 180 / np.pi)
                
                features['line_angle_variance'] = np.var(angles)
                features['horizontal_lines_ratio'] = np.sum((angles <= 30) | (angles >= 150)) / len(lines)
                features['vertical_lines_ratio'] = np.sum((angles >= 60) & (angles <= 120)) / len(lines)
            else:
                features.update({
                    'avg_line_length': 0, 'line_length_std': 0, 'line_angle_variance': 0,
                    'horizontal_lines_ratio': 0, 'vertical_lines_ratio': 0
                })
            
            vp_features = self.analyze_vanishing_point_distribution(vanishing_points, image.shape)
            features.update(vp_features)
            
            convergence_features = self.analyze_line_convergence_patterns(lines, vanishing_points)
            features.update(convergence_features)
            
            distortion_features = self.analyze_perspective_distortion(lines, image.shape)
            features.update(distortion_features)
            
            spatial_features = self.analyze_spatial_structure(lines, image.shape)
            features.update(spatial_features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

"""
This function processes images from organized folders to extract comprehensive VPIPCD features 
and compile them into a structured dataset for environment classification analysis. I systematically 
iterate through folders containing different environment types (hallway, staircase, room, openarea), 
applying the complete VPIPCD feature extraction pipeline to each image. The function handles various 
image formats and provides progress tracking for large datasets. After extracting features from 
all images, I organize the results into a pandas DataFrame with proper column ordering and save 
the complete dataset to CSV format. This comprehensive dataset creation process provides the 
foundation for developing and evaluating classification algorithms based on vanishing point and 
perspective cue analysis. The resulting dataset contains rich geometric and perspective features 
that capture the distinctive characteristics of different indoor environments, enabling research 
into perspective-based environment classification for robotic navigation and computer vision 
applications.
"""
def process_images_to_csv():
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    folders = {
        'hallway': 'hallway_test_photos',
        'staircase': 'staircase_test_photos', 
        'room': 'room_test_photos',
        'openarea': 'openarea_test_photos'
    }
    
    extractor = ImprovedVPIPCDExtractor()
    all_features = []
    
    print("Starting improved VPIPCD feature extraction...")
    
    for category, folder_name in folders.items():
        folder_path = os.path.join(base_path, folder_name)
        print(f"\nProcessing {category} images...")
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist!")
            continue
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images")
        
        for i, image_path in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f" Progress: {i+1}/{len(image_files)}")
            
            features = extractor.extract_features(str(image_path))
            if features is not None:
                features['category'] = category
                features['filename'] = image_path.name
                all_features.append(features)
    
    if all_features:
        df = pd.DataFrame(all_features)
        
        column_order = ['filename', 'category', 'image_path'] + \
                      [col for col in df.columns if col not in ['filename', 'category', 'image_path']]
        df = df[column_order]
        
        output_file = os.path.join(base_path, 'vpipcd_dataset_700.csv')
        df.to_csv(output_file, index=False)
        
        feature_cols = [col for col in df.columns if col not in ['filename', 'category', 'image_path']]
        print(f"\nExtracted features:")
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")
        
        return df
    else:
        print("No features extracted!")
        return None

if __name__ == "__main__":
    df = process_images_to_csv()
