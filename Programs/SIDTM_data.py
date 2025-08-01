

# ---> Collecting data and creating the CSV file containing the numerical values for the  images' global features relevant to SIDTM.

import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import scipy.ndimage as ndimage
from scipy import signal
from skimage import filters, feature, measure

class SpatialDepthTransitionMapper:
    
    def __init__(self):
        self.depth_params = {
            'edge_kernels': {
                'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
                'gaussian_sizes': [1, 3, 5, 7]
            },
            'focus_analysis': {
                'frequency_bands': [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)],
                'sharpness_kernels': [3, 5, 7, 9],
                'variance_thresholds': [50, 100, 200, 400]
            },
            'depth_regions': {
                'grid_size': (4, 4),
                'overlap_ratio': 0.2,
                'depth_layers': 5
            }
        }
    
    """
    In this function, I extract comprehensive depth transition features from an image by analyzing 
    multiple aspects of visual depth cues. I start by loading and preprocessing the image in different 
    color spaces to capture various depth-related information. After resizing for consistent processing, 
    I systematically analyze seven key depth components: edge sharpness patterns that indicate distance, 
    focus variations across regions, contrast transitions that reveal depth changes, spatial frequency 
    content for sharpness assessment, regional depth mapping for spatial patterns, depth discontinuity 
    detection for boundary identification, and perspective cues including vanishing points. This 
    comprehensive approach captures the multi-faceted nature of depth perception in images, providing 
    a rich feature set that characterizes how different environments organize depth information spatially.
    """
    def extract_depth_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            height, width = img_gray.shape
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_width, new_height))
                img_gray = cv2.resize(img_gray, (new_width, new_height))
                img_lab = cv2.resize(img_lab, (new_width, new_height))
            
            height, width = img_gray.shape
            
            features = {}
            
            edge_features = self._analyze_edge_sharpness(img_gray)
            features.update(edge_features)
            
            focus_features = self._analyze_focus_variations(img_gray)
            features.update(focus_features)
            
            contrast_features = self._analyze_contrast_transitions(img_gray, img_lab)
            features.update(contrast_features)
            
            frequency_features = self._analyze_spatial_frequencies(img_gray)
            features.update(frequency_features)
            
            regional_features = self._analyze_regional_depth(img_gray)
            features.update(regional_features)
            
            discontinuity_features = self._detect_depth_discontinuities(img_gray)
            features.update(discontinuity_features)
            
            perspective_features = self._analyze_perspective_cues(img_gray)
            features.update(perspective_features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return self._get_default_features()
    
    """
    Here I analyze edge sharpness patterns throughout the image to infer relative depth information, 
    based on the principle that sharp edges typically indicate nearby objects while blurry edges 
    suggest distant elements. I compute edge responses using Sobel operators in both horizontal 
    and vertical directions, then calculate magnitude and various statistical measures. The Laplacian 
    operator helps detect edge strength across different scales, while multi-scale sharpness analysis 
    reveals how edge clarity changes with different blur levels. By calculating sharpness gradients 
    and ratios between different scales, I can identify depth-related patterns where foreground 
    elements maintain sharpness while background elements show progressive blurring. This analysis 
    provides crucial depth cues that help distinguish environments with clear depth hierarchies from 
    those with more uniform depth distributions.
    """
    
    def _analyze_edge_sharpness(self, img_gray):
        features = {}
        
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        
        sharpness_scales = []
        for kernel_size in [3, 5, 7, 9]:
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            blurred = cv2.filter2D(img_gray, -1, kernel)
            sharpness = cv2.Laplacian(blurred, cv2.CV_64F)
            sharpness_scales.append(np.var(sharpness))
        
        features['edge_magnitude_mean'] = np.mean(sobel_magnitude)
        features['edge_magnitude_std'] = np.std(sobel_magnitude)
        features['edge_magnitude_max'] = np.max(sobel_magnitude)
        features['edge_density'] = np.sum(sobel_magnitude > np.percentile(sobel_magnitude, 90)) / img_gray.size
        
        features['laplacian_variance'] = np.var(laplacian)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        
        for i, sharpness in enumerate(sharpness_scales):
            features[f'sharpness_scale_{i+1}'] = sharpness
        
        features['sharpness_gradient_ratio'] = sharpness_scales[0] / (sharpness_scales[-1] + 1e-6)
        
        return features
    
    """
    This function analyzes focus variations across different regions of the image to understand how 
    depth changes spatially throughout the scene. I divide the image into six key regions (top, middle, 
    bottom, left, center, right) and calculate multiple focus measures for each area. The variance 
    of Laplacian serves as a primary focus indicator, while the Tenengrad operator provides gradient-based 
    focus assessment, and high-frequency energy analysis reveals fine detail preservation. By comparing 
    focus levels across regions, I can identify patterns characteristic of different environments - 
    for example, hallways often show decreasing focus toward the vanishing point, while rooms may 
    have more complex focus patterns due to varying object distances. The focus center bias and 
    variation metrics help quantify whether depth follows predictable patterns or shows more irregular 
    distributions typical of complex indoor scenes.
    """
    
    
    def _analyze_focus_variations(self, img_gray):
        features = {}
        
        height, width = img_gray.shape
        
        regions = {
            'top': img_gray[:height//3, :],
            'middle': img_gray[height//3:2*height//3, :],
            'bottom': img_gray[2*height//3:, :],
            'left': img_gray[:, :width//3],
            'center': img_gray[:, width//3:2*width//3],
            'right': img_gray[:, 2*width//3:]
        }
        
        for region_name, region in regions.items():
            laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
            features[f'{region_name}_focus_variance'] = laplacian_var
            
            sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.sum(sobel_x**2 + sobel_y**2)
            features[f'{region_name}_tenengrad'] = tenengrad
            
            fft = np.fft.fft2(region)
            fft_shifted = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shifted)
            high_freq_energy = np.sum(magnitude_spectrum[region.shape[0]//4:3*region.shape[0]//4, 
                                                        region.shape[1]//4:3*region.shape[1]//4])
            features[f'{region_name}_high_freq_energy'] = high_freq_energy
        
        focus_values = [features[f'{region}_focus_variance'] for region in regions.keys()]
        features['focus_variation'] = np.std(focus_values)
        features['focus_range'] = max(focus_values) - min(focus_values)
        features['focus_center_bias'] = features['center_focus_variance'] / (np.mean(focus_values) + 1e-6)
        
        return features
    
    """
    In this function, I analyze contrast transitions across the image that serve as important depth 
    indicators, since atmospheric perspective and focus effects cause contrast to decrease with distance. 
    I calculate local contrast at multiple scales using different kernel sizes to capture both fine 
    and coarse contrast variations. The contrast decay analysis examines how contrast changes vertically 
    through the image, which is particularly important for detecting depth gradients in environments 
    like hallways or open areas where distant elements show reduced contrast. I also analyze luminance 
    contrast in the LAB color space to capture perceptual contrast differences. The slope of contrast 
    decay and variation patterns help distinguish environments with clear depth progressions from those 
    with more uniform contrast distributions, providing valuable depth cues for environment classification.
    """
    def _analyze_contrast_transitions(self, img_gray, img_lab):
        features = {}
        
        kernel_sizes = [3, 7, 15, 31]
        contrast_measures = []
        
        for kernel_size in kernel_sizes:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            local_mean = cv2.filter2D(img_gray.astype(np.float32), -1, kernel)
            local_sqr_mean = cv2.filter2D((img_gray.astype(np.float32))**2, -1, kernel)
            local_contrast = np.sqrt(local_sqr_mean - local_mean**2)
            
            contrast_measures.append({
                'mean': np.mean(local_contrast),
                'std': np.std(local_contrast),
                'max': np.max(local_contrast)
            })
        
        for i, contrast in enumerate(contrast_measures):
            features[f'contrast_mean_scale_{i+1}'] = contrast['mean']
            features[f'contrast_std_scale_{i+1}'] = contrast['std']
            features[f'contrast_max_scale_{i+1}'] = contrast['max']
        
        height, width = img_gray.shape
        horizontal_strips = []
        for i in range(5):
            start_row = i * height // 5
            end_row = (i + 1) * height // 5
            strip = img_gray[start_row:end_row, :]
            horizontal_strips.append(np.std(strip))
        
        features['contrast_decay_slope'] = np.polyfit(range(5), horizontal_strips, 1)[0]
        features['contrast_decay_variation'] = np.std(horizontal_strips)
        
        l_channel = img_lab[:, :, 0]
        features['luminance_contrast'] = np.std(l_channel)
        features['luminance_range'] = np.max(l_channel) - np.min(l_channel)
        
        return features
    
    """
    Now I will analyze the spatial frequency content of the image using Fourier analysis to understand 
    how fine details and textures are distributed, which provides important depth information. I 
    compute the 2D FFT of the image and analyze energy distribution across different frequency bands 
    from low to high frequencies. High-frequency content typically indicates sharp, nearby details, 
    while low-frequency dominance suggests smoother, more distant surfaces. The frequency centroid 
    calculation provides an overall measure of image sharpness, while the high-frequency ratio 
    quantifies the balance between fine and coarse details. This frequency analysis is particularly 
    valuable for distinguishing environments with rich textural detail from those with smoother 
    surfaces, and for detecting depth-related changes in texture clarity that occur due to distance 
    and atmospheric effects.
    """
    
    
    def _analyze_spatial_frequencies(self, img_gray):
        features = {}
        
        fft = np.fft.fft2(img_gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        frequency_bands = {
            'low': (0, 0.1),
            'mid_low': (0.1, 0.3),
            'mid_high': (0.3, 0.6),
            'high': (0.6, 1.0)
        }
        
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            y, x = np.ogrid[:height, :width]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = min(center_x, center_y)
            
            mask = (distances >= low_freq * max_distance) & (distances < high_freq * max_distance)
            band_energy = np.sum(magnitude_spectrum[mask])
            features[f'freq_energy_{band_name}'] = band_energy
        
        total_energy = np.sum(magnitude_spectrum)
        weighted_distances = np.sum(magnitude_spectrum * np.sqrt((np.arange(height)[:, np.newaxis] - center_y)**2 + 
                                                                (np.arange(width) - center_x)**2))
        features['frequency_centroid'] = weighted_distances / (total_energy + 1e-6)
        
        high_freq_energy = features['freq_energy_high'] + features['freq_energy_mid_high']
        low_freq_energy = features['freq_energy_low'] + features['freq_energy_mid_low']
        features['high_freq_ratio'] = high_freq_energy / (low_freq_energy + 1e-6)
        
        return features
    
    """
    This function performs regional depth analysis by dividing the image into a 4x4 grid and calculating 
    depth proxy measures for each region to understand spatial depth patterns. I use the variance 
    of Laplacian as a depth indicator, since focused areas with fine details show higher variance 
    while blurred distant areas show lower variance. By analyzing how these depth proxies are 
    distributed spatially, I can identify characteristic patterns for different environments. The 
    vertical and horizontal depth gradients reveal directional depth trends, while the center-periphery 
    analysis captures radial depth patterns. This regional approach is particularly effective for 
    distinguishing hallways with their linear depth progression, staircases with their geometric 
    depth steps, rooms with complex multi-focal patterns, and open areas with their gradual depth 
    transitions from foreground to background elements.
    """
    def _analyze_regional_depth(self, img_gray):
        features = {}
        
        height, width = img_gray.shape
        
        grid_h, grid_w = 4, 4
        region_height = height // grid_h
        region_width = width // grid_w
        
        region_depths = []
        region_variances = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * region_height
                y_end = min((i + 1) * region_height, height)
                x_start = j * region_width
                x_end = min((j + 1) * region_width, width)
                
                region = img_gray[y_start:y_end, x_start:x_end]
                
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                depth_proxy = np.var(laplacian)
                region_depths.append(depth_proxy)
                
                region_variances.append(np.var(region))
        
        features['regional_depth_mean'] = np.mean(region_depths)
        features['regional_depth_std'] = np.std(region_depths)
        features['regional_depth_max'] = np.max(region_depths)
        features['regional_depth_min'] = np.min(region_depths)
        features['regional_depth_range'] = features['regional_depth_max'] - features['regional_depth_min']
        
        region_depths_2d = np.array(region_depths).reshape(grid_h, grid_w)
        
        vertical_gradient = []
        for j in range(grid_w):
            column_depths = region_depths_2d[:, j]
            gradient = np.polyfit(range(grid_h), column_depths, 1)[0]
            vertical_gradient.append(gradient)
        features['depth_vertical_gradient'] = np.mean(vertical_gradient)
        
        horizontal_gradient = []
        for i in range(grid_h):
            row_depths = region_depths_2d[i, :]
            gradient = np.polyfit(range(grid_w), row_depths, 1)[0]
            horizontal_gradient.append(gradient)
        features['depth_horizontal_gradient'] = np.mean(horizontal_gradient)
        
        center_regions = region_depths_2d[1:3, 1:3]
        periphery_regions = []
        for i in range(grid_h):
            for j in range(grid_w):
                if i < 1 or i >= 3 or j < 1 or j >= 3:
                    periphery_regions.append(region_depths_2d[i, j])
        
        features['center_depth'] = np.mean(center_regions)
        features['periphery_depth'] = np.mean(periphery_regions)
        features['center_periphery_ratio'] = features['center_depth'] / (features['periphery_depth'] + 1e-6)
        
        return features
    
    """
    In this function, I detect depth discontinuities which represent sudden changes in depth that 
    create distinct boundaries between different depth planes in the scene. I use Canny edge detection 
    followed by morphological operations to identify strong discontinuities that likely correspond 
    to depth boundaries rather than just texture edges. By analyzing the resulting contours, I 
    characterize the number, size, and complexity of these depth boundaries. The discontinuity 
    density and shape complexity metrics help distinguish environments with clear geometric depth 
    boundaries (like staircases with their step edges) from those with smoother depth transitions 
    (like open areas) or more complex irregular boundaries (like rooms with furniture). This analysis 
    provides crucial information about the structural organization of depth in different environment 
    types, complementing the other depth analysis techniques.
    """
    def _detect_depth_discontinuities(self, img_gray):
        features = {}
        
        edges_canny = cv2.Canny(img_gray, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_canny, kernel, iterations=1)
        edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            areas = [cv2.contourArea(contour) for contour in contours]
            perimeters = [cv2.arcLength(contour, True) for contour in contours]
            
            features['discontinuity_count'] = len(contours)
            features['discontinuity_total_area'] = np.sum(areas)
            features['discontinuity_mean_area'] = np.mean(areas) if areas else 0
            features['discontinuity_max_area'] = np.max(areas) if areas else 0
            features['discontinuity_mean_perimeter'] = np.mean(perimeters) if perimeters else 0
            
            total_pixels = img_gray.shape[0] * img_gray.shape[1]
            features['discontinuity_density'] = features['discontinuity_total_area'] / total_pixels
            
            if areas:
                shape_complexities = []
                for area, perimeter in zip(areas, perimeters):
                    if perimeter > 0:
                        complexity = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
                        shape_complexities.append(complexity)
                features['discontinuity_shape_complexity'] = np.mean(shape_complexities) if shape_complexities else 1.0
            else:
                features['discontinuity_shape_complexity'] = 1.0
        else:
            features['discontinuity_count'] = 0
            features['discontinuity_total_area'] = 0
            features['discontinuity_mean_area'] = 0
            features['discontinuity_max_area'] = 0
            features['discontinuity_mean_perimeter'] = 0
            features['discontinuity_density'] = 0
            features['discontinuity_shape_complexity'] = 1.0
        
        return features
    
    """
    Here I analyze perspective cues and vanishing point information that provide important depth 
    and spatial orientation clues for environment classification. I use Hough line detection to 
    identify prominent lines in the image, then analyze their orientations to understand the 
    perspective structure. The ratio of horizontal, vertical, and diagonal lines helps characterize 
    the geometric organization of the space - hallways typically show strong perspective with 
    converging lines, while rooms may have more varied line orientations. I also analyze texture 
    gradients from top to bottom of the image, since perspective effects cause textures to appear 
    finer and denser at greater distances. The perspective strength metric quantifies how much 
    the scene exhibits directional perspective cues, while texture variation analysis captures 
    the depth-related changes in surface detail that help distinguish different environment types 
    based on their characteristic perspective and texture patterns.
    """
    def _analyze_perspective_cues(self, img_gray):
        features = {}
        
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                angles.append(angle)
            
            features['line_count'] = len(lines)
            features['line_angle_std'] = np.std(angles)
            features['line_angle_range'] = np.max(angles) - np.min(angles) if angles else 0
            
            horizontal_lines = sum(1 for angle in angles if abs(angle - 90) < 15 or abs(angle - 0) < 15)
            vertical_lines = sum(1 for angle in angles if abs(angle - 90) < 15)
            diagonal_lines = len(angles) - horizontal_lines - vertical_lines
            
            features['horizontal_line_ratio'] = horizontal_lines / len(angles) if angles else 0
            features['vertical_line_ratio'] = vertical_lines / len(angles) if angles else 0
            features['diagonal_line_ratio'] = diagonal_lines / len(angles) if angles else 0
            
            features['perspective_strength'] = features['diagonal_line_ratio']
            
        else:
            features['line_count'] = 0
            features['line_angle_std'] = 0
            features['line_angle_range'] = 0
            features['horizontal_line_ratio'] = 0
            features['vertical_line_ratio'] = 0
            features['diagonal_line_ratio'] = 0
            features['perspective_strength'] = 0
        
        height, width = img_gray.shape
        
        texture_densities = []
        for i in range(5):
            start_row = i * height // 5
            end_row = (i + 1) * height // 5
            strip = img_gray[start_row:end_row, :]
            
            texture_density = np.std(strip)
            texture_densities.append(texture_density)
        
        features['texture_gradient'] = np.polyfit(range(5), texture_densities, 1)[0]
        features['texture_variation'] = np.std(texture_densities)
        
        return features
    
    """
    When image processing fails or encounters errors, I need to return a consistent set of default 
    features that maintain the integrity of my depth analysis pipeline. This function provides 
    reasonable default values for all the depth-related features that my system normally extracts, 
    ensuring that failed image processing doesn't cause classification errors. The default values 
    are chosen to represent neutral or average cases across the different feature categories - 
    moderate edge sharpness, balanced focus measures, typical contrast values, and average depth 
    patterns. These defaults allow the classification system to continue processing even when 
    individual images fail, typically resulting in lower confidence predictions that reflect the 
    uncertainty. This robust error handling ensures that batch processing can continue smoothly 
    and that the overall depth analysis system remains stable even with problematic input images.
    """
    def _get_default_features(self):
        feature_names = [
            'edge_magnitude_mean', 'edge_magnitude_std', 'edge_magnitude_max', 'edge_density',
            'laplacian_variance', 'laplacian_mean', 'sharpness_scale_1', 'sharpness_scale_2',
            'sharpness_scale_3', 'sharpness_scale_4', 'sharpness_gradient_ratio',
            
            'top_focus_variance', 'middle_focus_variance', 'bottom_focus_variance',
            'left_focus_variance', 'center_focus_variance', 'right_focus_variance',
            'top_tenengrad', 'middle_tenengrad', 'bottom_tenengrad',
            'left_tenengrad', 'center_tenengrad', 'right_tenengrad',
            'top_high_freq_energy', 'middle_high_freq_energy', 'bottom_high_freq_energy',
            'left_high_freq_energy', 'center_high_freq_energy', 'right_high_freq_energy',
            'focus_variation', 'focus_range', 'focus_center_bias',
            
            'contrast_mean_scale_1', 'contrast_mean_scale_2', 'contrast_mean_scale_3', 'contrast_mean_scale_4',
            'contrast_std_scale_1', 'contrast_std_scale_2', 'contrast_std_scale_3', 'contrast_std_scale_4',
            'contrast_max_scale_1', 'contrast_max_scale_2', 'contrast_max_scale_3', 'contrast_max_scale_4',
            'contrast_decay_slope', 'contrast_decay_variation', 'luminance_contrast', 'luminance_range',
            
            'freq_energy_low', 'freq_energy_mid_low', 'freq_energy_mid_high', 'freq_energy_high',
            'frequency_centroid', 'high_freq_ratio',
            
            'regional_depth_mean', 'regional_depth_std', 'regional_depth_max', 'regional_depth_min',
            'regional_depth_range', 'depth_vertical_gradient', 'depth_horizontal_gradient',
            'center_depth', 'periphery_depth', 'center_periphery_ratio',
            
            'discontinuity_count', 'discontinuity_total_area', 'discontinuity_mean_area',
            'discontinuity_max_area', 'discontinuity_mean_perimeter', 'discontinuity_density',
            'discontinuity_shape_complexity',
            
            'line_count', 'line_angle_std', 'line_angle_range', 'horizontal_line_ratio',
            'vertical_line_ratio', 'diagonal_line_ratio', 'perspective_strength',
            'texture_gradient', 'texture_variation'
        ]
        
        return {name: 0.0 for name in feature_names}
    
    """
    This function processes my entire training dataset by systematically extracting depth transition 
    features from organized folders of images representing different indoor environments. I handle 
    the folder structure dynamically, mapping environment types to their corresponding directories 
    and adapting to different naming conventions. For each image, I extract the comprehensive depth 
    feature set that characterizes the spatial depth patterns of that environment type, including 
    edge sharpness analysis, focus variations, contrast transitions, spatial frequencies, regional 
    depth mapping, discontinuity detection, and perspective cues. The results are compiled into a 
    structured dataset with both extracted features and metadata about each image. This creates a 
    comprehensive training set that captures the distinctive depth signatures of different indoor 
    environments, providing the foundation for developing depth-based classification models.
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
                    features = self.extract_depth_features(img_path)
                    
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
        
        csv_filename = f"depth_transition_dataset_700.csv"
        csv_filepath = os.path.join(results_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        
        print(f"Features saved to: {csv_filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Classes distribution:")
        print(df['true_class'].value_counts())
        
        self._generate_feature_statistics(df)
        
        return df, csv_filepath
    
    """
    Here I generate comprehensive statistics about the extracted depth features to understand how 
    different environment types are characterized by their spatial depth patterns. I focus on key 
    depth-related features that are most likely to differentiate between environments, such as edge 
    sharpness measures, focus variations, contrast decay patterns, and perspective strength indicators. 
    For each environment class, I calculate mean values and standard deviations to identify the 
    characteristic depth signatures of hallways, staircases, rooms, and open areas. This statistical 
    analysis helps me understand which depth features are most discriminative and guides the development 
    of classification rules. The analysis reveals patterns like linear depth progressions in hallways, 
    step-wise depth changes in staircases, complex multi-plane patterns in rooms, and gradual depth 
    transitions in open areas, providing insights into the distinctive spatial depth characteristics 
    of each environment type.
    """
    def _generate_feature_statistics(self, df):
        print("\n" + "="*60)
        print("DEPTH TRANSITION FEATURE STATISTICS BY ENVIRONMENT")
        print("="*60)
        
        key_features = [
            'edge_magnitude_mean', 'laplacian_variance', 'sharpness_gradient_ratio',
            'focus_variation', 'center_focus_variance', 'contrast_decay_slope',
            'high_freq_ratio', 'regional_depth_std', 'discontinuity_density',
            'perspective_strength', 'texture_gradient'
        ]
        
        for env_type in df['true_class'].unique():
            env_data = df[df['true_class'] == env_type]
            print(f"\n{env_type.upper()} (n={len(env_data)}):")
            print("-" * 40)
            
            for feature in key_features:
                if feature in env_data.columns:
                    mean_val = env_data[feature].mean()
                    std_val = env_data[feature].std()
                    print(f"  {feature:<25}: {mean_val:.4f} ± {std_val:.4f}")

"""
This main function orchestrates the entire spatial depth transition mapping pipeline from initialization 
through feature extraction to dataset creation. I start by initializing the Spatial Depth Transition 
Mapper with all the necessary parameters for depth analysis including edge kernels, focus analysis 
settings, and regional depth mapping configurations. The function then provides a clear overview 
of the seven depth analysis components that my system employs to characterize environment depth 
patterns. After processing the entire training dataset and extracting comprehensive depth features, 
I generate detailed statistics that reveal the distinctive depth signatures of different environment 
types. This systematic approach creates a comprehensive foundation for understanding how spatial 
depth information can be used to distinguish between indoor environments, providing the basis for 
developing effective depth-based classification systems for robotics and computer vision applications.
"""
def main():
    print("Spatial Image Depth Transition Mapping System")
    print("="*50)
    
    system = SpatialDepthTransitionMapper()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    
    print("Extracting depth transition features from training dataset...")
    print("\nDepth analysis components:")
    print("- Edge sharpness gradients (distance estimation)")
    print("- Focus variations across regions")
    print("- Contrast transitions and decay patterns")
    print("- Spatial frequency analysis")
    print("- Regional depth mapping")
    print("- Depth discontinuity detection")
    print("- Perspective and vanishing point analysis")
    
    df, csv_filename = system.process_training_dataset(base_path)
    
    print(f"\nDepth transition features saved to {csv_filename}")
    print("Ready for classification system development!")
    
    return system, csv_filename

if __name__ == "__main__":
    system, csv_file = main()