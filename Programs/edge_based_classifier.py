


#  "Statistical Edge Orientation Analysis and Edge Pattern Recognition" (SEOAEPR) ---> Implementation


import os
import datetime
import cv2
import numpy as np

import pandas as pd
from datetime import datetime

class FixedEdgeOrientationClassifier:
    """
    This Fixed Edge Orientation Classifier implements a rule-based approach for classifying indoor 
    environments using edge pattern analysis. I designed this system based on statistical insights 
    from actual data analysis, focusing on simple but effective decision tree logic rather than 
    complex scoring mechanisms. The classifier uses key discriminating features like horizontal and 
    vertical edge ratios, diagonal patterns, and orientation entropy to distinguish between hallways, 
    staircases, rooms, and open areas. My approach prioritizes interpretability and reliability by 
    using fixed thresholds derived from empirical analysis of edge orientation patterns in different 
    indoor environments.
    """
    
    def __init__(self):
        self.rules = {
            'open_area_detection': {
                'combined_hv_threshold': 0.20,
                'diagonal_threshold': 0.80,
                'min_confidence': 0.75
            },
            
            'indoor_classification': {
                'entropy_low_threshold': 2.35,
                'entropy_high_threshold': 2.45,
                'vh_ratio_threshold': 0.05,
                'pattern_uniformity_threshold': 0.76
            }
        }
    
    """
    In this function, I extract edges from an input image using Canny edge detection with standardized 
    preprocessing to ensure consistent results across all images. I resize larger images to a maximum 
    width of 800 pixels to maintain processing efficiency while preserving the essential edge patterns. 
    The preprocessing includes converting to grayscale and applying Gaussian blur to reduce noise that 
    could interfere with edge detection. I use fixed Canny thresholds that have proven effective across 
    various indoor environments. This consistent preprocessing pipeline ensures that the edge patterns 
    I extract are comparable across different images and environments, which is crucial for reliable 
    classification using my rule-based approach.
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
    Here I calculate the orientation angles of detected edges by computing gradients in both x and y 
    directions using Sobel operators. The gradients reveal how pixel intensities change across the 
    image, allowing me to determine the direction perpendicular to each edge. I convert the gradient 
    ratios to actual angles using arctangent, then normalize them to a 0-180 degree range for consistency. 
    To focus on the most significant structural elements, I filter out weak edges by keeping only those 
    with magnitudes above the 70th percentile. This selective approach ensures that I'm analyzing the 
    dominant edge patterns that truly characterize the environment, rather than getting distracted by 
    minor texture details or noise artifacts.
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
    This function extracts the essential features that I've identified as most discriminative for 
    environment classification. I calculate weighted ratios for horizontal, vertical, and diagonal 
    edges, where the weights are based on edge magnitude to emphasize stronger structural elements. 
    Beyond basic orientation ratios, I compute combined metrics like the total horizontal plus vertical 
    ratio and the vertical-to-horizontal ratio that have proven particularly effective for distinguishing 
    environment types. I also include orientation entropy to measure the randomness of edge directions 
    and pattern uniformity to quantify how evenly distributed the orientations are. This focused 
    feature set represents the core characteristics that reliably differentiate between different 
    indoor environments in my classification system.
    """
    def extract_features(self, orientations, magnitudes, img_shape):
        if orientations is None or len(orientations) == 0:
            return self._get_default_features()
        
        total_magnitude = np.sum(magnitudes)
        if total_magnitude == 0:
            return self._get_default_features()
            
        weights = magnitudes / total_magnitude
        
        horizontal_mask = ((orientations <= 22.5) | (orientations >= 157.5))
        vertical_mask = ((orientations >= 67.5) & (orientations <= 112.5))
        diagonal_45_mask = ((orientations >= 22.5) & (orientations <= 67.5))
        diagonal_135_mask = ((orientations >= 112.5) & (orientations <= 157.5))
        
        features = {
            'horizontal_ratio': np.sum(weights[horizontal_mask]),
            'vertical_ratio': np.sum(weights[vertical_mask]),
            'diagonal_45_ratio': np.sum(weights[diagonal_45_mask]),
            'diagonal_135_ratio': np.sum(weights[diagonal_135_mask]),
        }
        
        features['total_diagonal_ratio'] = features['diagonal_45_ratio'] + features['diagonal_135_ratio']
        features['combined_hv_ratio'] = features['horizontal_ratio'] + features['vertical_ratio']
        features['vertical_horizontal_ratio'] = features['vertical_ratio'] / (features['horizontal_ratio'] + 1e-6)
        
        features['orientation_entropy'] = self._calculate_entropy(orientations)
        
        ratios_array = np.array([features['horizontal_ratio'], features['vertical_ratio'],
                               features['diagonal_45_ratio'], features['diagonal_135_ratio']])
        features['pattern_uniformity'] = 1.0 - np.std(ratios_array)
        
        return features
    
    """
    When edge detection or orientation calculation fails, I need to return a consistent set of default 
    features that won't break my classification pipeline. This function provides reasonable default 
    values that represent a neutral or average case - not strongly horizontal, vertical, or diagonal, 
    with moderate entropy and pattern uniformity. These defaults ensure that failed image processing 
    doesn't cause classification errors, while the values are chosen to typically result in lower 
    confidence predictions. This safety mechanism allows my batch processing to continue smoothly 
    even when individual images fail to process correctly, maintaining the robustness of the overall 
    system.
    """
    def _get_default_features(self):
        return {
            'horizontal_ratio': 0.0, 'vertical_ratio': 0.0, 'diagonal_45_ratio': 0.0,
            'diagonal_135_ratio': 0.0, 'total_diagonal_ratio': 0.0, 'combined_hv_ratio': 0.0,
            'vertical_horizontal_ratio': 0.0, 'orientation_entropy': 2.0, 'pattern_uniformity': 0.75
        }
    
    """
    In this function, I calculate the entropy of the edge orientation distribution to measure how 
    randomly or uniformly the edge directions are distributed across the image. I create a histogram 
    of orientations divided into 18 bins across the 0-180 degree range, then compute the Shannon 
    entropy from the normalized probability distribution. Higher entropy values indicate that edge 
    orientations are spread out evenly with no dominant direction, typically seen in complex or 
    cluttered environments like rooms. Lower entropy suggests that certain orientations dominate, 
    which is characteristic of more structured environments like hallways or staircases. This entropy 
    measure serves as a key discriminator in my classification rules.
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
    This is the core classification function that implements my rule-based decision tree for environment 
    recognition. I start by extracting features from the input image, then apply a two-stage classification 
    process. First, I check for open areas using the combined horizontal-vertical ratio and diagonal 
    threshold, as these environments have distinctly different edge patterns. For indoor environments, 
    I use entropy as the primary discriminator, with low entropy indicating staircases and high entropy 
    suggesting rooms. For medium entropy cases, I apply additional rules using vertical-horizontal 
    ratios and pattern uniformity. This hierarchical approach reflects the natural separability I 
    discovered in the data and provides interpretable classification decisions with confidence estimates.
    """
    def classify_image(self, image_path, debug=False):
        edges, img_shape = self.extract_edges(image_path)
        if edges is None:
            return {'error': 'Failed to process image', 'predicted_class': 'unknown', 'confidence': 0.0}
        
        orientations, magnitudes = self.calculate_edge_orientations(edges)
        features = self.extract_features(orientations, magnitudes, img_shape)
        
        decision_path = []
        
        combined_hv = features['combined_hv_ratio']
        total_diagonal = features['total_diagonal_ratio']
        
        if combined_hv > self.rules['open_area_detection']['combined_hv_threshold'] or \
           total_diagonal < self.rules['open_area_detection']['diagonal_threshold']:
            
            decision_path.append(f"Open area detected: H+V={combined_hv:.3f} > 0.20 OR diagonal={total_diagonal:.3f} < 0.80")
            
            confidence = 0.75
            if combined_hv > 0.30:
                confidence += 0.15
            if total_diagonal < 0.70:
                confidence += 0.10
            
            return {
                'predicted_class': 'open_area',
                'confidence': min(1.0, confidence),
                'features': features,
                'decision_path': decision_path if debug else None,
                'image_path': image_path
            }
        
        entropy = features['orientation_entropy']
        vh_ratio = features['vertical_horizontal_ratio']
        pattern_uniformity = features['pattern_uniformity']
        
        decision_path.append(f"Indoor environment detected: H+V={combined_hv:.3f} <= 0.20 AND diagonal={total_diagonal:.3f} >= 0.80")
        
        if entropy < self.rules['indoor_classification']['entropy_low_threshold']:
            decision_path.append(f"Low entropy detected: {entropy:.3f} < 2.35 -> likely STAIRCASE")
            confidence = 0.70 + (2.35 - entropy) * 0.15
            predicted_class = 'staircase'
            
        elif entropy > self.rules['indoor_classification']['entropy_high_threshold']:
            decision_path.append(f"High entropy detected: {entropy:.3f} > 2.45 -> likely ROOM")
            confidence = 0.70 + (entropy - 2.45) * 0.20
            predicted_class = 'room'
            
        else:
            if vh_ratio < self.rules['indoor_classification']['vh_ratio_threshold']:
                decision_path.append(f"Medium entropy + low VH ratio: {vh_ratio:.3f} < 0.05 -> likely HALLWAY")
                confidence = 0.65
                predicted_class = 'hallway'
            else:
                if pattern_uniformity > self.rules['indoor_classification']['pattern_uniformity_threshold']:
                    decision_path.append(f"Medium entropy + high uniformity -> likely ROOM")
                    confidence = 0.60
                    predicted_class = 'room'
                else:
                    decision_path.append(f"Medium entropy + medium VH ratio -> likely STAIRCASE")
                    confidence = 0.60
                    predicted_class = 'staircase'
        
        return {
            'predicted_class': predicted_class,
            'confidence': min(1.0, confidence),
            'features': features,
            'decision_path': decision_path if debug else None,
            'image_path': image_path
        }
    
    """
    This function handles batch processing of multiple images by applying my classification method 
    to each image in sequence. I include progress tracking to monitor the processing of large image 
    sets, which is particularly useful when evaluating the classifier on test datasets. The function 
    maintains consistency by using the same classification parameters for all images and optionally 
    enables debug mode to capture decision paths for analysis. This batch processing capability is 
    essential for systematic evaluation and performance analysis of my classification system, allowing 
    me to efficiently process entire folders of test images and generate comprehensive results for 
    accuracy assessment and error analysis.
    """
    def classify_images(self, image_paths, debug=False):
        results = []
        for i, image_path in enumerate(image_paths):
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
            
            result = self.classify_image(image_path, debug=debug)
            results.append(result)
        
        return results
    
    """
    This function provides a comprehensive testing framework for evaluating my classifier's performance 
    on images with known ground truth labels. I process each test image and compare the predicted 
    class against the true class, tracking accuracy and providing detailed feedback for each prediction. 
    The debug mode captures the decision path, allowing me to understand exactly how each classification 
    decision was made. This detailed analysis helps me identify patterns in correct and incorrect 
    predictions, understand the strengths and weaknesses of my rule-based approach, and refine my 
    classification thresholds. The function provides both individual image results and overall accuracy 
    metrics, giving me a complete picture of classifier performance.
    """
    def test_classifier(self, test_images_info):
        print("Testing Fixed Edge Orientation Classifier")
        print("="*50)
        
        results = []
        correct = 0
        
        for img_info in test_images_info:
            result = self.classify_image(img_info['path'], debug=True)
            result['true_class'] = img_info['true_class']
            results.append(result)
            
            is_correct = result['predicted_class'] == img_info['true_class']
            if is_correct:
                correct += 1
            
            print(f"\nImage: {os.path.basename(img_info['path'])}")
            print(f"True: {img_info['true_class']}, Predicted: {result['predicted_class']}, Confidence: {result['confidence']:.3f}")
            print(f"Correct: {'✓' if is_correct else '✗'}")
            
            if result.get('decision_path'):
                print("Decision path:")
                for step in result['decision_path']:
                    print(f"  - {step}")
        
        accuracy = correct / len(test_images_info)
        print(f"\nOverall Accuracy: {accuracy:.3f} ({correct}/{len(test_images_info)})")
        
        return results
    
    """
    This function performs comprehensive evaluation of my classifier using a CSV file containing image 
    paths and their corresponding true class labels. I process all images, calculate overall accuracy, 
    and generate a detailed confusion matrix to understand classification patterns across different 
    environment types. The confusion matrix reveals which classes are frequently confused with each 
    other, helping me identify systematic errors in my classification rules. I also compute class-wise 
    accuracy to understand how well the classifier performs for each individual environment type. This 
    comprehensive evaluation approach provides the statistical foundation I need to assess classifier 
    performance and guide future improvements to my rule-based decision system.
    """
    def evaluate_on_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        
        print(f"Evaluating on {len(df)} images from CSV...")
        
        results = []
        confusion_matrix = {}
        
        for _, row in df.iterrows():
            result = self.classify_image(row['image_path'])
            result['true_class'] = row['true_class']
            results.append(result)
        
        correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
        accuracy = correct / len(results)
        
        classes = list(set(df['true_class']))
        for true_class in classes:
            confusion_matrix[true_class] = {}
            class_results = [r for r in results if r['true_class'] == true_class]
            
            for pred_class in classes:
                count = sum(1 for r in class_results if r['predicted_class'] == pred_class)
                confusion_matrix[true_class][pred_class] = count
        
        print(f"\nClassification Results:")
        print(f"Overall Accuracy: {accuracy:.3f} ({correct}/{len(results)})")
        
        print(f"\nConfusion Matrix:")
        print(f"{'True\\Pred':<12}", end="")
        for pred_class in sorted(classes):
            print(f"{pred_class:<12}", end="")
        print()
        
        for true_class in sorted(classes):
            print(f"{true_class:<12}", end="")
            for pred_class in sorted(classes):
                count = confusion_matrix[true_class].get(pred_class, 0)
                print(f"{count:<12}", end="")
            print()
        
        print(f"\nClass-wise Accuracy:")
        for true_class in sorted(classes):
            class_results = [r for r in results if r['true_class'] == true_class]
            class_correct = sum(1 for r in class_results if r['predicted_class'] == true_class)
            class_accuracy = class_correct / len(class_results) if class_results else 0
            print(f"  {true_class:<12}: {class_accuracy:.3f} ({class_correct}/{len(class_results)})")
        
        return {
            'overall_accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'detailed_results': results
        }

       
def main():
    classifier = FixedEdgeOrientationClassifier()
    return classifier


def test_on_separate_folders():
    classifier = FixedEdgeOrientationClassifier()
    
    base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
    test_folders = {
        'hallway': 'hallway_test_photos',
        'staircase': 'staircase_test_photos', 
        'room': 'room_test_photos',
        'open_area': 'openarea_test_photos'
    }
    
    print("Testing Edge Orientation Classifier on Test Folders")
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

if __name__ == "__main__":
    print("Fixed Edge Orientation Classifier")
    print("="*40)
    
    classifier = FixedEdgeOrientationClassifier()
    results = test_on_separate_folders()
    
    
    
# Overall Results:
# hallway: 0.364 (20/55)
# staircase: 0.519 (27/52)
# room: 0.695 (41/59)
# open_area: 0.677 (44/65)
# Overall: 0.571 (132/231)