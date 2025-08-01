
### KERNEL CLASSIFIER ---> HYBRID APPROACH THAT COMBIONES THE PREVIOUS TECHINQUES,but runs on test images.

import numpy as np
import cv2
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class CompleteHybridClassifier:
    def __init__(self):

        self.classifiers = {}
        self.available_methods = []
        
        # Import Method 1: Edge-based classifier
        try:
            from edge_based_classifier import FixedEdgeOrientationClassifier
            self.classifiers['edge'] = FixedEdgeOrientationClassifier()
            self.available_methods.append('Edge Orientation Analysis')
            print("✓ Method 1: Edge-based classifier loaded successfully")
        except Exception as e:
            print(f"✗ Method 1: Edge-based classifier failed to load: {e}")
        
        # Import Method 2: Color-based classifier
        try:
            from RCVDA_classifier import RegionalColorDistributionClassifier
            self.classifiers['color'] = RegionalColorDistributionClassifier()
            self.available_methods.append('Regional Color Distribution')
            print("✓ Method 2: Color-based classifier loaded successfully")
        except Exception as e:
            print(f"✗ Method 2: Color-based classifier failed to load: {e}")
        
        # Import Method 3: Depth-based classifier
        try:
            from SIDTM_classifier import SpatialDepthTransitionClassifier
            self.classifiers['depth'] = SpatialDepthTransitionClassifier()
            self.available_methods.append('Spatial Depth Transition')
            print("✓ Method 3: Depth-based classifier loaded successfully")
        except Exception as e:
            print(f"✗ Method 3: Depth-based classifier failed to load: {e}")
        
        # Import Method 4: Vanishing Point classifier
        try:
            from VPIPCD_classifier import EnhancedVPIPCDClassifier
            self.classifiers['vp'] = EnhancedVPIPCDClassifier()
            self.available_methods.append('Vanishing Point Detection')
            print("✓ Method 4: VP-based classifier loaded successfully")
        except Exception as e:
            print(f"✗ Method 4: VP-based classifier failed to load: {e}")
        
        # Import Method 5: Lighting-based classifier
        try:
            from IPALDC_CLASSIFIER import IPALDCClassifier
            self.classifiers['lighting'] = IPALDCClassifier()
            self.available_methods.append('Illumination Pattern Analysis')
            print("✓ Method 5: Lighting-based classifier loaded successfully")
        except Exception as e:
            print(f"✗ Method 5: Lighting-based classifier failed to load: {e}")
        
        print(f"\nHybrid classifier initialized with {len(self.classifiers)}/5 methods")
        
        # Environment name standardization
        self.environment_mapping = {
            'openarea': 'open_area',
            'open_area': 'open_area',
            'outdoor': 'open_area',
            'hallway': 'hallway',
            'corridor': 'hallway',
            'staircase': 'staircase',
            'stairs': 'staircase',
            'room': 'room',
            'indoor': 'room',
            'unknown': 'unknown'
        }
    
    def normalize_environment_name(self, env_name):
        if env_name is None:
            return 'unknown'
        
        env_name = str(env_name).lower().strip()
        return self.environment_mapping.get(env_name, env_name)
    
    def get_prediction_from_method(self, method_name, classifier, image_path):
        try:
            if method_name == 'edge':
                result = classifier.classify_image(image_path, debug=False)
                prediction = result.get('predicted_class', 'unknown')
                confidence = result.get('confidence', 0.0)
                
            elif method_name == 'color':
                result = classifier.classify_image(image_path, debug=False)
                prediction = result.get('predicted_class', 'unknown')
                confidence = result.get('confidence', 0.0)
                
            elif method_name == 'depth':
                result = classifier.classify_image(image_path, debug=False)
                prediction = result.get('predicted_class', 'unknown')
                confidence = result.get('confidence', 0.0)
                
            elif method_name == 'vp':
                prediction = classifier.classify_image(image_path)
                confidence = 0.7 if prediction != 'unknown' else 0.0
                
            elif method_name == 'lighting':
                prediction = classifier.classify_image(image_path)
                confidence = 0.7 if prediction != 'unknown' else 0.0
                
            else:
                return 'unknown', 0.0
            
            normalized_prediction = self.normalize_environment_name(prediction)
            return normalized_prediction, confidence
            
        except Exception as e:
            print(f"    Error in {method_name} method: {e}")
            return 'unknown', 0.0
    
    def classify_single_image(self, image_path, debug=False):
        if debug:
            print(f"\nAnalyzing: {os.path.basename(image_path)}")
        
        # Now we will get predictions from all methods ...
        method_predictions = {}
        votes = {}
        
        for method_name, classifier in self.classifiers.items():
            prediction, confidence = self.get_prediction_from_method(method_name, classifier, image_path)
            
            method_predictions[method_name] = {
                'prediction': prediction,
                'confidence': confidence
            }
            
            # Important ----> Count votes (exclude unknown predictions)
            if prediction != 'unknown':
                votes[prediction] = votes.get(prediction, 0) + 1
            
            if debug:
                print(f"  {method_name}: {prediction} (conf: {confidence:.3f})")
        
        # We would then determine "winner" environment by majority vote ...
        if votes:
            winner = max(votes, key=votes.get)
            winner_votes = votes[winner]
            total_votes = sum(votes.values())
            confidence = winner_votes / len(self.classifiers)  
        else:
            winner = 'unknown'
            winner_votes = 0
            total_votes = 0
            confidence = 0.0
        
        if debug:
            print(f"  Votes: {votes}")
            print(f"  Winner: {winner} ({winner_votes}/{len(self.classifiers)} methods)")
        
        return {
            'predicted_class': winner,
            'confidence': confidence,
            'method_predictions': method_predictions,
            'vote_breakdown': votes,
            'winner_votes': winner_votes,
            'total_methods': len(self.classifiers)
        }
    
    def test_complete_dataset(self):
        
        base_path = "/Users/shahmeer/Desktop/Robotics Vision Summer 2025 Research/photos"
        test_folders = {
            'hallway': 'hallway_test_photos',
            'staircase': 'staircase_test_photos', 
            'room': 'room_test_photos',
            'open_area': 'openarea_test_photos'
        }
        
        print("\n" + "="*80)
        print("COMPLETE HYBRID CLASSIFIER TEST - ALL 231 IMAGES")
        print("Testing 5 Methods with Majority Voting")
        print("="*80)
        
        total_images = 0
        total_correct = 0
        environment_stats = {}
        all_results = []
        method_agreement = {}
        
        for true_env, folder_name in test_folders.items():
            folder_path = os.path.join(base_path, folder_name)
            
            print(f"\n{'-'*60}")
            print(f"TESTING {true_env.upper()} ENVIRONMENT")
            print(f"Folder: {folder_name}")
            print(f"{'-'*60}")
            
            if not os.path.exists(folder_path):
                print(f"ERROR: Folder not found: {folder_path}")
                continue
        
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(image_extensions)]
            
            if not image_files:
                print(f"No images found in {folder_name}")
                continue
            
            print(f"Found {len(image_files)} images")
            
            environment_stats[true_env] = {
                'total': len(image_files),
                'correct': 0,
                'predictions': {},
                'detailed_results': []
            }
            
     
            correct_in_folder = 0
            
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(folder_path, img_file)
                
                # Print progress every 10 images
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    print(f"  Processing: {i+1}/{len(image_files)} images...")
                
                # Classify the image
                result = self.classify_single_image(img_path, debug=False)
                
                predicted_env = result['predicted_class']
                is_correct = (predicted_env == true_env)
                
                if is_correct:
                    correct_in_folder += 1
                    total_correct += 1
                
                total_images += 1
                
                if predicted_env not in environment_stats[true_env]['predictions']:
                    environment_stats[true_env]['predictions'][predicted_env] = 0
                environment_stats[true_env]['predictions'][predicted_env] += 1
                
    
                detailed_result = {
                    'image_file': img_file,
                    'true_class': true_env,
                    'predicted_class': predicted_env,
                    'is_correct': is_correct,
                    'confidence': result['confidence'],
                    'vote_breakdown': result['vote_breakdown'],
                    'method_predictions': result['method_predictions']
                }
                
                environment_stats[true_env]['detailed_results'].append(detailed_result)
                all_results.append(detailed_result)
                
                # Track method agreement
                methods_that_agreed = result['winner_votes']
                if methods_that_agreed not in method_agreement:
                    method_agreement[methods_that_agreed] = 0
                method_agreement[methods_that_agreed] += 1
            
            # Calculate folder accuracy
            environment_stats[true_env]['correct'] = correct_in_folder
            folder_accuracy = correct_in_folder / len(image_files) if len(image_files) > 0 else 0
            
            print(f"\n{true_env.upper()} RESULTS:")
            print(f"  Correct: {correct_in_folder}/{len(image_files)} ({folder_accuracy:.1%})")
            print(f"  Predictions breakdown:")
            for pred_env, count in environment_stats[true_env]['predictions'].items():
                percentage = count / len(image_files) * 100
                print(f"    {pred_env}: {count} images ({percentage:.1f}%)")
        
        # Print overall results
        print(f"\n{'='*80}")
        print("OVERALL HYBRID CLASSIFIER RESULTS")
        print(f"{'='*80}")
        
        overall_accuracy = total_correct / total_images if total_images > 0 else 0
        print(f"Overall Accuracy: {total_correct}/{total_images} ({overall_accuracy:.1%})")
        
        print(f"\nPer-Environment Accuracy:")
        for env, stats in environment_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {env}: {stats['correct']}/{stats['total']} ({accuracy:.1%})")
        
        # Method agreement analysis
        print(f"\nMethod Agreement Analysis:")
        for num_methods, count in sorted(method_agreement.items()):
            percentage = count / total_images * 100
            print(f"  {num_methods}/{len(self.classifiers)} methods agreed: {count} images ({percentage:.1f}%)")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        print(f"{'True\\Predicted':<15}", end="")
        all_environments = ['hallway', 'staircase', 'room', 'open_area', 'unknown']
        for env in all_environments:
            print(f"{env:<12}", end="")
        print()
        
        for true_env in ['hallway', 'staircase', 'room', 'open_area']:
            print(f"{true_env:<15}", end="")
            for pred_env in all_environments:
                count = environment_stats[true_env]['predictions'].get(pred_env, 0)
                print(f"{count:<12}", end="")
            print()
        
        print(f"\n{'='*80}")
        print("SUMMARY:")
        print(f"✓ Tested {total_images} total images")
        print(f"✓ Used {len(self.classifiers)} classification methods")
        print(f"✓ Achieved {overall_accuracy:.1%} overall accuracy")
        print(f"✓ Voting ensemble approach - no machine learning required")
        print(f"{'='*80}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_images': total_images,
            'environment_stats': environment_stats,
            'method_agreement': method_agreement,
            'all_results': all_results
        }

def main():
    
    print("INITIALIZING COMPLETE HYBRID CLASSIFIER")
    print("="*50)

    classifier = CompleteHybridClassifier()
    
    if len(classifier.classifiers) == 0:
        print("\nERROR: No classifiers were loaded successfully!")
        print("Please check that all your classifier files are in the same directory:")
        print("- edge_based_classifier.py")
        print("- RCVDA_classifier.py") 
        print("- SIDTM_classifier.py")
        print("- VPIPCD_classifier.py")
        print("- IPALDC_CLASSIFIER.py")
        return None
    

    results = classifier.test_complete_dataset()
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()
    
    
    
    
    