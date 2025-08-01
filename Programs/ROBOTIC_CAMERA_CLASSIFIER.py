

### INTEGRATING THE KERNEL METHOD (HYBRID APPROACH) TO WORK ON REAL-TIME ROBOTIC SYSTEMS, INSTEAD OF THE TEST IMAGES ON MY LAPTOP

import numpy as np
import cv2
import os
import time
from datetime import datetime 
import warnings
warnings.filterwarnings('ignore')

class RealTimeCameraClassifier:
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
        
        print(f"\nReal-time classifier initialized with {len(self.classifiers)}/5 methods")
        

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
        

        self.save_directory = "captured_images"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"Created directory: {self.save_directory}")
    
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
            
            # Normalize the prediction
            normalized_prediction = self.normalize_environment_name(prediction)
            return normalized_prediction, confidence
            
        except Exception as e:
            print(f"    Error in {method_name} method: {e}")
            return 'unknown', 0.0
    
    def classify_captured_image(self, image_path, show_details=True):

        if show_details:
            print(f"\n{'='*60}")
            print(f"ANALYZING CAPTURED IMAGE: {os.path.basename(image_path)}")
            print(f"{'='*60}")
        
       
        method_predictions = {}
        votes = {}
        
        for method_name, classifier in self.classifiers.items():
            if show_details:
                print(f"Running {method_name} method...")
            
            prediction, confidence = self.get_prediction_from_method(method_name, classifier, image_path)
            
            method_predictions[method_name] = {
                'prediction': prediction,
                'confidence': confidence
            }
            
            
            if prediction != 'unknown':
                votes[prediction] = votes.get(prediction, 0) + 1
            
            if show_details:
                print(f"  {method_name.upper()}: {prediction} (confidence: {confidence:.3f})")
        
        # Determine winner by majority vote
        if votes:
            winner = max(votes, key=votes.get)
            winner_votes = votes[winner]
            total_votes = sum(votes.values())
            confidence = winner_votes / len(self.classifiers)  # Percentage of methods that agreed
        else:
            winner = 'unknown'
            winner_votes = 0
            total_votes = 0
            confidence = 0.0
        
        if show_details:
            print(f"\n{'='*60}")
            print(f"VOTING RESULTS:")
            print(f"{'='*60}")
            for env, vote_count in votes.items():
                percentage = vote_count / len(self.classifiers) * 100
                print(f"  {env.upper()}: {vote_count}/{len(self.classifiers)} votes ({percentage:.1f}%)")
            
            print(f"\n WINNER: {winner.upper()}")
            print(f"   Consensus: {winner_votes}/{len(self.classifiers)} methods agreed ({confidence:.1%})")
            print(f"{'='*60}")
        
        return {
            'predicted_class': winner,
            'confidence': confidence,
            'method_predictions': method_predictions,
            'vote_breakdown': votes,
            'winner_votes': winner_votes,
            'total_methods': len(self.classifiers)
        }
    
    def capture_and_classify_single_image(self):

        # Initialize the camera (laptop, webcam, or desktop)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        # Set camera resolution 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("CAMERA READY - SINGLE IMAGE CAPTURE")
        print("="*60)
        print("Instructions:")
        print("  - Position camera towards the environment you want to classify")
        print("  - Press SPACE to capture image")
        print("  - Press 'q' to quit")
        print("="*60)
        
        while True:
            
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Display the resulting frame
            cv2.putText(frame, "Press SPACE to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Environment Classifier Camera', frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space key to capture
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_{timestamp}.jpg"
                filepath = os.path.join(self.save_directory, filename)
                
                # Save the captured image
                cv2.imwrite(filepath, frame)
                print(f"\n✓ Image captured and saved: {filepath}")
                
                # Classify the captured image
                result = self.classify_captured_image(filepath, show_details=True)
                
                # Show result on image
                result_frame = frame.copy()
                prediction = result['predicted_class']
                confidence = result['confidence']
                
                # Add text overlay with result
                cv2.putText(result_frame, f"Environment: {prediction.upper()}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Confidence: {confidence:.1%}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, "Press SPACE for new capture, 'q' to quit", 
                           (10, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display result for 3 seconds
                cv2.imshow('Classification Result', result_frame)
                cv2.waitKey(3000)  # Show for 3 seconds
                cv2.destroyWindow('Classification Result')
                
                break
                
            elif key == ord('q'):  # Quit
                break
        
        # Release everything
        cap.release()
        cv2.destroyAllWindows()
        
        return result if 'result' in locals() else None
    
    def start_continuous_classification(self):

        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("CONTINUOUS ENVIRONMENT CLASSIFICATION")
        print("="*60)
        print("Instructions:")
        print("  - Camera will auto-capture and classify every 5 seconds")
        print("  - Move camera to different environments to test")
        print("  - Press 'q' to quit")
        print("="*60)
        
        last_capture_time = 0
        capture_interval = 5  # seconds
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            current_time = time.time()
            
            # Add countdown overlay
            time_until_capture = capture_interval - (current_time - last_capture_time)
            if time_until_capture > 0:
                cv2.putText(frame, f"Next capture in: {int(time_until_capture)}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Capturing and analyzing...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Continuous Environment Classifier', frame)
            
            # Auto-capture every 5 seconds
            if current_time - last_capture_time >= capture_interval:
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"auto_captured_{timestamp}.jpg"
                filepath = os.path.join(self.save_directory, filename)
                
                # Save the captured image
                cv2.imwrite(filepath, frame)
                print(f"\n📸 Auto-captured: {filepath}")
                
                # Classify the captured image
                result = self.classify_captured_image(filepath, show_details=True)
                
                last_capture_time = current_time
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
  
        cap.release()
        cv2.destroyAllWindows()

def main():    
    print("REAL-TIME CAMERA ENVIRONMENT CLASSIFIER")
    print("For Robotic Systems Lab Testing")
    print("="*50)
    
    # Create the classifier
    classifier = RealTimeCameraClassifier()
    
    if len(classifier.classifiers) == 0:
        print("\nERROR: No classifiers were loaded successfully!")
        print("Please check that all your classifier files are in the same directory:")
        print("- edge_based_classifier.py")
        print("- RCVDA_classifier.py") 
        print("- SIDTM_classifier.py")
        print("- VPIPCD_classifier.py")
        print("- IPALDC_CLASSIFIER.py")
        return
    
    ## Choose mode (allow the user to choose what they want to do)
    print(f"\nClassifier ready with {len(classifier.classifiers)} methods!")
    print("\nSelect Mode:")
    print("1. Single Image Capture (press SPACE to capture)")
    print("2. Continuous Classification (auto-capture every 5 seconds)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print("\nStarting Single Image Capture Mode...")
            result = classifier.capture_and_classify_single_image()
            if result:
                print(f"\nFinal Result: {result['predicted_class']} (confidence: {result['confidence']:.1%})")
        
        elif choice == '2':
            print("\nStarting Continuous Classification Mode...")
            classifier.start_continuous_classification()
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice! Please try again entering 1, 2, or 3.")

if __name__ == "__main__":
    main()

