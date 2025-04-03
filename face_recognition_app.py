import os
import cv2
import numpy as np
from sklearn.svm import SVC
import pickle
import dlib
import face_recognition  # Install with: pip install face_recognition

class EnhancedFaceRecognitionSystem:
    
    
    def detect_faces(self, image):
        """Detect faces in an image and return face locations"""
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use face_recognition library to detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        # Convert to (x, y, w, h) format for compatibility
        faces = [(left, top, right-left, bottom-top) for (top, right, bottom, left) in face_locations]
        
        return faces, face_locations
    
    def add_face_from_image(self, image_path, person_name):
        """Add a face from an image file"""
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found")
            return False
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False
            
        # Convert to RGB (required by face_recognition library)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if len(face_locations) == 0:
            print(f"Error: No face detected in {image_path}")
            return False
            
        # Get the largest face if multiple are detected
        if len(face_locations) > 1:
            print(f"Warning: Multiple faces detected in {image_path}. Using the largest face.")
            face_location = max(face_locations, key=lambda rect: (rect[2]-rect[0]) * (rect[3]-rect[1]))
        else:
            face_location = face_locations[0]
            
        # Compute face encoding
        face_encoding = face_recognition.face_encodings(rgb_image, [face_location])[0]
        
        # Save to model data
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(person_name)
        
        # Save the model
        self._save_model()
        
        # Save a copy of the face in the data directory
        if not os.path.exists(os.path.join(self.data_dir, person_name)):
            os.makedirs(os.path.join(self.data_dir, person_name))
            
        # Extract and save the face
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        
        # Generate unique filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = os.path.join(self.data_dir, person_name, f"{person_name}_{timestamp}.jpg")
        cv2.imwrite(face_filename, face_image)
        
        print(f"Face of {person_name} added from {image_path}")
        return True
    
    def add_face_from_webcam(self, person_name, camera_id=0):
        """Add a face captured from webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return False
        
        print("Capturing face from webcam...")
        print("Press 'c' to capture, 'q' to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
                
            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces, face_locations = self.detect_faces(frame)
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Capture Face', frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # 'c' to capture
            if key == ord('c') and len(faces) > 0:
                # Take the largest face
                largest_face_idx = np.argmax([w*h for (x, y, w, h) in faces])
                face_location = face_locations[largest_face_idx]
                
                # Convert to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Compute face encoding
                face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
                
                # Save to model data
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(person_name)
                
                # Save the model
                self._save_model()
                
                # Save a copy of the face
                if not os.path.exists(os.path.join(self.data_dir, person_name)):
                    os.makedirs(os.path.join(self.data_dir, person_name))
                    
                # Extract and save the face
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                
                # Generate unique filename
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                face_filename = os.path.join(self.data_dir, person_name, f"{person_name}_{timestamp}.jpg")
                cv2.imwrite(face_filename, face_image)
                
                print(f"Face of {person_name} captured and saved")
                
                # Close
                cap.release()
                cv2.destroyAllWindows()
                return True
            
            # 'q' to quit
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def _save_model(self):
        """Save the face recognition model to disk"""
        model_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved with {len(self.known_face_encodings)} face(s)")
    
    def run_recognition(self, camera_id=0, tolerance=0.6):
        """Run real-time face recognition using webcam"""
        if len(self.known_face_encodings) == 0:
            print("No faces in the model. Please add faces first.")
            return
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Starting face recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
                
            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB (required by face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                # Compute face encodings
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Process each detected face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance=tolerance
                    )
                    
                    # Calculate face distances (smaller = more similar)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    # Get the most likely match
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            # Green rectangle for recognized face
                            color = (0, 255, 0)
                        else:
                            name = "Unknown"
                            # Red rectangle for unknown face
                            color = (0, 0, 255)
                            
                        # Draw rectangle around face
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw label with name and confidence
                        label = f"{name}: {confidence:.2f}"
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        cv2.putText(
                            frame, 
                            label, 
                            (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            0.6, 
                            (255, 255, 255), 
                            1
                        )
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Wait for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def adjust_recognition_sensitivity(self, tolerance=0.6):
        """Adjust the recognition sensitivity"""
        print(f"Recognition sensitivity set to {tolerance}")
        print("Lower values = stricter matching, higher values = more permissive")
        return tolerance

def main():
    """Main function to demonstrate the face recognition system"""
    print("Enhanced Face Recognition System")
    print("===============================")
    print("Optimized for recognition with few training images")
    
    # Initialize the system
    system = EnhancedFaceRecognitionSystem()
    tolerance = 0.6  # Default recognition sensitivity
    
    while True:
        print("\nOptions:")
        print("1. Add face from image file")
        print("2. Add face from webcam")
        print("3. Run face recognition")
        print("4. Adjust recognition sensitivity")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            person_name = input("Enter the person's name: ")
            image_path = input("Enter the path to the image file: ")
            system.add_face_from_image(image_path, person_name)
            
        elif choice == '2':
            person_name = input("Enter the person's name: ")
            system.add_face_from_webcam(person_name)
            
        elif choice == '3':
            system.run_recognition(tolerance=tolerance)
            
        elif choice == '4':
            new_tolerance = float(input("Enter recognition sensitivity (0.4-0.8, lower=stricter): ") or "0.6")
            tolerance = system.adjust_recognition_sensitivity(new_tolerance)
            
        elif choice == '5':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()