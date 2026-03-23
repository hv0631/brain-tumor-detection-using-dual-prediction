import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import matplotlib.pyplot as plt 

class DualPredictionBrainTumorDetector:
    def __init__(self, image_folder_path):
        """
        Initialize the Dual Prediction Brain Tumor Detection system
        Args:
            image_folder_path (str): Path to the folder containing brain scan images
        """
        self.image_folder_path = image_folder_path
        self.csv_data = None
        self.csv_model = None
        self.image_model = None
        self.scaler = StandardScaler()
        
        # Data splits
        self.X_train_csv = None
        self.X_test_csv = None
        self.y_train = None
        self.y_test = None
        self.X_train_images = None
        self.X_test_images = None
        
        # Predictions
        self.csv_predictions = None
        self.image_predictions = None
        self.csv_probabilities = None
        self.image_probabilities = None
        
        # Test data paths for image model
        self.test_image_paths = None
        
        self.feature_columns = [
            'area', 'perimeter', 'compactness', 'contrast', 'energy', 
            'homogeneity', 'entropy', 'mean_intensity', 'std_intensity'
        ]
        
        # Define class mappings
        self.class_mapping = {
            'glioma': 1,
            'meningioma': 1,
            'pituitary': 1,
            'no-tumor': 0,
            'notumor': 0,
            'normal': 0,
            'benign': 0
        }
    
    def find_all_images(self):
        """Find all images in the folder structure"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        all_images = []
        
        print("Scanning for images...")
        
        # Check for subfolders
        subfolders = []
        for item in os.listdir(self.image_folder_path):
            item_path = os.path.join(self.image_folder_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
        
        if subfolders:
            print(f"Found {len(subfolders)} subfolders: {subfolders}")
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(self.image_folder_path, subfolder)
                print(f"\nScanning subfolder: {subfolder}")
                
                # Determine label from subfolder name
                label = self.determine_label_from_class_name(subfolder)
                
                # Find images in this subfolder
                subfolder_images = []
                try:
                    for file in os.listdir(subfolder_path):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            img_path = os.path.join(subfolder_path, file)
                            subfolder_images.append((img_path, label, subfolder))
                    
                    print(f"  Found {len(subfolder_images)} images in {subfolder} (label: {label})")
                    all_images.extend(subfolder_images)
                    
                except Exception as e:
                    print(f"Error accessing {subfolder}: {e}")
        
        print(f"\nTotal images found: {len(all_images)}")
        return all_images
    
    def determine_label_from_class_name(self, class_name):
        """Determine label from class/folder name"""
        class_name_lower = class_name.lower().replace('-', '').replace('_', '')
        
        if class_name_lower in self.class_mapping:
            return self.class_mapping[class_name_lower]
        
        tumor_keywords = ['tumor', 'glioma', 'meningioma', 'pituitary', 'cancer', 'malignant']
        no_tumor_keywords = ['no', 'normal', 'benign', 'healthy']
        
        for keyword in tumor_keywords:
            if keyword in class_name_lower:
                return 1
        
        for keyword in no_tumor_keywords:
            if keyword in class_name_lower:
                return 0
        
        print(f"Unknown class name '{class_name}', assigning label 1 (tumor)")
        return 1
    
    def extract_features_from_image(self, image_path):
        """Extract features from a brain scan image"""
        try:
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            # Resize image for consistent processing
            image = cv2.resize(image, (256, 256))
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Threshold to create binary image
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate geometric features
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                else:
                    compactness = 0
            else:
                area = perimeter = compactness = 0
            
            # Calculate texture features
            mean_intensity = np.mean(image)
            std_intensity = np.std(image)
            
            # Simple contrast measure
            contrast = np.std(image.astype(np.float32))
            
            # Simple energy measure
            normalized_image = image.astype(np.float32) / 255.0
            energy = np.sum(normalized_image ** 2) / (256 * 256)
            
            # Simple homogeneity measure using gradients
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            homogeneity = 1.0 / (1.0 + np.mean(gradient_magnitude))
            
            # Simple entropy measure
            hist, _ = np.histogram(image, bins=256, range=(0, 256))
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist = hist / np.sum(hist)
                entropy = -np.sum(hist * np.log2(hist))
            else:
                entropy = 0
            
            features = {
                'area': area,
                'perimeter': perimeter,
                'compactness': compactness,
                'contrast': contrast,
                'energy': energy,
                'homogeneity': homogeneity,
                'entropy': entropy,
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
            return None
    
    def load_image_for_cnn(self, image_path, target_size=(128, 128)):
        """Load and preprocess image for CNN model"""
        try:
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def process_images_to_csv(self, output_csv_path='brain_tumor_features.csv'):
        """Process all images and create CSV file with features"""
        print("=" * 60)
        print("PROCESSING IMAGES TO CSV")
        print("=" * 60)
        
        # Find all images
        all_images = self.find_all_images()
        
        if len(all_images) == 0:
            print("No images found to process!")
            return None
        
        # Process each image
        features_list = []
        successful_processing = 0
        
        print(f"\nProcessing {len(all_images)} images...")
        
        for i, (image_path, label, class_name) in enumerate(all_images):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(all_images)}")
            
            features = self.extract_features_from_image(image_path)
            if features is not None:
                features['filename'] = os.path.basename(image_path)
                features['image_path'] = image_path
                features['label'] = label
                features['class_name'] = class_name
                features_list.append(features)
                successful_processing += 1
        
        print(f"\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total images found: {len(all_images)}")
        print(f"Successfully processed: {successful_processing}")
        print(f"Failed processing: {len(all_images) - successful_processing}")
        
        if len(features_list) == 0:
            print("No features extracted!")
            return None
        
        # Create DataFrame
        self.csv_data = pd.DataFrame(features_list)
        
        # Save to CSV
        self.csv_data.to_csv(output_csv_path, index=False)
        print(f"CSV file saved: {output_csv_path}")
        
        # Display statistics
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(self.csv_data)}")
        
        # Class distribution
        class_counts = self.csv_data['class_name'].value_counts()
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        # Label distribution
        label_counts = self.csv_data['label'].value_counts()
        print(f"\nLabel distribution:")
        print(f"  No Tumor (0): {label_counts.get(0, 0)}")
        print(f"  Tumor (1): {label_counts.get(1, 0)}")
        
        return self.csv_data
    
    def prepare_data_splits(self, test_size=0.2, random_state=42):
        """Prepare train/test splits for both CSV and image data"""
        if self.csv_data is None:
            print("No CSV data available!")
            return False
        
        print(f"\n" + "=" * 60)
        print("PREPARING DATA SPLITS")
        print("=" * 60)
        
        # Prepare CSV features and labels
        X_csv = self.csv_data[self.feature_columns]
        y = self.csv_data['label']
        image_paths = self.csv_data['image_path'].values
        
        # Split data (same split for both models)
        try:
            indices = np.arange(len(X_csv))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # CSV data splits
            self.X_train_csv = X_csv.iloc[train_idx]
            self.X_test_csv = X_csv.iloc[test_idx]
            self.y_train = y.iloc[train_idx]
            self.y_test = y.iloc[test_idx]
            
            # Image paths for image model
            train_image_paths = image_paths[train_idx]
            self.test_image_paths = image_paths[test_idx]
            
            # Load images for CNN
            print("Loading training images...")
            train_images = []
            for i, img_path in enumerate(train_image_paths):
                if i % 100 == 0:
                    print(f"Loading training images: {i}/{len(train_image_paths)}")
                img = self.load_image_for_cnn(img_path)
                if img is not None:
                    train_images.append(img)
            
            print("Loading test images...")
            test_images = []
            for i, img_path in enumerate(self.test_image_paths):
                if i % 100 == 0:
                    print(f"Loading test images: {i}/{len(self.test_image_paths)}")
                img = self.load_image_for_cnn(img_path)
                if img is not None:
                    test_images.append(img)
            
            self.X_train_images = np.array(train_images)
            self.X_test_images = np.array(test_images)
            
            print(f"Data splits prepared successfully!")
            print(f"CSV Training samples: {len(self.X_train_csv)}")
            print(f"CSV Testing samples: {len(self.X_test_csv)}")
            print(f"Image Training samples: {len(self.X_train_images)}")
            print(f"Image Testing samples: {len(self.X_test_images)}")
            
            return True
            
        except Exception as e:
            print(f"Error preparing data splits: {e}")
            return False
    
    def train_csv_model(self, model_type='random_forest'):
        if self.X_train_csv is None:
            print("No CSV training data available!")
            return None
        
        print(f"\n" + "=" * 60)
        print(f"TRAINING CSV MODEL ({model_type.upper()})")
        print("=" * 60)
        
        # Scale features if needed
        if model_type in ['svm', 'logistic_regression']:
            X_train_scaled = self.scaler.fit_transform(self.X_train_csv)
            X_test_scaled = self.scaler.transform(self.X_test_csv)
        else:
            X_train_scaled = self.X_train_csv
            X_test_scaled = self.X_test_csv
        
        # Initialize model
        if model_type == 'random_forest':
            self.csv_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.csv_model = SVC(kernel='rbf', random_state=42, probability=True)
        elif model_type == 'logistic_regression':
            self.csv_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Train model
        print("Training CSV model...")
        self.csv_model.fit(X_train_scaled, self.y_train)
        
        # Make predictions
        print("Making CSV predictions...")
        self.csv_predictions = self.csv_model.predict(X_test_scaled)
        self.csv_probabilities = self.csv_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate accuracy
        csv_accuracy = accuracy_score(self.y_test, self.csv_predictions)
        print(f"CSV Model Accuracy: {csv_accuracy:.4f}")
        
        return self.csv_model
    
    def create_cnn_model(self, input_shape=(128, 128, 1)):
        """Create CNN model for image classification"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_image_model(self, epochs=20, batch_size=32):
        """Train CNN model on images"""
        if self.X_train_images is None or len(self.X_train_images) == 0:
            print("No image training data available!")
            return None
        
        print(f"\n" + "=" * 60)
        print("TRAINING IMAGE MODEL (CNN)")
        print("=" * 60)
        
        # Create model
        self.image_model = self.create_cnn_model()
        
        print("CNN Model Architecture:")
        self.image_model.summary()
        
        # Train model
        print(f"\nTraining CNN model for {epochs} epochs...")
        history = self.image_model.fit(
            self.X_train_images, self.y_train.values,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Make predictions
        print("Making image predictions...")
        self.image_probabilities = self.image_model.predict(self.X_test_images).flatten()
        self.image_predictions = (self.image_probabilities > 0.5).astype(int)
        
        # Calculate accuracy
        image_accuracy = accuracy_score(self.y_test, self.image_predictions)
        print(f"Image Model Accuracy: {image_accuracy:.4f}")
        
        return self.image_model
    
    def compare_predictions(self):
        """Compare predictions from both models"""
        if self.csv_predictions is None or self.image_predictions is None:
            print("Both models must be trained first!")
            return
        
        print(f"\n" + "=" * 60)
        print("DUAL MODEL COMPARISON")
        print("=" * 60)
        
        # Calculate accuracies
        csv_accuracy = accuracy_score(self.y_test, self.csv_predictions)
        image_accuracy = accuracy_score(self.y_test, self.image_predictions)
        
        print(f"CSV Model Accuracy: {csv_accuracy:.4f}")
        print(f"Image Model Accuracy: {image_accuracy:.4f}")
        
        # Agreement between models
        agreement = np.mean(self.csv_predictions == self.image_predictions)
        print(f"Model Agreement: {agreement:.4f}")
        
        # Cases where both models agree and are correct
        both_correct = (self.csv_predictions == self.y_test) & (self.image_predictions == self.y_test)
        both_correct_rate = np.mean(both_correct)
        print(f"Both Models Correct: {both_correct_rate:.4f}")
        
        # Cases where both models agree but are wrong
        both_wrong = (self.csv_predictions != self.y_test) & (self.image_predictions != self.y_test) & (self.csv_predictions == self.image_predictions)
        both_wrong_rate = np.mean(both_wrong)
        print(f"Both Models Wrong (but agree): {both_wrong_rate:.4f}")
        
        # Ensemble prediction (majority vote)
        ensemble_predictions = ((self.csv_probabilities + self.image_probabilities) / 2 > 0.5).astype(int)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_predictions)
        print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
        
        return {
            'csv_accuracy': csv_accuracy,
            'image_accuracy': image_accuracy,
            'agreement': agreement,
            'both_correct_rate': both_correct_rate,
            'both_wrong_rate': both_wrong_rate,
            'ensemble_accuracy': ensemble_accuracy
        }
    
    def generate_dual_confusion_matrices(self, save_path='dual_confusion_matrices.png'):
        """Generate confusion matrices for both models"""
        if self.csv_predictions is None or self.image_predictions is None:
            print("Both models must be trained first!")
            return
        
        print(f"\n" + "=" * 60)
        print("GENERATING DUAL CONFUSION MATRICES")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # CSV Model Confusion Matrix
        cm_csv = confusion_matrix(self.y_test, self.csv_predictions)
        sns.heatmap(cm_csv, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Tumor', 'Tumor'], 
                    yticklabels=['No Tumor', 'Tumor'])
        axes[0].set_title('CSV Model Confusion Matrix')
        axes[0].set_ylabel('Actual Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Image Model Confusion Matrix
        cm_image = confusion_matrix(self.y_test, self.image_predictions)
        sns.heatmap(cm_image, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=['No Tumor', 'Tumor'], 
                    yticklabels=['No Tumor', 'Tumor'])
        axes[1].set_title('Image Model Confusion Matrix')
        axes[1].set_ylabel('Actual Label')
        axes[1].set_xlabel('Predicted Label')
        
        # Ensemble Model Confusion Matrix
        ensemble_predictions = ((self.csv_probabilities + self.image_probabilities) / 2 > 0.5).astype(int)
        cm_ensemble = confusion_matrix(self.y_test, ensemble_predictions)
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
                    xticklabels=['No Tumor', 'Tumor'], 
                    yticklabels=['No Tumor', 'Tumor'])
        axes[2].set_title('Ensemble Model Confusion Matrix')
        axes[2].set_ylabel('Actual Label')
        axes[2].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Dual confusion matrices saved: {save_path}")
    
    def save_dual_prediction_results(self, output_path='dual_prediction_results.csv'):
        """Save detailed comparison of both model predictions"""
        if self.csv_predictions is None or self.image_predictions is None:
            print("Both models must be trained first!")
            return None
        
        # Create results DataFrame
        test_indices = self.X_test_csv.index
        results_df = pd.DataFrame({
            'filename': self.csv_data.loc[test_indices, 'filename'].values,
            'image_path': self.test_image_paths,
            'actual_label': self.y_test.values,
            'csv_prediction': self.csv_predictions,
            'image_prediction': self.image_predictions,
            'csv_probability': self.csv_probabilities,
            'image_probability': self.image_probabilities,
            'models_agree': (self.csv_predictions == self.image_predictions),
            'csv_correct': (self.csv_predictions == self.y_test.values),
            'image_correct': (self.image_predictions == self.y_test.values),
            'both_correct': (self.csv_predictions == self.y_test.values) & (self.image_predictions == self.y_test.values)
        })
        
        # Add ensemble prediction
        results_df['ensemble_prediction'] = ((self.csv_probabilities + self.image_probabilities) / 2 > 0.5).astype(int)
        results_df['ensemble_correct'] = (results_df['ensemble_prediction'] == self.y_test.values)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        print(f"Dual prediction results saved: {output_path}")
        print(f"Total predictions: {len(results_df)}")
        print(f"Cases where models agree: {sum(results_df['models_agree'])}")
        print(f"Cases where both models correct: {sum(results_df['both_correct'])}")
        
        return results_df
    
    def save_trained_models(self, csv_model_path="brain_tumor_csv_model.pkl", image_model_path="brain_tumor_image_model.h5", scaler_path="brain_tumor_scaler.pkl"):
        print(f"\n" + "=" * 60)
        print("SAVING TRAINED MODELS")
        print("=" * 60)
        
        try:
            # Save CSV model
            if self.csv_model is not None:
                joblib.dump(self.csv_model, csv_model_path)
                print(f"CSV model saved: {csv_model_path}")
            
            # Save image model
            if self.image_model is not None:
                self.image_model.save(image_model_path)
                print(f"Image model saved: {image_model_path}")
            
            # Save scaler
            if self.scaler is not None:
                joblib.dump(self.scaler, scaler_path)
                print(f"Scaler saved: {scaler_path}")
            
            print("All models saved successfully!")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def test_single_image(self, image_path, show_visualization=True):
        """
        Test a single image with the trained models
        
        Args:
            image_path (str): Path to the brain scan image
            show_visualization (bool): Whether to show the result visualization
        """
        print(f"\n" + "=" * 60)
        print(f"TESTING IMAGE: {os.path.basename(image_path)}")
        print("=" * 60)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        results = {}
        
        # CSV Model Prediction
        if self.csv_model is not None:
            print("🔬 Making CSV prediction...")
            features = self.extract_features_from_image(image_path)
            if features is not None:
                feature_df = pd.DataFrame([features])[self.feature_columns]
                
                # Check if scaler exists and is fitted
                try:
                    if self.scaler is not None and hasattr(self.scaler, 'mean_'):
                        # Scaler is fitted, use it
                        feature_scaled = self.scaler.transform(feature_df)
                    else:
                        # No scaler or not fitted, use raw features
                        feature_scaled = feature_df
                except:
                    # If any error, use raw features
                    feature_scaled = feature_df
                
                csv_pred = self.csv_model.predict(feature_scaled)[0]
                csv_prob = self.csv_model.predict_proba(feature_scaled)[0]
                
                results['csv_prediction'] = csv_pred
                results['csv_probability'] = csv_prob[1]
                results['csv_confidence'] = csv_prob[csv_pred]
                
                print(f"  CSV Result: {'TUMOR' if csv_pred == 1 else 'NO TUMOR'}")
                print(f"  CSV Confidence: {csv_prob[csv_pred]:.4f}")
        
        # Image Model Prediction
        if self.image_model is not None:
            print("Making CNN prediction...")
            image_data = self.load_image_for_cnn(image_path)
            if image_data is not None:
                image_batch = np.expand_dims(image_data, axis=0)
                image_prob = self.image_model.predict(image_batch, verbose=0)[0][0]
                image_pred = 1 if image_prob > 0.5 else 0
                
                results['image_prediction'] = image_pred
                results['image_probability'] = image_prob
                results['image_confidence'] = image_prob if image_pred == 1 else (1 - image_prob)
                
                print(f"  CNN Result: {'TUMOR' if image_pred == 1 else 'NO TUMOR'}")
                print(f"  CNN Confidence: {results['image_confidence']:.4f}")
        
        # Ensemble Prediction
        if 'csv_probability' in results and 'image_probability' in results:
            ensemble_prob = (results['csv_probability'] + results['image_probability']) / 2
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            ensemble_conf = ensemble_prob if ensemble_pred == 1 else (1 - ensemble_prob)
            
            results['ensemble_prediction'] = ensemble_pred
            results['ensemble_probability'] = ensemble_prob
            results['ensemble_confidence'] = ensemble_conf
            
            print(f"ENSEMBLE Result: {'TUMOR' if ensemble_pred == 1 else 'NO TUMOR'}")
            print(f"ENSEMBLE Confidence: {ensemble_conf:.4f}")
        
        # Show visualization
        if show_visualization:
            self.visualize_prediction(image_path, results)
        
        return results
    def visualize_prediction(self, image_path, results):
        """
        Visualize the prediction results
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            plt.figure(figsize=(12, 6))
            
            # Show image
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Brain Scan: {os.path.basename(image_path)}")
            plt.axis('off')
            
            # Show predictions
            plt.subplot(1, 2, 2)
            plt.axis('off')
            
            y_pos = 0.9
            plt.text(0.1, y_pos, "PREDICTION RESULTS", fontsize=16, fontweight='bold')
            y_pos -= 0.15
            
            if 'csv_prediction' in results:
                csv_result = "TUMOR" if results['csv_prediction'] == 1 else "NO TUMOR"
                plt.text(0.1, y_pos, f"CSV: {csv_result}", fontsize=12, fontweight='bold')
                plt.text(0.1, y_pos-0.05, f"Confidence: {results['csv_confidence']:.3f}", fontsize=10)
                y_pos -= 0.15
            
            if 'image_prediction' in results:
                img_result = "TUMOR" if results['image_prediction'] == 1 else "NO TUMOR"
                plt.text(0.1, y_pos, f"CNN: {img_result}", fontsize=12, fontweight='bold')
                plt.text(0.1, y_pos-0.05, f"Confidence: {results['image_confidence']:.3f}", fontsize=10)
                y_pos -= 0.15
            
            if 'ensemble_prediction' in results:
                ens_result = "TUMOR" if results['ensemble_prediction'] == 1 else "NO TUMOR"
                color = 'red' if results['ensemble_prediction'] == 1 else 'green'
                plt.text(0.1, y_pos, f"FINAL: {ens_result}", fontsize=14, fontweight='bold', color=color)
                plt.text(0.1, y_pos-0.05, f"Confidence: {results['ensemble_confidence']:.3f}", fontsize=12)
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error showing visualization: {e}")


def main():
    """Main execution function for dual prediction system"""
    print("DUAL PREDICTION BRAIN TUMOR DETECTION SYSTEM")
    print("=" * 60)
    
    # Set your image folder path
    image_folder = r"C:\Users\hvver\Desktop\ntcc\data\Brian\Training"
    
    # Initialize detector
    detector = DualPredictionBrainTumorDetector(image_folder)
    
    try:
        # Step 1: Process images and create CSV
        print("STEP 1: Processing images to CSV...")
        csv_data = detector.process_images_to_csv('brain_tumor_features.csv')
        
        if csv_data is None or len(csv_data) == 0:
            print("Failed to process images. Exiting.")
            return
        
        # Step 2: Prepare data splits
        print("STEP 2: Preparing data splits...")
        if not detector.prepare_data_splits():
            print("Failed to prepare data splits. Exiting.")
            return
        
        # Step 3: Train CSV model
        print("STEP 3: Training CSV model...")
        csv_model = detector.train_csv_model(model_type='random_forest')
        
        if csv_model is None:
            print("Failed to train CSV model. Exiting.")
            return
        
        # Step 4: Train Image model
        print("STEP 4: Training Image model...")
        # CHANGE epochs=10 to epochs=30 for your 99.29% accuracy
        image_model = detector.train_image_model(epochs=30, batch_size=32)
        
        if image_model is None:
            print("Failed to train image model. Exiting.")
            return
        
        # Step 5: Compare predictions
        print("STEP 5: Comparing predictions...")
        comparison_results = detector.compare_predictions()
        
        # Step 6: Generate visualizations
        print("STEP 6: Generating visualizations...")
        detector.generate_dual_confusion_matrices('dual_confusion_matrices.png')
        
        # Step 7: Save results
        print("STEP 7: Saving detailed results...")
        detector.save_dual_prediction_results('dual_prediction_results.csv')
        
        # Step 8: Save models
        print("STEP 8: Saving trained models...")
        detector.save_trained_models()
        
        # Step 9: Test on new image (FIXED VERSION)
        print("\n" + "=" * 60)
        print("STEP 9: Testing trained model")
        print("=" * 60)
        
        # Ask if user wants to test
        test_choice = input("Do you want to test an image? (y/n): ").strip().lower()
        
        if test_choice == 'y':
            # Use the get_image_path function
            img_path = get_image_path()
            
            if img_path and os.path.exists(img_path):
                print(f"Testing image: {img_path}")
                test_result = detector.test_single_image(img_path, show_visualization=True)
            else:
                print(f"Image not found: {img_path}")
                print("Skipping test phase.")
        else:
            print("Skipping test phase.")
        
        print("\n" + "=" * 60)
        print("DUAL PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("- brain_tumor_features.csv")
        print("- dual_confusion_matrices.png")
        print("- dual_prediction_results.csv")
        print("- brain_tumor_csv_model.pkl")
        print("- brain_tumor_image_model.h5")
        print("- brain_tumor_scaler.pkl")
        print(f"\nFinal Results:")
        print(f"CSV Model Accuracy: {comparison_results['csv_accuracy']:.4f}")
        print(f"Image Model Accuracy: {comparison_results['image_accuracy']:.4f}")
        print(f"Ensemble Accuracy: {comparison_results['ensemble_accuracy']:.4f}")
        
        # RETURN the detector so you can use it for more testing
        return detector
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def test_saved_model(image_path):
    print("TESTING WITH SAVED MODELS")
    print("=" * 50)

    try:
        # Load saved models
        print("Loading saved models...")
        csv_model = joblib.load("brain_tumor_csv_model.pkl")
        image_model = keras.models.load_model("brain_tumor_image_model.h5")
        scaler = joblib.load("brain_tumor_scaler.pkl")
        print("Models loaded successfully!")
        
        # Create a temporary detector instance for testing
        detector = DualPredictionBrainTumorDetector("")  # Empty path since we're not training
        detector.csv_model = csv_model
        detector.image_model = image_model
        detector.scaler = scaler
        
        # Test the image
        result = detector.test_single_image(image_path, show_visualization=True)
        
        if result and 'ensemble_prediction' in result:
            final_result = "TUMOR DETECTED" if result['ensemble_prediction'] == 1 else "NO TUMOR"
            final_confidence = result['ensemble_confidence']
            print(f"\nFINAL DIAGNOSIS: {final_result}")
            print(f"CONFIDENCE: {final_confidence:.4f} ({final_confidence*100:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"Error testing with saved models: {e}")
        return None
    
# ADD THIS FUNCTION AFTER test_saved_model()
def get_image_path():
    """
    Get image path from user with automatic path cleaning
    Handles: drag & drop, copy-paste, manual entry with any format
    """
    print("\nEnter image path (drag & drop or paste):")
    
    img_path = input("Path: ").strip()
    
    # Remove quotes (from drag & drop or copy-paste)
    img_path = img_path.strip('"').strip("'")
    
    # Remove 'r' prefix if present
    if img_path.startswith('r"') or img_path.startswith("r'"):
        img_path = img_path[2:-1]
    
    # Normalize path for the operating system
    img_path = os.path.normpath(img_path)
    
    return img_path

if __name__ == "__main__":
    print("BRAIN TUMOR DETECTION SYSTEM")
    print("=" * 60)
    print("Choose an option:")
    print("1. Train new model and test")
    print("2. Test with existing saved model")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Train and test
        detector = main()
        
        if detector is not None:
            # Optional: Test more images after training
            while True:
                test_another = input("\nTest another image? (y/n): ").strip().lower()
                if test_another == 'y':
                    img_path = get_image_path()
                    
                    if img_path and os.path.exists(img_path):
                        print(f"Found: {img_path}")
                        detector.test_single_image(img_path)
                    else:
                        print(f"Image not found: {img_path}")
                else:
                    break
                
    elif choice == "2":
        # Test with saved model
        img_path = get_image_path()
        
        if img_path and os.path.exists(img_path):
            print(f"Found: {img_path}")
            test_saved_model(img_path)
        else:
            print(f"Image not found: {img_path}")
    
    else:
        print("Invalid choice!")
    
    print("\nProgram finished!")