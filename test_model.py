"""
Model Testing and Visualization
==============================

Test saved plateau classification models on real data and generate
comprehensive visualizations of predictions vs actual results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    """Class for testing saved models and generating visualizations."""
    
    def __init__(self, models_dir="./models", data_dir="./Data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = self.models_dir / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def load_best_model(self):
        """Load the best saved model and scaler."""
        # First try to load PyTorch model
        pytorch_model_path = self.models_dir / "pytorch_neural_net.pth"
        pytorch_metadata_path = self.models_dir / "pytorch_model_metadata.json"
        
        if pytorch_model_path.exists():
            return self.load_pytorch_model()
        
        # Fallback to scikit-learn model
        try:
            model_path = self.models_dir / "best_plateau_classifier.pkl"
            scaler_path = self.models_dir / "scaler.pkl"
            metadata_path = self.models_dir / "model_metadata.json"
            
            print(f"Loading model from: {model_path}")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            print(f"✓ Model loaded successfully: {type(model).__name__}")
            return model, scaler, metadata
            
        except FileNotFoundError as e:
            print(f"❌ Error loading model: {e}")
            print("Please train a model first using plateau_classification.py or plateau_classification_gpu.py")
            return None, None, None
    
    def load_pytorch_model(self):
        """Load PyTorch model specifically."""
        import torch
        from plateau_classification_gpu import PyTorchNeuralNet
        
        model_path = self.models_dir / "pytorch_neural_net.pth"
        metadata_path = self.models_dir / "pytorch_model_metadata.json"
        
        print(f"Loading PyTorch model from: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Recreate model
        input_size = checkpoint['input_size']
        model = PyTorchNeuralNet(input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"✓ PyTorch model loaded successfully")
        print(f"Model trained on device: {checkpoint.get('device', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('validation_accuracy', 'unknown'):.4f}")
        
        # Create a dummy scaler since PyTorch model was trained with scaled data
        # We'll need to recreate the scaler from the training process
        scaler = self.create_scaler_for_pytorch()
        
        return model, scaler, metadata
    
    def create_scaler_for_pytorch(self):
        """Create a scaler for PyTorch model testing."""
        from sklearn.preprocessing import StandardScaler
        from plateau_classification import PlateauClassifier
        
        print("Creating scaler for PyTorch model...")
        
        # Load training data to fit scaler
        try:
            train_data = pd.read_csv(self.data_dir / "train_raw.csv")
            classifier = PlateauClassifier()
            classifier.train_data = train_data
            classifier.prepare_features()
            
            return classifier.scaler
        except:
            print("⚠️  Could not recreate scaler from training data, using standard scaler")
            return StandardScaler()
    
    def load_test_data(self):
        """Load and prepare test data."""
        try:
            test_data = pd.read_csv(self.data_dir / "test_raw.csv")
            print(f"✓ Test data loaded: {test_data.shape}")
            return test_data
        except FileNotFoundError:
            print("❌ Test data not found. Creating synthetic test data...")
            return self.create_synthetic_test_data()
    
    def create_synthetic_test_data(self):
        """Create synthetic test data that mimics real plateau behavior."""
        print("Creating synthetic test data with realistic plateau patterns...")
        
        np.random.seed(42)
        data_points = []
        
        # Create 3 synthetic experiments with different patterns
        experiments = ['synthetic_exp_1', 'synthetic_exp_2', 'synthetic_exp_3']
        
        for exp_id, exp_name in enumerate(experiments):
            n_points = np.random.randint(800, 1200)
            time = np.linspace(0, 100, n_points)
            
            # Create realistic temperature profiles
            if exp_id == 0:  # Heating with plateau
                temp = 20 + 0.5 * time + 10 * np.sin(0.1 * time) + np.random.normal(0, 2, n_points)
                # Add plateau region
                plateau_start, plateau_end = 300, 600
                temp[plateau_start:plateau_end] = np.mean(temp[plateau_start:plateau_end]) + np.random.normal(0, 1, plateau_end - plateau_start)
                zones = np.ones(n_points)  # Mostly heating
                zones[plateau_start:plateau_end] = 0  # Plateau region
                
            elif exp_id == 1:  # Cooling with plateau
                temp = 80 - 0.3 * time + 5 * np.cos(0.08 * time) + np.random.normal(0, 1.5, n_points)
                plateau_start, plateau_end = 200, 500
                temp[plateau_start:plateau_end] = np.mean(temp[plateau_start:plateau_end]) + np.random.normal(0, 0.8, plateau_end - plateau_start)
                zones = np.full(n_points, 2)  # Mostly cooling
                zones[plateau_start:plateau_end] = 0  # Plateau region
                
            else:  # Complex pattern with multiple phases
                temp = 25 + 15 * np.sin(0.05 * time) + 0.2 * time + np.random.normal(0, 2, n_points)
                zones = np.zeros(n_points)
                # Add heating and cooling phases
                zones[100:300] = 1  # Heating
                zones[600:800] = 2  # Cooling
                # Rest are plateaus (0)
            
            # Create DataFrame for this experiment
            exp_data = pd.DataFrame({
                'experiment': exp_name,
                'time': time,
                'temp': temp,
                'zones': zones.astype(int)
            })
            
            data_points.append(exp_data)
        
        synthetic_data = pd.concat(data_points, ignore_index=True)
        print(f"✓ Created synthetic test data: {synthetic_data.shape}")
        print(f"Zone distribution: {synthetic_data['zones'].value_counts().to_dict()}")
        
        return synthetic_data
    
    def engineer_test_features(self, df, feature_names):
        """Engineer features for test data using the same process as training."""
        print("Engineering features for test data...")
        
        from plateau_classification import PlateauClassifier
        classifier = PlateauClassifier()
        
        # Use the same feature engineering process
        features_df = classifier.engineer_features(df)
        
        # Select only the features that were used in training
        available_features = [col for col in feature_names if col in features_df.columns]
        missing_features = [col for col in feature_names if col not in features_df.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                features_df[feature] = 0
        
        X_test = features_df[feature_names]
        y_test = features_df['zones']
        
        print(f"✓ Features engineered: {X_test.shape}")
        return X_test, y_test, features_df
    
    def test_model_performance(self, model, scaler, X_test, y_test):
        """Test model performance and generate metrics."""
        print("\nTesting model performance...")
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Check if it's a PyTorch model
        if hasattr(model, 'forward'):  # PyTorch model
            y_pred, y_proba = self.predict_pytorch_model(model, X_test_scaled)
        else:  # Scikit-learn model
            y_pred = model.predict(X_test_scaled)
            # Get prediction probabilities if available
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = ['Plateau', 'Heating', 'Cooling']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return y_pred, y_proba, accuracy, report
    
    def predict_pytorch_model(self, model, X_test_scaled):
        """Make predictions with PyTorch model."""
        import torch
        
        model.eval()
        with torch.no_grad():
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_test_scaled)
            
            # Get predictions
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert back to numpy
            y_pred = predictions.numpy()
            y_proba = probabilities.numpy()
        
        return y_pred, y_proba
    
    def create_prediction_visualizations(self, test_data, y_test, y_pred, y_proba=None):
        """Create comprehensive visualizations of predictions."""
        print("\nCreating prediction visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Plateau', 'Heating', 'Cooling'],
                   yticklabels=['Plateau', 'Heating', 'Cooling'])
        plt.title('Confusion Matrix - Predictions vs Actual', fontsize=16)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        confusion_path = self.results_dir / 'confusion_matrix_test.png'
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved: {confusion_path}")
        
        # 2. Time Series Predictions for each experiment
        experiments = test_data['experiment'].unique()
        
        fig, axes = plt.subplots(len(experiments), 1, figsize=(15, 5*len(experiments)))
        if len(experiments) == 1:
            axes = [axes]
        
        colors = {0: 'blue', 1: 'red', 2: 'green'}
        labels = {0: 'Plateau', 1: 'Heating', 2: 'Cooling'}
        
        for i, exp in enumerate(experiments):
            exp_mask = test_data['experiment'] == exp
            exp_data = test_data[exp_mask].reset_index(drop=True)
            exp_actual = y_test[exp_mask]
            exp_pred = y_pred[exp_mask]
            
            # Convert to pandas Series if they're numpy arrays
            if isinstance(exp_actual, np.ndarray):
                exp_actual = pd.Series(exp_actual).reset_index(drop=True)
            if isinstance(exp_pred, np.ndarray):
                exp_pred = pd.Series(exp_pred).reset_index(drop=True)
            
            ax = axes[i]
            
            # Plot temperature
            ax2 = ax.twinx()
            ax2.plot(exp_data['time'], exp_data['temp'], 'k-', alpha=0.7, linewidth=1, label='Temperature')
            ax2.set_ylabel('Temperature', fontsize=10)
            ax2.legend(loc='upper right')
            
            # Plot actual zones as background
            for zone in [0, 1, 2]:
                zone_mask = exp_actual == zone
                if zone_mask.any():
                    ax.scatter(exp_data['time'][zone_mask], [zone]*sum(zone_mask), 
                             c=colors[zone], alpha=0.6, s=20, label=f'Actual {labels[zone]}')
            
            # Plot predictions as markers
            for zone in [0, 1, 2]:
                zone_mask = exp_pred == zone
                if zone_mask.any():
                    ax.scatter(exp_data['time'][zone_mask], [zone + 0.1]*sum(zone_mask), 
                             c=colors[zone], alpha=0.9, s=30, marker='^', 
                             label=f'Predicted {labels[zone]}')
            
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Zone Classification', fontsize=10)
            ax.set_title(f'Predictions vs Actual - {exp}', fontsize=12)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Plateau', 'Heating', 'Cooling'])
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timeseries_path = self.results_dir / 'timeseries_predictions.png'
        plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Time series predictions saved: {timeseries_path}")
        
        # 3. Prediction Confidence (if probabilities available)
        if y_proba is not None:
            plt.figure(figsize=(12, 8))
            
            # Plot confidence for each class
            for i, class_name in enumerate(['Plateau', 'Heating', 'Cooling']):
                plt.subplot(2, 2, i+1)
                confidence = y_proba[:, i]
                plt.hist(confidence, bins=50, alpha=0.7, color=colors[i])
                plt.title(f'{class_name} Prediction Confidence')
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            # Overall confidence distribution
            plt.subplot(2, 2, 4)
            max_confidence = np.max(y_proba, axis=1)
            plt.hist(max_confidence, bins=50, alpha=0.7, color='purple')
            plt.title('Maximum Prediction Confidence')
            plt.xlabel('Max Confidence Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            confidence_path = self.results_dir / 'prediction_confidence.png'
            plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Prediction confidence plots saved: {confidence_path}")
        
        # 4. Performance Summary
        self.create_performance_summary(y_test, y_pred)
    
    def create_performance_summary(self, y_test, y_pred):
        """Create a performance summary visualization."""
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        # Create summary plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        classes = ['Plateau', 'Heating', 'Cooling']
        x_pos = np.arange(len(classes))
        
        # Precision
        axes[0].bar(x_pos, precision, color=['blue', 'red', 'green'], alpha=0.7)
        axes[0].set_title('Precision by Class')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(classes)
        axes[0].set_ylim(0, 1)
        for i, v in enumerate(precision):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Recall
        axes[1].bar(x_pos, recall, color=['blue', 'red', 'green'], alpha=0.7)
        axes[1].set_title('Recall by Class')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(classes)
        axes[1].set_ylim(0, 1)
        for i, v in enumerate(recall):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # F1-Score
        axes[2].bar(x_pos, f1, color=['blue', 'red', 'green'], alpha=0.7)
        axes[2].set_title('F1-Score by Class')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(classes)
        axes[2].set_ylim(0, 1)
        for i, v in enumerate(f1):
            axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        summary_path = self.results_dir / 'performance_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Performance summary saved: {summary_path}")
    
    def save_test_results(self, test_data, y_test, y_pred, y_proba, accuracy, report):
        """Save test results to files."""
        print("\nSaving test results...")
        
        # Create results DataFrame
        results_df = test_data.copy()
        results_df['actual_zone'] = y_test
        results_df['predicted_zone'] = y_pred
        results_df['correct_prediction'] = (y_test == y_pred)
        
        if y_proba is not None:
            for i, class_name in enumerate(['plateau_prob', 'heating_prob', 'cooling_prob']):
                results_df[class_name] = y_proba[:, i]
            results_df['prediction_confidence'] = np.max(y_proba, axis=1)
        
        # Save to CSV
        results_csv_path = self.results_dir / 'test_results.csv'
        results_df.to_csv(results_csv_path, index=False)
        print(f"✓ Test results saved: {results_csv_path}")
        
        # Save summary statistics
        summary = {
            'test_accuracy': float(accuracy),
            'total_samples': len(y_test),
            'correct_predictions': int(sum(y_test == y_pred)),
            'class_distribution': {
                'plateau': int(sum(y_test == 0)),
                'heating': int(sum(y_test == 1)),
                'cooling': int(sum(y_test == 2))
            },
            'classification_report': report
        }
        
        summary_path = self.results_dir / 'test_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Test summary saved: {summary_path}")
    
    def run_comprehensive_test(self):
        """Run comprehensive model testing."""
        print("🚀 Starting Comprehensive Model Testing")
        print("=" * 50)
        
        # Load model
        model, scaler, metadata = self.load_best_model()
        if model is None:
            return
        
        # Load test data
        test_data = self.load_test_data()
        
        # Get feature names from metadata or create them
        feature_names = metadata.get('features', [])
        if not feature_names:
            print("⚠️  No feature names found in metadata, using default feature set")
            # Create default feature names based on the feature engineering process
            feature_names = [
                'temp', 'temp_diff', 'temp_diff2', 'temp_ma_5', 'temp_std_5', 'temp_diff_ma_5',
                'temp_ma_10', 'temp_std_10', 'temp_diff_ma_10', 'temp_ma_20', 'temp_std_20', 
                'temp_diff_ma_20', 'temp_change_rate', 'relative_time', 'cumsum_temp', 
                'cumsum_temp_diff', 'temp_acceleration', 'is_local_max', 'is_local_min', 
                'experiment_encoded'
            ]
        
        # Engineer features
        X_test, y_test, features_df = self.engineer_test_features(test_data, feature_names)
        
        # Test model
        y_pred, y_proba, accuracy, report = self.test_model_performance(model, scaler, X_test, y_test)
        
        # Create visualizations
        self.create_prediction_visualizations(test_data, y_test, y_pred, y_proba)
        
        # Save results
        self.save_test_results(test_data, y_test, y_pred, y_proba, accuracy, report)
        
        print("\n🎉 Testing completed successfully!")
        print(f"📊 Results saved in: {self.results_dir}")
        print(f"🎯 Final Test Accuracy: {accuracy:.4f}")

def main():
    """Main function to run model testing."""
    tester = ModelTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 