
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
from tqdm import tqdm

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class PlateauClassifier:
    """
    A comprehensive class for plateau detection in time-temperature data.
    
    This classifier implements multiple machine learning algorithms with
    Bayesian optimization for hyperparameter tuning.
    """
    
    def __init__(self, data_dir="./Data", models_dir="./models"):
        """Initialize the classifier with data and models directories."""
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)  # Create models directory if it doesn't exist
        self.train_data = None
        self.test_data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.models_performance = {}
        
    def load_data(self):
        """Load training and test data from CSV files."""
        print("Loading data...")
        self.train_data = pd.read_csv(self.data_dir / "train_raw.csv")
        self.test_data = pd.read_csv(self.data_dir / "test_raw.csv")
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Unique experiments in training: {self.train_data['experiment'].nunique()}")
        print(f"Zone distribution in training:")
        print(self.train_data['zones'].value_counts())
        
    def engineer_features(self, df):
        """
        Engineer features from time-temperature data.
        
        Features created:
        1. Temperature derivatives (1st and 2nd order)
        2. Moving averages and standard deviations
        3. Temperature change rates
        4. Relative position in experiment
        5. Cumulative statistics
        """
        print("Engineering features...")
        
        features_list = []
        experiments = df['experiment'].unique()
        
        # Use tqdm to show progress for feature engineering
        for experiment in tqdm(experiments, desc="Processing experiments"):
            exp_data = df[df['experiment'] == experiment].copy()
            
            # Ensure numeric types for time and temp columns
            exp_data['time'] = pd.to_numeric(exp_data['time'], errors='coerce')
            exp_data['temp'] = pd.to_numeric(exp_data['temp'], errors='coerce')
            if 'zones' in exp_data.columns:
                exp_data['zones'] = pd.to_numeric(exp_data['zones'], errors='coerce')
            
            exp_data = exp_data.sort_values('time').reset_index(drop=True)
            
            # Basic features
            exp_data['temp_diff'] = exp_data['temp'].diff()
            exp_data['temp_diff2'] = exp_data['temp_diff'].diff()
            
            # Moving averages
            for window in [5, 10, 20]:
                exp_data[f'temp_ma_{window}'] = exp_data['temp'].rolling(window=window, center=True).mean()
                exp_data[f'temp_std_{window}'] = exp_data['temp'].rolling(window=window, center=True).std()
                exp_data[f'temp_diff_ma_{window}'] = exp_data['temp_diff'].rolling(window=window, center=True).mean()
            
            # Temperature change rate
            exp_data['temp_change_rate'] = exp_data['temp_diff'] / exp_data['time'].diff()
            
            # Relative position in experiment
            exp_data['relative_time'] = exp_data['time'] / exp_data['time'].max()
            
            # Cumulative features
            exp_data['cumsum_temp'] = exp_data['temp'].cumsum()
            exp_data['cumsum_temp_diff'] = exp_data['temp_diff'].cumsum()
            
            # Temperature acceleration
            exp_data['temp_acceleration'] = exp_data['temp_change_rate'].diff()
            
            # Local extrema indicators
            exp_data['is_local_max'] = (
                (exp_data['temp'].shift(1) < exp_data['temp']) & 
                (exp_data['temp'].shift(-1) < exp_data['temp'])
            ).astype(int)
            
            exp_data['is_local_min'] = (
                (exp_data['temp'].shift(1) > exp_data['temp']) & 
                (exp_data['temp'].shift(-1) > exp_data['temp'])
            ).astype(int)
            
            # Encode experiment type
            exp_data['experiment_encoded'] = self.label_encoder.fit_transform([experiment] * len(exp_data))
            
            features_list.append(exp_data)
        
        features_df = pd.concat(features_list, ignore_index=True)
        
        # Fill NaN values with appropriate methods
        features_df = features_df.bfill().ffill().fillna(0)
        
        return features_df
    
    def prepare_features(self):
        """Prepare feature matrices for training."""
        print("Preparing features...")
        
        # Engineer features for training data
        train_features = self.engineer_features(self.train_data)
        
        # Define feature columns (exclude target and identifiers)
        feature_cols = [col for col in train_features.columns 
                       if col not in ['zones', 'experiment', 'time']]
        
        self.features = feature_cols
        self.X_train = train_features[feature_cols]
        self.y_train = train_features['zones']
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        print(f"Number of features: {len(feature_cols)}")
        print(f"Training samples: {len(self.X_train)}")
        
    def get_bayesian_search_spaces(self):
        """Define Bayesian optimization search spaces for different models."""
        # Check if GPU is available for XGBoost
        import platform
        is_mac = platform.system() == 'Darwin'
        
        search_spaces = {
            'random_forest': {
                'n_estimators': Integer(50, 500),
                'max_depth': Integer(5, 50),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', None]),
                'bootstrap': Categorical([True, False])
            },
            'xgboost': {
                'n_estimators': Integer(50, 500),
                'max_depth': Integer(3, 20),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
                'gamma': Real(0, 10),
                'reg_alpha': Real(0, 10),
                'reg_lambda': Real(0, 10),
                # Add GPU support if available
                'tree_method': Categorical(['hist', 'gpu_hist']) if not is_mac else Categorical(['hist']),
                'predictor': Categorical(['cpu_predictor', 'gpu_predictor']) if not is_mac else Categorical(['cpu_predictor'])
            },
            'svm': {
                'C': Real(0.1, 100, prior='log-uniform'),
                'gamma': Real(0.0001, 1, prior='log-uniform'),
                'kernel': Categorical(['rbf', 'poly', 'sigmoid'])
            },
            'neural_network': {
                'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                'activation': Categorical(['relu', 'tanh']),
                'solver': Categorical(['adam', 'lbfgs']),
                'alpha': Real(0.0001, 0.1, prior='log-uniform'),
                'learning_rate': Categorical(['constant', 'adaptive'])
            }
        }
        return search_spaces
    
    def train_model_with_bayesian_optimization(self, model_name, model_class, search_space, n_iter=50):
        """
        Train a model using Bayesian optimization for hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            model_class: Model class to instantiate
            search_space: Hyperparameter search space
            n_iter: Number of Bayesian optimization iterations
            
        Returns:
            Best model and its performance metrics
        """
        print(f"\nTraining {model_name} with Bayesian Optimization...")
        print(f"Running {n_iter} iterations with 5-fold cross-validation...")
        
        # Initialize model
        if model_name == 'neural_network':
            base_model = model_class(random_state=42, max_iter=1000, early_stopping=True)
        elif model_name == 'xgboost':
            # Enable verbosity for XGBoost
            base_model = model_class(random_state=42, verbosity=1)
        else:
            base_model = model_class(random_state=42)
        
        # Custom callback to show progress
        class ProgressCallback:
            def __init__(self, n_iter):
                self.n_iter = n_iter
                self.pbar = tqdm(total=n_iter, desc=f"Optimizing {model_name}")
                self.current_iter = 0
                
            def __call__(self, res):
                self.current_iter += 1
                self.pbar.update(1)
                self.pbar.set_postfix({
                    'best_score': f"{res.fun:.4f}" if hasattr(res, 'fun') else "N/A"
                })
                
        progress_callback = ProgressCallback(n_iter)
        
        # Bayesian search with verbose output
        bayes_search = BayesSearchCV(
            base_model,
            search_space,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1,  # Show progress
            n_points=1,  # Evaluate one point at a time for better progress tracking
            error_score='raise'
        )
        
        # Fit the model with progress tracking
        print(f"Starting optimization for {model_name}...")
        
        # Wrap the training data with tqdm for feature engineering progress
        try:
            bayes_search.fit(self.X_train_scaled, self.y_train)
        except Exception as e:
            print(f"Error during training: {e}")
            raise
        finally:
            if hasattr(progress_callback, 'pbar'):
                progress_callback.pbar.close()
        
        # Get best model and parameters
        best_model = bayes_search.best_estimator_
        best_params = bayes_search.best_params_
        best_score = bayes_search.best_score_
        
        print(f"\nBest parameters for {model_name}: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        
        # Store performance
        self.models_performance[model_name] = {
            'best_params': best_params,
            'cv_score': best_score,
            'model': best_model
        }
        
        # Save individual model
        import joblib
        model_path = self.models_dir / f"{model_name}_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        return best_model, best_score
    
    def train_all_models(self):
        """Train all models with Bayesian optimization."""
        # Check system capabilities
        import platform
        print("\nSystem Information:")
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python version: {sys.version.split()[0]}")
        
        # Check for GPU availability
        try:
            import xgboost as xgb
            print(f"XGBoost version: {xgb.__version__}")
            # Check if GPU is available for XGBoost
            if hasattr(xgb, 'device'):
                print("XGBoost GPU support: Available")
            else:
                print("XGBoost GPU support: Not available (CPU mode)")
        except:
            pass
        
        models = {
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier,
            'svm': SVC,
            'neural_network': MLPClassifier
        }
        
        search_spaces = self.get_bayesian_search_spaces()
        
        best_score = 0
        best_model_name = None
        
        print(f"\nTraining {len(models)} models with Bayesian optimization...")
        for model_name, model_class in tqdm(models.items(), desc="Training models"):
            model, score = self.train_model_with_bayesian_optimization(
                model_name, 
                model_class, 
                search_spaces[model_name],
                n_iter=30  # Adjust iterations based on computational resources
            )
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                self.best_model = model
        
        print(f"\nBest model: {best_model_name} with CV score: {best_score:.4f}")
        
    def evaluate_on_test_data(self):
        """Evaluate the best model on test data."""
        print("\nEvaluating on test data...")
        
        # Prepare test features
        test_features = self.engineer_features(self.test_data)
        X_test = test_features[self.features]
        y_test = test_features['zones']
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.best_model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Plateau', 'Heating', 'Cooling']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        confusion_matrix_path = self.models_dir / 'confusion_matrix.png'
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"Confusion matrix saved to: {confusion_matrix_path}")
        
        return accuracy
    
    def save_model_comparison(self):
        """Save model comparison results."""
        comparison_df = pd.DataFrame([
            {
                'Model': name,
                'CV Score': perf['cv_score'],
                'Best Parameters': str(perf['best_params'])
            }
            for name, perf in self.models_performance.items()
        ])
        
        comparison_df = comparison_df.sort_values('CV Score', ascending=False)
        comparison_csv_path = self.models_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"\nModel comparison saved to: {comparison_csv_path}")
        
        # Plot model comparison
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_df['Model'], comparison_df['CV Score'])
        plt.ylabel('Cross-Validation Score')
        plt.xlabel('Model')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        comparison_plot_path = self.models_dir / 'model_comparison.png'
        plt.savefig(comparison_plot_path)
        plt.close()
        print(f"Model comparison plot saved to: {comparison_plot_path}")
        
    def run_pipeline(self):
        """Run the complete machine learning pipeline."""
        print("Starting Plateau Classification Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Prepare features
        self.prepare_features()
        
        # Train models with Bayesian optimization
        self.train_all_models()
        
        # Evaluate on test data
        test_accuracy = self.evaluate_on_test_data()
        
        # Save model comparison
        self.save_model_comparison()
        
        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        
        return self.best_model, test_accuracy
    
def main():
    """Main function to run the plateau classification."""
    classifier = PlateauClassifier(data_dir="./Data", models_dir="./models")
    best_model, test_accuracy = classifier.run_pipeline()
    
    # Save the best model and scaler
    import joblib
    best_model_path = classifier.models_dir / 'best_plateau_classifier.pkl'
    scaler_path = classifier.models_dir / 'scaler.pkl'
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(classifier.scaler, scaler_path)
    
    print(f"\nBest model saved to: {best_model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Save model metadata
    metadata = {
        'test_accuracy': test_accuracy,
        'features': classifier.features,
        'models_performance': {k: {
            'cv_score': v['cv_score'],
            'best_params': v['best_params']
        } for k, v in classifier.models_performance.items()}
    }
    
    import json
    metadata_path = classifier.models_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Model metadata saved to: {metadata_path}")
    
if __name__ == "__main__":
    main() 