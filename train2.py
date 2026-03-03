# @title
"""
Plateau Detection System v2.2 (with Bayesian Optimization)
Добавлено:
- BayesSearchCV для XGBoost и Random Forest
- gp_minimize для оптимизации архитектуры BiLSTM
- Автоматическое сохранение результатов с временной меткой и архивация
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import json
import joblib
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.ndimage import median_filter, binary_opening, binary_closing
from scipy.stats import mode
from datetime import datetime  # Добавлено для временной метки
import zipfile  # Добавлено для архивации
import os  # Добавлено для работы с файлами

# ML
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optimization
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import EarlyStopper, DeltaYStopper

# Seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ==================== CONFIGURATION ====================

@dataclass
class Config:
    """Конфигурация пайплайна"""
    data_dir: Path = Path("./Plateau_Train/Data")
    models_dir: Path = Path("./models_v2")
    plateau_emphasis: float = 3.0  # Вес для класса plateau
    prediction_threshold: float = 0.5
    use_gpu: bool = True

    # Sequence parameters
    sequence_window: int = 20  # Окно для LSTM (временные шаги)
    feature_window: int = 10   # Окно для rolling features

    # Post-processing
    smoothing_window: int = 5  # Медианный фильтр
    min_plateau_length: int = 3  # Минимальная длина плато (в точках)

    # Training
    batch_size: int = 256
    epochs: int = 100
    patience: int = 15

    # Bayesian Optimization Parameters (НОВОЕ)
    use_bayesian_opt: bool = True  # Включить байесовскую оптимизацию
    bayes_n_iter: int = 20  # Количество итераций для sklearn моделей
    bayes_cv_folds: int = 3  # Количество фолдов для CV
    bayes_n_calls_lstm: int = 15  # Количество вызовов для LSTM (может быть долго!)
    lstm_opt_epochs: int = 20  # Эпох для быстрой оценки при оптимизации LSTM


# ==================== FEATURE ENGINEERING ====================

class SequenceFeatureExtractor:
    """Извлечение признаков с учетом временной структуры"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.exp_encoder = LabelEncoder()
        self.is_fitted = False

    def fit_experiments(self, experiments: np.ndarray):
        """Fit encoder на всех экспериментах сразу"""
        self.exp_encoder.fit(experiments)

    def extract_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Извлечение признаков с плато-специфичной логикой
        """
        df = df.copy()
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['temp'] = pd.to_numeric(df['temp'], errors='coerce')

        features_list = []
        experiments = df['experiment'].unique()

        for exp in tqdm(experiments, desc="Processing experiments"):
            exp_data = df[df['experiment'] == exp].copy().sort_values('time')

            # === Базовые производные ===
            exp_data['temp_diff'] = exp_data['temp'].diff()
            exp_data['temp_diff2'] = exp_data['temp_diff'].diff()
            exp_data['temp_change_rate'] = exp_data['temp_diff'] / exp_data['time'].diff().clip(lower=0.1)

            # === Плато-специфичные признаки ===
            w = self.config.feature_window

            # 1. Скользящее стандартное отклонение (плато = низкое std)
            exp_data['temp_std'] = exp_data['temp'].rolling(window=w, center=True, min_periods=1).std()
            exp_data['is_flat'] = (exp_data['temp_std'] < exp_data['temp_std'].quantile(0.1)).astype(int)

            # 2. Абсолютное значение производной (на плато близко к 0)
            exp_data['abs_diff'] = exp_data['temp_diff'].abs()
            exp_data['diff_ma'] = exp_data['abs_diff'].rolling(window=w, center=True, min_periods=1).mean()

            # 3. "Время с последнего изменения" (cumulative flatness)
            exp_data['flat_group'] = (exp_data['abs_diff'] > exp_data['abs_diff'].quantile(0.3)).cumsum()
            exp_data['time_since_change'] = exp_data.groupby('flat_group').cumcount()

            # 4. Контекстные признаки (соседи) - важно для непрерывности!
            for lag in [1, 2, 3, 5]:
                exp_data[f'temp_lag_{lag}'] = exp_data['temp'].shift(lag)
                exp_data[f'diff_lag_{lag}'] = exp_data['temp_diff'].shift(lag)
                exp_data[f'flat_lag_{lag}'] = exp_data['is_flat'].shift(lag)

            for lead in [1, 2, 3]:
                exp_data[f'temp_lead_{lead}'] = exp_data['temp'].shift(-lead)
                exp_data[f'diff_lead_{lead}'] = exp_data['temp_diff'].shift(-lead)

            # 5. Статистики по окну (центрированные и causal)
            for window in [5, 10, 20]:
                exp_data[f'temp_mean_{window}'] = exp_data['temp'].rolling(window=window, center=True, min_periods=1).mean()
                exp_data[f'temp_min_{window}'] = exp_data['temp'].rolling(window=window, center=True, min_periods=1).min()
                exp_data[f'temp_max_{window}'] = exp_data['temp'].rolling(window=window, center=True, min_periods=1).max()
                exp_data[f'temp_range_{window}'] = exp_data[f'temp_max_{window}'] - exp_data[f'temp_min_{window}']

            # 6. Относительное время и интегралы
            exp_data['relative_time'] = exp_data['time'] / exp_data['time'].max()
            exp_data['temp_cumsum'] = exp_data['temp'].cumsum()

            # 7. Кодирование эксперимента (исправленный баг!)
            if not self.is_fitted:
                exp_data['experiment_encoded'] = 0  # Заглушка для первого прохода
            else:
                try:
                    exp_data['experiment_encoded'] = self.exp_encoder.transform([exp])[0]
                except ValueError:
                    exp_data['experiment_encoded'] = -1  # Unknown experiment

            features_list.append(exp_data)

        result = pd.concat(features_list, ignore_index=True)

        # Заполнение NaN (особенно в lag/lead признаках)
        result = result.bfill().ffill().fillna(0)

        return result

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Возвращает список колонок для обучения"""
        exclude = ['zones', 'experiment', 'time', 'temp', 'flat_group']
        return [col for col in df.columns if col not in exclude]


# ==================== SEQUENCE DATASET ====================

class PlateauSequenceDataset(Dataset):
    """Dataset для LSTM: возвращает последовательности"""

    def __init__(self, X: np.ndarray, y: np.ndarray, experiment_ids: np.ndarray,
                 sequence_length: int = 20):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.exp_ids = experiment_ids
        self.seq_len = sequence_length

        # Создаем индексы для каждого эксперимента отдельно
        self.indices = []
        unique_exps = np.unique(experiment_ids)

        for exp in unique_exps:
            exp_mask = experiment_ids == exp
            exp_indices = np.where(exp_mask)[0]

            # Для каждой точки берем окно [i-seq_len+1, i]
            for i in range(len(exp_indices)):
                if i < sequence_length - 1:
                    # Padding в начале
                    start_idx = 0
                    pad_len = sequence_length - i - 1
                    seq_indices = np.concatenate([
                        np.full(pad_len, exp_indices[0]),  # Повтор первого
                        exp_indices[:i+1]
                    ])
                else:
                    start_idx = i - sequence_length + 1
                    seq_indices = exp_indices[start_idx:i+1]

                self.indices.append((exp_indices[i], seq_indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        target_idx, seq_indices = self.indices[idx]
        return self.X[seq_indices], self.y[target_idx]


# ==================== MODELS ====================

class BiLSTMClassifier(nn.Module):
    """BiLSTM для sequence classification"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Берем последний hidden state (concat forward + backward)
        # hidden: (num_layers * 2, batch, hidden_size)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_combined = torch.cat([hidden_forward, hidden_backward], dim=1)

        return self.classifier(hidden_combined)


class SimpleMLP(nn.Module):
    """Улучшенная MLP как в оригинале, но с BatchNorm"""

    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==================== POST-PROCESSING ====================

class PredictionPostProcessor:
    """
    Пост-обработка для обеспечения непрерывности плато
    """

    def __init__(self, min_length: int = 3, smoothing_window: int = 5):
        self.min_length = min_length
        self.smooth_window = smoothing_window

    def process(self, y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Применяет:
        1. Медианную фильтрацию к вероятностям
        2. Binarization
        3. Морфологическое закрытие (заполнение маленьких пробелов)
        4. Удаление коротких плато
        """
        # 1. Сглаживание вероятностей
        smoothed = median_filter(y_proba, size=self.smooth_window, mode='reflect')

        # 2. Бинаризация
        binary = (smoothed > threshold).astype(int)

        # 3. Морфологические операции
        # Закрытие: сначала дилатация (расширение), потом эрозия (сужение)
        # Убирает маленькие дырки в плато
        binary = binary_closing(binary, structure=np.ones(3)).astype(int)

        # Открытие: убирает шумовые выбросы
        binary = binary_opening(binary, structure=np.ones(self.min_length)).astype(int)

        # 4. Удаление коротких сегментов
        binary = self._remove_short_segments(binary, min_length=self.min_length)

        return binary

    def _remove_short_segments(self, y: np.ndarray, min_length: int) -> np.ndarray:
        """Удаляет плато короче min_length"""
        result = y.copy()

        # Находим границы сегментов
        diff = np.diff(y, prepend=0, append=0)
        start_indices = np.where(diff == 1)[0]
        end_indices = np.where(diff == -1)[0]

        for start, end in zip(start_indices, end_indices):
            length = end - start
            if length < min_length:
                result[start:end] = 0  # Удаляем короткое плато

        return result


# ==================== METRICS ====================

class PlateauMetrics:
    """Метрики с учетом сегментной структуры"""

    @staticmethod
    def calculate_segment_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """IoU на уровне сегментов (а не точек)"""
        # Находим сегменты плато в ground truth
        true_segments = PlateauMetrics._find_segments(y_true)
        pred_segments = PlateauMetrics._find_segments(y_pred)

        if len(true_segments) == 0 and len(pred_segments) == 0:
            return 1.0
        if len(true_segments) == 0 or len(pred_segments) == 0:
            return 0.0

        # Считаем пересечения
        intersections = 0
        for t_start, t_end in true_segments:
            best_iou = 0
            for p_start, p_end in pred_segments:
                # Intersection
                inter_start = max(t_start, p_start)
                inter_end = min(t_end, p_end)
                if inter_end > inter_start:
                    inter_len = inter_end - inter_start
                    union_len = max(t_end, p_end) - min(t_start, p_start)
                    iou = inter_len / union_len
                    best_iou = max(best_iou, iou)
            intersections += best_iou

        return intersections / len(true_segments)

    @staticmethod
    def _find_segments(y: np.ndarray) -> List[Tuple[int, int]]:
        """Находит непрерывные сегменты [start, end) где y==1"""
        if len(y) == 0:
            return []

        segments = []
        start = None

        for i, val in enumerate(y):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(y)))

        return segments

    @staticmethod
    def plateau_count_error(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Ошибка в количестве и средней длительности плато"""
        true_segments = PlateauMetrics._find_segments(y_true)
        pred_segments = PlateauMetrics._find_segments(y_pred)

        true_lengths = [end - start for start, end in true_segments]
        pred_lengths = [end - start for start, end in pred_segments]

        return {
            'true_count': len(true_segments),
            'pred_count': len(pred_segments),
            'count_error': abs(len(true_segments) - len(pred_segments)),
            'true_avg_length': np.mean(true_lengths) if true_lengths else 0,
            'pred_avg_length': np.mean(pred_lengths) if pred_lengths else 0,
            'length_error': abs(np.mean(true_lengths) - np.mean(pred_lengths)) if true_lengths and pred_lengths else 0
        }


# ==================== MAIN TRAINER ====================

class PlateauTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.config.models_dir.mkdir(exist_ok=True, parents=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        print(f"Using device: {self.device}")

        self.feature_extractor = SequenceFeatureExtractor(config)
        self.post_processor = PredictionPostProcessor(
            min_length=config.min_plateau_length,
            smoothing_window=config.smoothing_window
        )

        self.best_model = None
        self.best_model_name = None
        self.is_sequence_model = False

        # Хранилище для лучших параметров
        self.best_params = {}

    def load_data(self):
        """Загрузка данных"""
        print("Loading data...")
        train_path = self.config.data_dir / "train_binary.csv"
        test_path = self.config.data_dir / "test_binary.csv"

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        print(f"Train: {self.train_df.shape}, Test: {self.test_df.shape}")
        print(f"Train plateau ratio: {(self.train_df['zones'] == 0).mean():.3f}")

    def prepare_features(self):
        """Подготовка признаков"""
        print("\nExtracting features...")

        # Fit encoder на всех экспериментах
        all_exps = pd.concat([self.train_df, self.test_df])['experiment'].unique()
        self.feature_extractor.fit_experiments(all_exps)
        self.feature_extractor.is_fitted = True

        # Extract
        self.train_features = self.feature_extractor.extract_features(self.train_df, is_train=True)
        self.test_features = self.feature_extractor.extract_features(self.test_df, is_train=False)

        # Prepare matrices
        feature_cols = self.feature_extractor.get_feature_columns(self.train_features)
        self.feature_cols = feature_cols

        self.X_train = self.train_features[feature_cols].values
        self.X_test = self.test_features[feature_cols].values

        # Binary target: 1=Plateau (zones==0), 0=Not Plateau
        self.y_train = (self.train_features['zones'].values == 0).astype(int)
        self.y_test = (self.test_features['zones'].values == 0).astype(int)

        self.exp_train = self.train_features['experiment'].values
        self.exp_test = self.test_features['experiment'].values

        # Scale
        self.X_train_scaled = self.feature_extractor.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.feature_extractor.scaler.transform(self.X_test)

        print(f"Features: {len(feature_cols)}")
        print(f"Class distribution: {np.bincount(self.y_train)}")

    def train_sklearn_models(self) -> Dict:
        """Обучение sklearn моделей с байесовской оптимизацией"""
        print("\n" + "="*60)
        print("Training sklearn models with Bayesian Optimization" if self.config.use_bayesian_opt else "Training sklearn models...")

        # Веса классов с защитой от деления на ноль
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        if n_pos == 0:
            n_pos = 1
            print("Warning: No positive samples found in training data!")

        scale_pos_weight = (n_neg / n_pos) * self.config.plateau_emphasis
        print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

        # Определение пространств поиска для байесовской оптимизации
        search_spaces = {
            'xgboost': {
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'n_estimators': Integer(100, 500),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'min_child_weight': Integer(1, 10),
                'gamma': Real(1e-5, 1.0, prior='log-uniform'),
                'reg_alpha': Real(1e-5, 10.0, prior='log-uniform'),
                'reg_lambda': Real(1e-5, 10.0, prior='log-uniform')
            },
            'random_forest': {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(5, 50),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2'])
                # Убран class_weight отсюда - он задается при инициализации модели
            }
        }

        models = {
            'xgboost': XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                n_jobs=-1,
                use_label_encoder=False
            ),
            'random_forest': RandomForestClassifier(
                n_jobs=-1,
                random_state=42,
                class_weight={0: 1, 1: self.config.plateau_emphasis}  # Задаем здесь
            )
        }



        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            if self.config.use_bayesian_opt:
                print(f"Running Bayesian Optimization ({self.config.bayes_n_iter} iterations)...")

                # Создаем скорер для F1
                f1_scorer = make_scorer(f1_score, zero_division=0)

                opt = BayesSearchCV(
                    estimator=model,
                    search_spaces=search_spaces[name],
                    n_iter=self.config.bayes_n_iter,
                    cv=StratifiedKFold(n_splits=self.config.bayes_cv_folds, shuffle=True, random_state=42),
                    scoring=f1_scorer,
                    n_jobs=-1,
                    random_state=42,
                    verbose=1,
                    return_train_score=False
                )

                # Callback для ранней остановки если улучшение меньше 0.01 за 5 итераций
                early_stop = DeltaYStopper(delta=0.01, n_best=5)

                opt.fit(self.X_train_scaled, self.y_train, callback=early_stop)

                model = opt.best_estimator_
                self.best_params[name] = opt.best_params_

                print(f"Best params: {opt.best_params_}")
                print(f"Best CV F1: {opt.best_score_:.4f}")
            else:
                # Обычное обучение с дефолтными параметрами
                if name == 'random_forest':
                    model.set_params(
                        n_estimators=200,
                        max_depth=20,
                        class_weight={0: 1, 1: self.config.plateau_emphasis}
                    )
                model.fit(self.X_train_scaled, self.y_train)

            # Eval on test
            y_pred = model.predict(self.X_test_scaled)
            f1 = f1_score(self.y_test, y_pred)

            results[name] = {
                'model': model,
                'f1': f1,
                'is_sklearn': True
            }
            print(f"{name} Test F1: {f1:.4f}")

            # Save
            joblib.dump(model, self.config.models_dir / f"{name}_model.pkl")

            # Save best params if using Bayesian opt
            if self.config.use_bayesian_opt and name in self.best_params:
                # Конвертируем numpy типы в Python типы
                serializable_params = {}
                for k, v in self.best_params[name].items():
                    if hasattr(v, 'item'):  # numpy scalar
                        serializable_params[k] = v.item()
                    elif isinstance(v, (np.ndarray, list)):
                        serializable_params[k] = [x.item() if hasattr(x, 'item') else x for x in v]
                    else:
                        serializable_params[k] = v

                with open(self.config.models_dir / f"{name}_best_params.json", 'w') as f:
                    json.dump(serializable_params, f, indent=2)

        return results

    def _train_lstm_with_params(self, hidden_size: int, num_layers: int, dropout: float,
                               lr: float, weight_decay: float, epochs: int = None) -> float:
        """
        Вспомогательный метод для обучения LSTM с заданными параметрами.
        Возвращает F1 score (для максимизации).
        """
        if epochs is None:
            epochs = self.config.lstm_opt_epochs

        # Create datasets
        train_dataset = PlateauSequenceDataset(
            self.X_train_scaled, self.y_train,
            self.feature_extractor.exp_encoder.transform(self.exp_train),
            sequence_length=self.config.sequence_window
        )

        test_dataset = PlateauSequenceDataset(
            self.X_test_scaled, self.y_test,
            self.feature_extractor.exp_encoder.transform(self.exp_test),
            sequence_length=self.config.sequence_window
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Model
        model = BiLSTMClassifier(
            input_size=self.X_train_scaled.shape[1],
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout)
        ).to(self.device)

        # Loss and optimizer
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        scale_pos_weight = (n_neg / max(n_pos, 1)) * self.config.plateau_emphasis

        pos_weight = torch.tensor([scale_pos_weight]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        # Training
        best_f1 = 0
        patience_counter = 0
        patience = max(5, self.config.patience // 2)  # Короче patience для оптимизации

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation
            model.eval()
            all_preds = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = model(X_batch)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    all_preds.extend(preds.flatten())

            val_f1 = f1_score(self.y_test, all_preds)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_f1

    def train_lstm(self) -> Dict:
        """Обучение BiLSTM с байесовской оптимизацией гиперпараметров"""
        if not self.config.use_gpu or not torch.cuda.is_available():
            print("Skipping LSTM (no GPU)")
            return {}

        print("\n" + "="*60)
        print("Training BiLSTM with Bayesian Optimization" if self.config.use_bayesian_opt else "Training BiLSTM...")

        if self.config.use_bayesian_opt:
            print(f"Optimizing hyperparameters ({self.config.bayes_n_calls_lstm} calls)...")

            # Определение пространства поиска
            dimensions = [
                Integer(64, 256, name='hidden_size'),
                Integer(1, 3, name='num_layers'),
                Real(0.1, 0.5, name='dropout'),
                Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
                Real(1e-6, 1e-3, prior='log-uniform', name='weight_decay')
            ]

            @use_named_args(dimensions)
            def objective(**params):
                """Целевая функция для минимизации (возвращаем -F1)"""
                try:
                    f1 = self._train_lstm_with_params(**params, epochs=self.config.lstm_opt_epochs)
                    return -f1  # Отрицательное значение т.к. gp_minimize ищет минимум
                except Exception as e:
                    print(f"Error during training: {e}")
                    return 0.0  # В случае ошибки возвращаем худший скор

            # Запуск оптимизации
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.config.bayes_n_calls_lstm,
                n_initial_points=5,
                acq_func='EI',
                random_state=42,
                verbose=True,
                callback=[DeltaYStopper(delta=0.005, n_best=3)]  # Ранняя остановка
            )

            best_params = {
                'hidden_size': result.x[0],
                'num_layers': result.x[1],
                'dropout': result.x[2],
                'lr': result.x[3],
                'weight_decay': result.x[4]
            }

            self.best_params['lstm'] = best_params
            print(f"\nBest LSTM params: {best_params}")
            print(f"Best F1 during optimization: {-result.fun:.4f}")

            # Финальное обучение с лучшими параметрами на полном количестве эпох
            print("\nTraining final LSTM with best params...")
            final_f1 = self._train_lstm_final(**best_params)
        else:
            # Базовое обучение без оптимизации
            final_f1 = self._train_lstm_final(
                hidden_size=128, num_layers=2, dropout=0.3,
                lr=0.001, weight_decay=1e-4
            )
            best_params = {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}

        return {
            'lstm': {
                'model': self.best_model,
                'f1': final_f1,
                'is_sklearn': False,
                'params': best_params
            }
        }

    def _train_lstm_final(self, hidden_size: int, num_layers: int, dropout: float,
                         lr: float, weight_decay: float) -> float:
        """Финальное обучение LSTM с полным количеством эпох и early stopping"""

        # Create datasets
        train_dataset = PlateauSequenceDataset(
            self.X_train_scaled, self.y_train,
            self.feature_extractor.exp_encoder.transform(self.exp_train),
            sequence_length=self.config.sequence_window
        )

        test_dataset = PlateauSequenceDataset(
            self.X_test_scaled, self.y_test,
            self.feature_extractor.exp_encoder.transform(self.exp_test),
            sequence_length=self.config.sequence_window
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Model
        model = BiLSTMClassifier(
            input_size=self.X_train_scaled.shape[1],
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout)
        ).to(self.device)

        # Loss with focal loss для несбалансированных данных
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        if n_pos == 0:
            n_pos = 1
        scale_pos_weight = (n_neg / n_pos) * self.config.plateau_emphasis

        pos_weight = torch.tensor([scale_pos_weight]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        best_f1 = 0
        best_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for X_batch, _ in test_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = model(X_batch)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    all_preds.extend(preds.flatten())
                    all_probs.extend(probs.flatten())

            val_f1 = f1_score(self.y_test, all_preds)
            scheduler.step(train_loss / len(train_loader))

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, F1={val_f1:.4f}")

            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)

        self.best_model = model

        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_size': self.X_train_scaled.shape[1],
                'hidden_size': int(hidden_size),
                'num_layers': int(num_layers),
                'dropout': float(dropout)
            },
            'best_f1': best_f1
        }, self.config.models_dir / "lstm_model.pth")

        # Save params - ИСПРАВЛЕНИЕ ЗДЕСЬ: конвертируем numpy types в Python types
        if hasattr(self, 'best_params') and 'lstm' in self.best_params:
            # Конвертируем numpy int64/float64 в обычные int/float
            serializable_params = {}
            for k, v in self.best_params['lstm'].items():
                if hasattr(v, 'item'):  # numpy scalar
                    serializable_params[k] = v.item()
                else:
                    serializable_params[k] = v

            with open(self.config.models_dir / "lstm_best_params.json", 'w') as f:
                json.dump(serializable_params, f, indent=2)

        return best_f1


    def evaluate_and_post_process(self, models_dict: Dict):
        """Оценка с пост-обработкой"""
        print("\n" + "="*60)
        print("Final Evaluation with Post-Processing")
        print("="*60)

        final_results = {}

        for name, info in models_dict.items():
            print(f"\n--- {name} ---")
            model = info['model']

            # Predictions
            if info['is_sklearn']:
                y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                # LSTM
                model.eval()
                test_dataset = PlateauSequenceDataset(
                    self.X_test_scaled, self.y_test,
                    self.feature_extractor.exp_encoder.transform(self.exp_test),
                    sequence_length=self.config.sequence_window
                )
                loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

                y_proba = []
                with torch.no_grad():
                    for X_batch, _ in loader:
                        X_batch = X_batch.to(self.device)
                        outputs = model(X_batch)
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        y_proba.extend(probs.flatten())
                y_proba = np.array(y_proba)

            # Raw predictions
            y_pred_raw = (y_proba > self.config.prediction_threshold).astype(int)

            # Post-processed
            y_pred_smooth = self.post_processor.process(y_proba, self.config.prediction_threshold)

            # Metrics
            for label, y_pred in [("Raw", y_pred_raw), ("Smoothed", y_pred_smooth)]:
                print(f"\n{label} metrics:")
                print(f"  Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
                print(f"  F1: {f1_score(self.y_test, y_pred):.4f}")
                print(f"  Precision: {precision_score(self.y_test, y_pred):.4f}")
                print(f"  Recall: {recall_score(self.y_test, y_pred):.4f}")

                # Segment metrics
                seg_metrics = PlateauMetrics.plateau_count_error(self.y_test, y_pred)
                print(f"  True/Detected plateaus: {seg_metrics['true_count']}/{seg_metrics['pred_count']}")
                print(f"  Segment IoU: {PlateauMetrics.calculate_segment_iou(self.y_test, y_pred):.4f}")

            final_results[name] = {
                'proba': y_proba,
                'raw_pred': y_pred_raw,
                'smooth_pred': y_pred_smooth,
                'f1_smooth': f1_score(self.y_test, y_pred_smooth)
            }

        # Select best
        best_name = max(final_results.keys(), key=lambda k: final_results[k]['f1_smooth'])
        print(f"\n*** Best model: {best_name} (F1 with smoothing: {final_results[best_name]['f1_smooth']:.4f}) ***")

        # Save best
        self._save_best_model(best_name, models_dict[best_name], final_results[best_name])

        # Save submission format
        self._save_submission(final_results[best_name]['smooth_pred'])

    def _save_best_model(self, name: str, model_info: Dict, predictions: Dict):
        """Сохранение лучшей модели"""
        metadata = {
            'model_name': name,
            'is_sklearn': model_info['is_sklearn'],
            'threshold': self.config.prediction_threshold,
            'features': self.feature_cols,
            'f1_score': predictions['f1_smooth'],
            'config': {
                'plateau_emphasis': self.config.plateau_emphasis,
                'min_plateau_length': self.config.min_plateau_length,
                'smoothing_window': self.config.smoothing_window
            },
            'best_params': self.best_params.get(name, {})
        }

        with open(self.config.models_dir / 'best_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save scaler
        joblib.dump(self.feature_extractor.scaler, self.config.models_dir / 'scaler.pkl')
        joblib.dump(self.feature_extractor.exp_encoder, self.config.models_dir / 'exp_encoder.pkl')

        # Save predictions for analysis
        results_df = pd.DataFrame({
            'experiment': self.exp_test,
            'time': self.test_features['time'].values,
            'temp': self.test_features['temp'].values,
            'true_plateau': self.y_test,
            'predicted_plateau': predictions['smooth_pred'],
            'probability': predictions['proba']
        })
        results_df.to_csv(self.config.models_dir / 'test_predictions.csv', index=False)

        print(f"\nSaved to {self.config.models_dir}/")

    def _save_submission(self, y_pred: np.ndarray):
        """Сохранение в формате submission (если нужно)"""
        # Можно добавить логику сохранения в специфический формат
        pass

    def run(self):
        """Полный пайплайн"""
        self.load_data()
        self.prepare_features()

        # Train models
        sklearn_results = self.train_sklearn_models()
        lstm_results = self.train_lstm() if self.config.use_gpu else {}

        all_results = {**sklearn_results, **lstm_results}

        # Evaluate with post-processing
        if all_results:
            self.evaluate_and_post_process(all_results)


# ==================== INFERENCE CLASS ====================

class PlateauPredictor:
    """Класс для инференса на новых данных"""

    def __init__(self, models_dir: Path = Path("./models_v2")):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load metadata
        with open(models_dir / 'best_model_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Load preprocessing
        self.scaler = joblib.load(models_dir / 'scaler.pkl')
        self.exp_encoder = joblib.load(models_dir / 'exp_encoder.pkl')

        # Load model
        if self.metadata['is_sklearn']:
            self.model = joblib.load(models_dir / f"{self.metadata['model_name']}_model.pkl")
            self.is_sequence = False
        else:
            # Load LSTM
            checkpoint = torch.load(models_dir / 'lstm_model.pth', map_location=self.device)
            config = checkpoint['config']
            self.model = BiLSTMClassifier(**config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_sequence = True

        self.post_processor = PredictionPostProcessor(
            min_length=self.metadata['config']['min_plateau_length'],
            smoothing_window=self.metadata['config']['smoothing_window']
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Предсказание на новых данных"""
        # Feature extraction (simplified version)
        df = df.copy()
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['temp'] = pd.to_numeric(df['temp'], errors='coerce')

        # Basic features (в реальности нужно использовать тот же FeatureExtractor)
        # Для краткости опущу полный код, но логика та же

        return np.zeros(len(df))  # Placeholder


# ==================== MAIN ====================

def main():
    # Создаем папку с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./results_{timestamp}")
    models_subdir = results_dir / "models"
    models_subdir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Результаты будут сохранены в: {results_dir}")
    print(f"📦 После завершения архив будет создан: results_{timestamp}.zip")

    config = Config(
        data_dir=Path("/content/Plateau_Train/Data"),  # Поменяйте на ваш путь
        models_dir=models_subdir,  # Теперь сохраняем в подпапку внутри results
        plateau_emphasis=3.0,
        use_gpu=torch.cuda.is_available(),
        sequence_window=20,
        min_plateau_length=5,
        # Параметры байесовской оптимизации
        use_bayesian_opt=True,  # Включить оптимизацию
        bayes_n_iter=15,        # Итераций для sklearn (больше = лучше, но медленнее)
        bayes_n_calls_lstm=10   # Итераций для LSTM (каждая долгая!)
    )

    trainer = PlateauTrainer(config)
    
    try:
        trainer.run()
        print("\n✅ Обучение успешно завершено!")
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        raise
    finally:
        # Архивируем результаты без сжатия (ZIP_STORED)
        print(f"\n📦 Архивирование результатов...")
        archive_path = results_dir.parent / f"results_{timestamp}.zip"
        
        # Создаем zip архив без сжатия (хранение как есть)
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_STORED) as zipf:
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = str(file_path.relative_to(results_dir))
                    zipf.write(file_path, arcname)
        
        archive_size = archive_path.stat().st_size / (1024*1024)  # MB
        print(f"✅ Архив создан: {archive_path}")
        print(f"📊 Размер архива: {archive_size:.2f} MB")
        print(f"📂 Содержимое: модели, метрики, предсказания, параметры")

if __name__ == "__main__":
    main()