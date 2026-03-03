import pandas as pd
import numpy as np
from pathlib import Path
import json
import zipfile
from datetime import datetime
import warnings
import argparse
import sys

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           roc_auc_score, precision_recall_curve, f1_score,
                           precision_score, recall_score, precision_recall_fscore_support,
                           roc_curve)


class PlateauMetrics:
    """Метрики с учетом сегментной структуры плато"""

    @staticmethod
    def calculate_segment_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """IoU на уровне сегментов"""
        true_segments = PlateauMetrics._find_segments(y_true)
        pred_segments = PlateauMetrics._find_segments(y_pred)

        if len(true_segments) == 0 and len(pred_segments) == 0:
            return 1.0
        if len(true_segments) == 0 or len(pred_segments) == 0:
            return 0.0

        intersections = 0
        for t_start, t_end in true_segments:
            best_iou = 0
            for p_start, p_end in pred_segments:
                inter_start = max(t_start, p_start)
                inter_end = min(t_end, p_end)
                if inter_end > inter_start:
                    inter_len = inter_end - inter_start
                    union_len = max(t_end, p_end) - min(t_start, p_start)
                    iou = inter_len / union_len
                    best_iou = max(best_iou, iou)
            intersections += best_iou

        return intersections / len(true_segments) if len(true_segments) > 0 else 0.0

    @staticmethod
    def _find_segments(y: np.ndarray):
        """Находит непрерывные сегменты [start, end) где y==1"""
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
    def plateau_count_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


class ResultsVisualizer:
    """Визуализация результатов с сохранением в папку модели"""

    def __init__(self, models_dir="./models_v2"):
        """
        Args:
            models_dir: Папка с моделью (содержит best_model_metadata.json и test_predictions.csv)
        """
        self.input_dir = Path(models_dir)

        # Создаем уникальную папку для визуализаций внутри папки модели
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.input_dir / f"visualizations_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")

        self._load_data()

    def _load_data(self):
        """Загрузка данных из папки модели"""
        metadata_path = self.input_dir / 'best_model_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        csv_path = self.input_dir / 'test_predictions.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Predictions not found: {csv_path}")

        df = pd.read_csv(csv_path)
        self.y_test = df['true_plateau'].values
        self.y_pred = df['predicted_plateau'].values
        self.y_proba = df['probability'].values
        self.test_features = df[['experiment', 'time', 'temp']].copy()

        self.prediction_threshold = self.metadata.get('threshold', 0.5)
        self.is_pytorch = not self.metadata.get('is_sklearn', True)
        self.plateau_emphasis = self.metadata.get('config', {}).get('plateau_emphasis', 3.0)

        self.metrics = {
            'f1': f1_score(self.y_test, self.y_pred),
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_proba)
        }

        self.segment_metrics = PlateauMetrics.plateau_count_error(self.y_test, self.y_pred)
        self.segment_iou = PlateauMetrics.calculate_segment_iou(self.y_test, self.y_pred)

        print(f"Loaded: F1={self.metrics['f1']:.4f}, IoU={self.segment_iou:.4f}")

    def create_all_visualizations(self):
        """Создание всех графиков"""
        print("\nCreating visualizations...")

        self.create_enhanced_timeseries()
        self.create_performance_summary()
        self.create_confusion_matrix()
        self.create_precision_recall_curve()
        self.create_roc_curve()
        self.create_plateau_detail_view()
        self.create_segment_analysis()

    def archive_results(self):
        """Создание ZIP архива папки с визуализациями"""
        archive_path = self.output_dir.with_suffix('.zip')
        
        print(f"\nCreating archive: {archive_path.name}")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_STORED) as zipf:
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.output_dir)
                    zipf.write(file_path, arcname)
        
        print(f"Archive created: {archive_path}")
        return archive_path

    def create_enhanced_timeseries(self):
        """Упрощенный временной ряд"""
        experiments = self.test_features['experiment'].unique()
        fig, axes = plt.subplots(len(experiments), 1, figsize=(16, 6*len(experiments)))
        
        if len(experiments) == 1:
            axes = [axes]

        colors = {0: '#3498db', 1: '#e74c3c'}
        labels = {0: 'Not Plateau', 1: 'Plateau'}

        for i, exp in enumerate(experiments):
            exp_mask = self.test_features['experiment'] == exp
            exp_data = self.test_features[exp_mask].reset_index(drop=True)
            
            y_true = pd.Series(self.y_test[exp_mask]).reset_index(drop=True)
            y_pred = pd.Series(self.y_pred[exp_mask]).reset_index(drop=True)
            y_proba = pd.Series(self.y_proba[exp_mask]).reset_index(drop=True)

            ax = axes[i]
            ax.set_ylim(-0.3, 1.8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Not Plateau (0)', 'Plateau (1)'])
            
            ax2 = ax.twinx()
            
            ax2.plot(exp_data['time'], exp_data['temp'], 'k-', alpha=0.7, label='Temperature')

            for zone in [0, 1]:
                mask = y_true == zone
                if np.any(mask):
                    ax.scatter(exp_data['time'][mask], [zone]*np.sum(mask), 
                              c=colors[zone], alpha=0.6, s=60, marker='o', label=f'Actual {labels[zone]}')

            offset = 0.1
            for zone in [0, 1]:
                mask = y_pred == zone
                if np.any(mask):
                    ax.scatter(exp_data['time'][mask], [zone + offset]*np.sum(mask), 
                              c=colors[zone], alpha=0.9, s=80, marker='^', label=f'Predicted {labels[zone]}')

            ax.plot(exp_data['time'], y_proba, 'g--', alpha=0.8, linewidth=2, label='Probability')
            ax.axhline(y=self.prediction_threshold, color='gray', linestyle=':', alpha=0.7)

            ax.set_title(f'Experiment: {exp} (IoU: {self.segment_iou:.3f})')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels1 + labels2, lines1 + lines2))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_timeseries_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: enhanced_timeseries_binary.png")

    def create_plateau_detail_view(self):
        """ДЕТАЛЬНЫЙ ПРОСМОТР"""
        print("Creating plateau detail view...")

        y_pred_series = pd.Series(self.y_pred).reset_index(drop=True)
        y_test_series = pd.Series(self.y_test).reset_index(drop=True)
        y_proba_series = pd.Series(self.y_proba).reset_index(drop=True)
        test_features = self.test_features.reset_index(drop=True)

        plateau_indices = np.where(y_pred_series == 1)[0]

        if len(plateau_indices) == 0:
            print("No predicted plateaus found, skipping detail view")
            return

        first_plateau = plateau_indices[0]
        last_plateau = plateau_indices[-1]

        context = 20
        start_idx = max(0, first_plateau - context)
        end_idx = min(len(y_pred_series), last_plateau + context)

        time_segment = test_features.iloc[start_idx:end_idx]['time'].reset_index(drop=True)
        temp_segment = test_features.iloc[start_idx:end_idx]['temp'].reset_index(drop=True)
        actual_segment = y_test_series.iloc[start_idx:end_idx].reset_index(drop=True)
        pred_segment = y_pred_series.iloc[start_idx:end_idx].reset_index(drop=True)
        proba_segment = y_proba_series.iloc[start_idx:end_idx].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(16, 6))

        colors = {0: '#3498db', 1: '#e74c3c'}
        labels = {0: 'Not Plateau', 1: 'Plateau'}

        ax.set_ylim(-0.3, 1.8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Not Plateau (0)', 'Plateau (1)'])
        ax.set_ylabel('Zone Classification', fontsize=12, fontweight='bold')

        ax2 = ax.twinx()
        ax2.set_ylabel('Temperature (°C)', fontsize=12, color='black', fontweight='bold')

        ax2.plot(time_segment, temp_segment, color='black',
                linewidth=2.5, alpha=0.7, label='Temperature', zorder=1)

        plateau_times_start = test_features.iloc[first_plateau]['time']
        plateau_times_end = test_features.iloc[last_plateau]['time']
        ax.axvspan(plateau_times_start, plateau_times_end,
                  alpha=0.25, color='gold', zorder=0, label='Predicted Plateau Zone')

        for zone in [0, 1]:
            zone_mask = actual_segment == zone
            if np.any(zone_mask):
                ax.scatter(time_segment[zone_mask],
                          [zone]*sum(zone_mask),
                          c=colors[zone], alpha=0.7, s=100, marker='o',
                          label=f'Actual {labels[zone]}', zorder=3)

        offset = 0.12
        for zone in [0, 1]:
            zone_mask = pred_segment == zone
            if np.any(zone_mask):
                ax.scatter(time_segment[zone_mask],
                          [zone + offset]*sum(zone_mask),
                          c=colors[zone], alpha=0.95, s=120, marker='^',
                          label=f'Predicted {labels[zone]}', zorder=4)

        ax.plot(time_segment, proba_segment, 'g--', alpha=0.8,
               linewidth=2.5, label='Plateau Probability', zorder=2)
        ax.axhline(y=self.prediction_threshold, color='gray',
                  linestyle=':', alpha=0.7, linewidth=2,
                  label=f'Threshold ({self.prediction_threshold})', zorder=1)

        exp_name = test_features.iloc[first_plateau]['experiment']
        duration = plateau_times_end - plateau_times_start
        num_points = last_plateau - first_plateau + 1

        ax.set_title(f'Detailed View. Test Data - Experiment {exp_name}\n'
                    f'From first to last Predicted Plateau '
                    f'({plateau_times_start:.1f}s to {plateau_times_end:.1f}s, '
                    f'{num_points} points, Δ{duration:.1f}s)',
                    fontsize=13, fontweight='bold', pad=10)

        ax.set_xlabel('Time', fontsize=12, fontweight='bold')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels1 + labels2, lines1 + lines2))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left',
                 fontsize=9, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plateau_detail_zoom_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: plateau_detail_zoom_binary.png")

    def create_performance_summary(self):
        """Графики precision/recall/f1"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, self.y_pred, zero_division=0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        classes = ['Not Plateau', 'Plateau']
        colors = ['#3498db', '#e74c3c']

        metrics = [precision[:2], recall[:2], f1[:2]]
        titles = ['Precision', 'Recall', 'F1-Score']

        for ax, metric, title in zip(axes, metrics, titles):
            ax.bar(classes, metric, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(f'{title} by Class', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1.1)
            for i, v in enumerate(metric):
                ax.text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: performance_summary_binary.png")

    def create_confusion_matrix(self):
        """Матрица ошибок"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Plateau', 'Plateau'],
                   yticklabels=['Not Plateau', 'Plateau'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix (IoU: {self.segment_iou:.3f})', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if total > 0:
                    percentage = cm[i,j]/total*100
                    plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)',
                            ha='center', va='center', fontsize=10, color='red', fontweight='bold')

        plt.savefig(self.output_dir / 'confusion_matrix_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: confusion_matrix_binary.png")

    def create_precision_recall_curve(self):
        """PR кривая"""
        precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, self.y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, 'purple', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (AUPRC: {np.mean(precision_vals):.3f})')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: precision_recall_curve.png")

    def create_roc_curve(self):
        """ROC кривая"""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: roc_curve.png")

    def create_segment_analysis(self):
        """Анализ сегментов плато"""
        true_segs = PlateauMetrics._find_segments(self.y_test)
        pred_segs = PlateauMetrics._find_segments(self.y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Количество плато
        ax = axes[0]
        categories = ['True Plateaus', 'Predicted Plateaus']
        counts = [len(true_segs), len(pred_segs)]
        colors = ['#27ae60', '#e67e22']
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Count')
        ax.set_title('Plateau Count Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Распределение длин
        ax = axes[1]
        true_lengths = [end - start for start, end in true_segs] if true_segs else [0]
        pred_lengths = [end - start for start, end in pred_segs] if pred_segs else [0]

        bins = np.linspace(0, max(max(true_lengths, default=10), max(pred_lengths, default=10)), 15)
        ax.hist(true_lengths, bins=bins, alpha=0.6, label=f'True (μ={np.mean(true_lengths):.1f})',
               color='#27ae60', edgecolor='black')
        ax.hist(pred_lengths, bins=bins, alpha=0.6, label=f'Predicted (μ={np.mean(pred_lengths):.1f})',
               color='#e67e22', edgecolor='black')
        ax.set_xlabel('Plateau Length (points)')
        ax.set_ylabel('Frequency')
        ax.set_title('Plateau Length Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'segment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: segment_analysis.png")


def visualize_models(models_dir="./models_v2"):
    """Основная функция для создания визуализаций в папке модели"""
    print(f"Processing: {models_dir}")
    
    viz = ResultsVisualizer(models_dir=models_dir)
    viz.create_all_visualizations()
    archive_path = viz.archive_results()
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"  Folder: {viz.output_dir}")
    print(f"  Archive: {archive_path}")
    print(f"{'='*60}")


def main():
    """Точка входа с поддержкой аргументов командной строки и Jupyter/Colab"""
    parser = argparse.ArgumentParser(description='Визуализация результатов Plateau Detection')
    parser.add_argument('models_dir', nargs='?', type=str, default='./models_v2',
                       help='Путь к папке с моделью (default: ./models_v2)')
    
    # FIX для Jupyter/Colab: используем parse_known_args вместо parse_args
    # чтобы игнорировать системные аргументы вроде -f
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Note: Ignoring unknown args: {unknown}")
    
    visualize_models(args.models_dir)


if __name__ == "__main__":

    main()
