"""
Evaluation and Performance Analysis Module
==========================================

Comprehensive model evaluation:
- Classification metrics
- Confusion matrix
- Per-class analysis
- ROC curves
- Feature importance
- Error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd


class ModelEvaluator:
    """
    Comprehensive model evaluation suite
    
    Why comprehensive evaluation?
    - Single metrics (accuracy) insufficient
    - Need per-class performance
    - Identify error patterns
    - Guide improvement efforts
    """
    
    def __init__(self, class_names=None):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names (e.g., ['Squat', 'Pushup', 'Curl'])
        """
        self.class_names = class_names or ['Class 0', 'Class 1', 'Class 2']
        self.n_classes = len(self.class_names)
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate all classification metrics
        
        Metrics:
        - Accuracy: Overall correctness
        - Precision: Of predicted positives, how many correct?
        - Recall: Of actual positives, how many found?
        - F1-Score: Harmonic mean of precision and recall
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC)
            
        Returns:
            dict: All metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics (macro average)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted average (accounts for class imbalance)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(10, 8), save_path=None):
        """
        Plot confusion matrix
        
        Why confusion matrix?
        - Shows error patterns
        - Which classes are confused?
        - Diagonal = correct predictions
        - Off-diagonal = errors
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
        
        # Print normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized Confusion Matrix (Row = True Class):")
        print(pd.DataFrame(
            cm_normalized,
            index=self.class_names,
            columns=self.class_names
        ).round(3))
    
    def plot_classification_report(self, y_true, y_pred, figsize=(10, 6), save_path=None):
        """
        Visualize classification report as heatmap
        
        Shows:
        - Precision, Recall, F1 per class
        - Support (number of samples)
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Extract metrics for each class
        metrics_df = pd.DataFrame({
            class_name: {
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1-Score': report[class_name]['f1-score'],
                'Support': report[class_name]['support']
            }
            for class_name in self.class_names
        }).T
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap of metrics
        sns.heatmap(
            metrics_df[['Precision', 'Recall', 'F1-Score']],
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ax1,
            cbar_kws={'label': 'Score'}
        )
        ax1.set_title('Per-Class Performance Metrics', fontweight='bold')
        ax1.set_ylabel('Exercise Type')
        
        # Support bar chart
        ax2.bar(metrics_df.index, metrics_df['Support'])
        ax2.set_title('Class Support', fontweight='bold')
        ax2.set_xlabel('Exercise Type')
        ax2.set_ylabel('Number of Samples')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Classification report saved to {save_path}")
        
        plt.show()
        
        # Print detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
    
    def plot_roc_curves(self, y_true, y_prob, figsize=(10, 8), save_path=None):
        """
        Plot ROC curves for multi-class classification
        
        Why ROC curves?
        - Threshold-independent evaluation
        - Shows true positive vs false positive trade-off
        - AUC = Area Under Curve (0.5-1.0, higher is better)
        
        Args:
            y_true: True labels (integers)
            y_prob: Predicted probabilities (n_samples, n_classes)
        """
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        plt.figure(figsize=figsize)
        
        # Calculate ROC curve and AUC for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curves saved to {save_path}")
        
        plt.show()
    
    def analyze_errors(self, y_true, y_pred, confidence=None, top_k=10):
        """
        Analyze prediction errors
        
        Identifies:
        - Most common misclassifications
        - Low confidence errors
        - Systematic error patterns
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidence: Prediction confidence scores (optional)
            top_k: Number of error patterns to show
        """
        # Find errors
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*60}")
        print(f"Total samples: {len(y_true)}")
        print(f"Errors: {len(error_indices)} ({len(error_indices)/len(y_true)*100:.1f}%)")
        print(f"{'='*60}\n")
        
        if len(error_indices) == 0:
            print("✓ No errors! Perfect predictions.")
            return
        
        # Error patterns (true → predicted)
        error_patterns = {}
        for idx in error_indices:
            pattern = (self.class_names[y_true[idx]], self.class_names[y_pred[idx]])
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        # Sort by frequency
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top {min(top_k, len(sorted_patterns))} Error Patterns:")
        print(f"{'True Class':<15} → {'Predicted':<15} {'Count':<10} {'% of Errors'}")
        print("-" * 60)
        
        for (true_class, pred_class), count in sorted_patterns[:top_k]:
            pct = count / len(error_indices) * 100
            print(f"{true_class:<15} → {pred_class:<15} {count:<10} {pct:.1f}%")
        
        # Confidence analysis (if provided)
        if confidence is not None:
            print(f"\n{'='*60}")
            print("CONFIDENCE ANALYSIS")
            print(f"{'='*60}")
            
            # Average confidence for correct vs incorrect
            correct_conf = confidence[~errors].mean()
            error_conf = confidence[errors].mean()
            
            print(f"Average confidence (correct): {correct_conf:.3f}")
            print(f"Average confidence (errors): {error_conf:.3f}")
            print(f"Difference: {correct_conf - error_conf:.3f}")
            
            # Low confidence errors
            if len(error_indices) > 0:
                error_confidences = confidence[error_indices]
                low_conf_threshold = 0.6
                low_conf_errors = np.sum(error_confidences < low_conf_threshold)
                
                print(f"\nLow confidence errors (< {low_conf_threshold}): {low_conf_errors}")
                print(f"High confidence errors (≥ {low_conf_threshold}): {len(error_indices) - low_conf_errors}")
                print("\n⚠️ High confidence errors indicate systematic issues!")
    
    def compare_models(self, results_dict):
        """
        Compare multiple models
        
        Args:
            results_dict: {
                'Model Name': {
                    'accuracy': float,
                    'f1_macro': float,
                    'inference_time': float
                },
                ...
            }
        """
        df = pd.DataFrame(results_dict).T
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(df.round(4))
        print("="*60)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Accuracy
        df['accuracy'].plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Accuracy Comparison', fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim([0, 1])
        axes[0].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # F1 Score
        df['f1_macro'].plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('F1-Score Comparison', fontweight='bold')
        axes[1].set_ylabel('F1-Score (Macro)')
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        
        # Inference Time
        if 'inference_time' in df.columns:
            df['inference_time'].plot(kind='bar', ax=axes[2], color='salmon')
            axes[2].set_title('Inference Time', fontweight='bold')
            axes[2].set_ylabel('Time per Sample (ms)')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


def print_summary(metrics, model_name="Model"):
    """
    Print formatted metrics summary
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print("\n" + "="*60)
    print(f"{model_name} - PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision_macro']:.4f} (macro)")
    print(f"Recall:         {metrics['recall_macro']:.4f} (macro)")
    print(f"F1-Score:       {metrics['f1_macro']:.4f} (macro)")
    print("="*60)
    
    # Performance rating
    acc = metrics['accuracy']
    if acc >= 0.95:
        rating = "🌟 EXCELLENT"
    elif acc >= 0.90:
        rating = "✅ GOOD"
    elif acc >= 0.85:
        rating = "⚠️ ACCEPTABLE"
    else:
        rating = "❌ NEEDS IMPROVEMENT"
    
    print(f"Overall Rating: {rating}")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("Model Evaluation Module")
    print("="*60)
    
    # Create dummy predictions
    np.random.seed(42)
    n_samples = 300
    n_classes = 3
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    
    # Introduce some errors (10% error rate)
    error_indices = np.random.choice(n_samples, size=30, replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, 30)
    
    # Random probabilities
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names=['Squat', 'Push-up', 'Curl'])
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
    print_summary(metrics, "Example Model")
    
    # Visualizations
    print("\nGenerating visualizations...\n")
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_classification_report(y_true, y_pred)
    evaluator.plot_roc_curves(y_true, y_prob)
    
    # Error analysis
    confidence = y_prob.max(axis=1)
    evaluator.analyze_errors(y_true, y_pred, confidence)
    
    print("\n✓ Evaluation module ready!")
