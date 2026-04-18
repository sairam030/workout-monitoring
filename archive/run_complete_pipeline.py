#!/usr/bin/env python3
"""
Complete Pipeline Runner
========================

Runs the entire workout monitoring pipeline end-to-end:
1. Generate synthetic pose data
2. Feature engineering
3. Data balancing
4. Model training
5. Evaluation

This demonstrates the complete multimedia analytics workflow.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, '.')

def print_step(step_num, total_steps, title):
    """Print formatted step header"""
    print("\n" + "="*70)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("="*70)

def main():
    """Run complete pipeline"""
    
    print("\n" + "="*70)
    print("WORKOUT MONITORING SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis will run the entire multimedia analytics pipeline:")
    print("  1. Data Generation")
    print("  2. Feature Engineering")
    print("  3. Data Balancing (SMOTE)")
    print("  4. Model Training (LSTM, RF, SVM)")
    print("  5. Evaluation & Comparison")
    print("\nEstimated time: 5-10 minutes")
    print("-"*70)
    
    input("\nPress Enter to start...")
    
    start_time = time.time()
    
    # Step 1: Generate Data
    print_step(1, 5, "Data Generation")
    print("Generating realistic pose sequences...")
    
    import generate_realistic_pose_data
    if not Path('data/processed').exists() or len(list(Path('data/processed').glob('*.npz'))) < 20:
        generate_realistic_pose_data.generate_complete_dataset()
    else:
        print("✓ Data already exists, skipping generation")
    
    # Step 2: Feature Engineering
    print_step(2, 5, "Feature Engineering")
    print("Extracting biomechanical features...")
    
    from src.preprocessing.feature_engineering import FeatureEngineer
    import numpy as np
    
    engineer = FeatureEngineer()
    os.makedirs('data/features', exist_ok=True)
    
    # Load and process all sequences
    data_files = sorted(Path('data/processed').glob('*.npz'))
    features_list = []
    labels_exercise = []
    labels_form = []
    
    print(f"Processing {len(data_files)} sequences...")
    for i, filepath in enumerate(data_files, 1):
        data = np.load(filepath, allow_pickle=True)
        landmarks = data['landmarks']
        
        # Handle metadata (might be array or dict)
        if 'metadata' in data:
            metadata = data['metadata']
            if isinstance(metadata, np.ndarray):
                metadata = metadata.item()
            exercise = metadata.get('exercise', 'unknown')
            form = metadata.get('form', 'correct')
        else:
            # Parse from filename: exercise_form_###.npz
            parts = filepath.stem.split('_')
            exercise = parts[0] if len(parts) > 0 else 'unknown'
            form = parts[1] if len(parts) > 1 else 'correct'
        
        # Engineer features
        features = engineer.engineer_features(landmarks)
        
        features_list.append(features)
        labels_exercise.append(exercise)
        labels_form.append(form)
        
        if i % 5 == 0:
            print(f"  Processed {i}/{len(data_files)}...")
    
    # Save features
    features_array = np.array(features_list)
    np.savez_compressed(
        'data/features/engineered_features.npz',
        features=features_array,
        labels_exercise=labels_exercise,
        labels_form=labels_form
    )
    
    print(f"✓ Saved {len(features_list)} feature vectors ({features_array.shape[1]} features each)")
    
    # Step 3: Data Balancing
    print_step(3, 5, "Data Balancing")
    print("Applying SMOTE to handle class imbalance...")
    
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels
    le_exercise = LabelEncoder()
    le_form = LabelEncoder()
    
    y_exercise = le_exercise.fit_transform(labels_exercise)
    y_form = le_form.fit_transform(labels_form)
    
    # Combine exercise and form into single label
    y_combined = y_exercise * 2 + y_form  # Unique label per (exercise, form) pair
    
    print(f"Original class distribution:")
    unique, counts = np.unique(y_combined, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples")
    
    # Train-test split BEFORE balancing
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\n✓ Balanced training set:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples")
    
    print(f"\nTest set (unchanged): {len(y_test)} samples")
    
    # Save balanced data
    os.makedirs('data/balanced', exist_ok=True)
    np.savez_compressed(
        'data/balanced/train_balanced.npz',
        X=X_train_balanced,
        y=y_train_balanced
    )
    np.savez_compressed(
        'data/balanced/test.npz',
        X=X_test,
        y=y_test
    )
    
    # Step 4: Model Training
    print_step(4, 5, "Model Training")
    print("Training multiple models...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    
    models = {}
    
    # Random Forest
    print("\n[1/2] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_balanced, y_train_balanced)
    models['Random Forest'] = rf
    print(f"  ✓ Training accuracy: {accuracy_score(y_train_balanced, rf.predict(X_train_balanced)):.3f}")
    
    # SVM
    print("\n[2/2] Training SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_balanced, y_train_balanced)
    models['SVM'] = svm
    print(f"  ✓ Training accuracy: {accuracy_score(y_train_balanced, svm.predict(X_train_balanced)):.3f}")
    
    # Step 5: Evaluation
    print_step(5, 5, "Model Evaluation")
    print("Evaluating models on test set...")
    
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'predictions': y_pred
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Results Summary:")
    print(f"  - Dataset: {len(features_list)} sequences")
    print(f"  - Features: {features_array.shape[1]} per sequence")
    print(f"  - Training samples (balanced): {len(X_train_balanced)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy:.1%} accuracy)")
    
    print(f"\n📁 Generated Artifacts:")
    print(f"  - data/features/engineered_features.npz")
    print(f"  - data/balanced/train_balanced.npz")
    print(f"  - data/balanced/test.npz")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    
    print(f"\n✅ Next Steps:")
    print(f"  1. Open notebooks/ to explore detailed analysis")
    print(f"  2. Run: jupyter lab notebooks/")
    print(f"  3. Start with 02_eda_detailed.ipynb")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
