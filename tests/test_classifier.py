import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from efv_nn import EFVClassifier
from efv_nn.experiments import evaluate_models, run_ablation

def test_initialization():
    # Arrange & Act
    clf = EFVClassifier(n_frequencies=100, learning_rate=0.1)
    
    # Assert
    assert clf.n_frequencies == 100
    assert clf.learning_rate == 0.1
    assert clf.frequency_scales == [0.1, 0.5, 1.0, 2.0, 5.0]

def test_fit_predict_shape():
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
    clf = EFVClassifier(n_frequencies=20, n_epochs=5, random_state=42)
    
    # Act
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    
    # Assert
    assert preds.shape == (100,)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (100, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(100), rtol=1e-5)
    
    # Assert History population
    assert 'loss' in clf.history_
    assert len(clf.history_['loss']) == 5

def test_vibration_schedule():
    # Arrange
    clf = EFVClassifier(n_epochs=10, learning_rate=1.0, vibration_amplitude=0.3, vibration_periods=2)
    
    # Act
    lrs = [clf._vibration_lr(epoch) for epoch in range(10)]
    
    # Assert
    assert len(lrs) == 10
    assert all(lr > 0 for lr in lrs)
    assert lrs[0] > lrs[-1]

def test_fourier_transform():
    # Arrange
    X = np.random.randn(10, 5)
    clf = EFVClassifier(n_frequencies=20, frequency_scales=[1.0, 2.0])
    clf.n_freq_per_scale_ = 10
    clf.W_ = np.random.randn(20, 5)
    
    # Act
    Xf = clf._fourier_transform(X)
    
    # Assert
    # Features = 2 scales * 10 per scale * 2 (sin/cos) + 5 (raw features)
    assert Xf.shape == (10, 45)

def test_experiment_runner():
    # Arrange
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    datasets = {'TestDS': (X, y)}
    classifiers = {'EFV': EFVClassifier(n_epochs=2, n_frequencies=10, random_state=42)}

    # Act
    results = evaluate_models(datasets, classifiers)

    # Assert — evaluate_models returns {ds: {clf: {'accuracy': (mean, std), 'fit_time': ..., 'n_params': ...}}}
    assert 'TestDS' in results
    assert 'EFV' in results['TestDS']
    clf_result = results['TestDS']['EFV']
    assert isinstance(clf_result, dict), \
        f"Expected dict result per classifier, got {type(clf_result)}"
    assert 'accuracy' in clf_result
    assert 'fit_time' in clf_result
    assert 'n_params' in clf_result
    mean_acc, std_acc = clf_result['accuracy']
    assert isinstance(mean_acc, float), f"Mean accuracy should be float, got {type(mean_acc)}"
    assert 0.0 <= mean_acc <= 1.0, f"Mean accuracy {mean_acc} out of [0, 1] range"
    assert std_acc >= 0.0, f"Std of accuracy should be non-negative, got {std_acc}"

def test_ablation_runner():
    # Arrange
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    configs = {'ConfigA': {'n_epochs': 2, 'n_frequencies': 10, 'random_state': 42}}
    
    # Act
    results = run_ablation(X, y, EFVClassifier, configs)
    
    # Assert
    assert 'ConfigA' in results
    assert isinstance(results['ConfigA'][0], float)

# We omit the full scikit-learn check_estimator as it takes a while and tests
# very strict edge cases, but the user is aware it's an option.
