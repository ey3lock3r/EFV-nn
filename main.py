import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

from efv_nn import EFVClassifier, evaluate_models, run_ablation, plot_efv_results

def main():
    print("="*65)
    print("  ENERGY-FREQUENCY-VIBRATION (EFV) LEARNING ALGORITHM")
    print("  'Think in terms of energy, frequency and vibration' — Tesla")
    print("="*65)
    
    # Load datasets
    datasets = {}
    for loader, name in [(load_iris, 'Iris (4D, 3-class)'),
                          (load_wine, 'Wine (13D, 3-class)'),
                          (load_breast_cancer, 'Cancer (30D, 2-class)'),
                          (load_digits, 'Digits (64D, 10-class)')]:
        d = loader()
        datasets[name] = (d.data, d.target)
    
    base_params = dict(n_frequencies=200, frequency_scales=[0.1, 0.5, 1.0, 2.0, 5.0],
                       n_epochs=150, learning_rate=0.5, vibration_amplitude=0.3,
                       vibration_periods=4, vibration_noise_init=0.05,
                       vibration_noise_decay=0.97, l2_reg=0.0001, random_state=42)

    classifiers = {
        'EFV (Ours)': EFVClassifier(**base_params),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Reg.': LogisticRegression(max_iter=1000, random_state=42),
        'MLP (2-layer)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Naive Bayes': GaussianNB(),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = evaluate_models(datasets, classifiers, cv=cv)
    
    # Ablation
    print(f"\n{'─'*65}")
    print(f"  ABLATION STUDY (Wine)")
    print(f"{'─'*65}")
    
    wine_X, wine_y = datasets['Wine (13D, 3-class)']
    ablation_configs = {
        'Full EFV': base_params,
        'No Vibration': {**base_params, 'vibration_amplitude': 0.0, 'vibration_noise_init': 0.0},
        'No Multi-Scale': {**base_params, 'frequency_scales': [1.0]},
        'No Fourier (raw only)': {**base_params, 'n_frequencies': 5, 'frequency_scales': [0.0001]},
        'High Vibration': {**base_params, 'vibration_amplitude': 0.8, 'vibration_noise_init': 0.2},
    }
    
    ablation_results = run_ablation(wine_X, wine_y, EFVClassifier, ablation_configs, cv=cv)
    
    # Visualization
    clf_vis = EFVClassifier(**base_params)
    clf_vis.fit(wine_X, wine_y)
    plot_efv_results(clf_vis, results, ablation_results)

if __name__ == "__main__":
    main()
