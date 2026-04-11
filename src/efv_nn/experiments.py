import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import is_classifier

def count_parameters(clf, X_shape=None, n_classes=None):
    """
    Estimate number of 'trainable' or key structural parameters for various models.
    """
    try:
        if hasattr(clf, 'theta_') and hasattr(clf, 'bias_'): # EFV
            return clf.theta_.size + clf.bias_.size
        if hasattr(clf, 'coef_'): # Linear models, SVM
            params = clf.coef_.size
            if hasattr(clf, 'intercept_'):
                params += clf.intercept_.size
            return params
        if hasattr(clf, 'coefs_'): # MLP
            p = sum(c.size for c in clf.coefs_)
            p += sum(b.size for b in clf.intercepts_)
            return p
        if hasattr(clf, 'estimators_'): # RF, GBT
            try:
                # Handle 1D or 2D array of estimators
                all_estimators = np.array(clf.estimators_).ravel()
                return sum(count_parameters(e) for e in all_estimators)
            except:
                return 0
        if hasattr(clf, 'tree_'): # Decision Tree
            return clf.tree_.node_count
        if hasattr(clf, 'theta_'): # GaussianNB or others
            p = clf.theta_.size
            if hasattr(clf, 'var_'): p += clf.var_.size
            elif hasattr(clf, 'sigma_'): p += clf.sigma_.size
            return p
        if hasattr(clf, 'n_support_'): # SVC (Alternative)
            return clf.support_vectors_.size
        if hasattr(clf, 'n_neighbors'): # KNN
            return 0 # Non-parametric
    except:
        pass
    return 0

def evaluate_models(datasets, classifiers, cv=None):
    """
    Generic experiment runner to compare any set of classifiers on multiple datasets.
    Includes training time and parameter count analysis.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    for ds_name, (X, y) in datasets.items():
        n_classes = len(np.unique(y))
        print(f"\n{'─'*65}")
        print(f"  {ds_name}  (n={X.shape[0]}, features={X.shape[1]}, classes={n_classes})")
        print(f"{'─'*65}")
        print(f"  {'Classifier':18s} | {'Accuracy':18s} | {'Fit Time':10s} | {'Params':10s}")
        print(f"  {'─'*18}─┼─{'─'*18}─┼─{'─'*10}─┼─{'─'*10}")
        
        results[ds_name] = {}
        for clf_name, clf in classifiers.items():
            # Get parameter count (may need a fit first for some sklearn models to reveal structure)
            # We fit on a small subset if not fitted, or just fit once here
            try:
                temp_clf = clone_and_fit(clf, X, y)
                n_params = count_parameters(temp_clf, X.shape, n_classes)
            except:
                n_params = 0

            cv_results = cross_validate(clf, X, y, cv=cv, scoring='accuracy', return_train_score=False)
            
            scores = cv_results['test_score']
            times = cv_results['fit_time']
            
            results[ds_name][clf_name] = {
                'accuracy': (scores.mean(), scores.std()),
                'fit_time': (times.mean(), times.std()),
                'n_params': n_params
            }
            
            print(f"  {clf_name:18s} | {scores.mean():.4f} ± {scores.std():.4f} | {times.mean():.4f}s | {n_params:,}")
            
    return results

def clone_and_fit(clf, X, y):
    from sklearn.base import clone
    c = clone(clf)
    # Shuffle to ensure all classes are present in the small fit
    idx = np.random.choice(len(X), min(len(X), 200), replace=False)
    c.fit(X[idx], y[idx])
    return c

def run_ablation(X, y, classifier_class, ablation_configs, cv=None):
    """
    Run ablation study on varying hyperparameter configurations for an algorithm.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ablation_results = {}
    for cfg_name, params in ablation_configs.items():
        clf = classifier_class(**params)
        cv_results = cross_validate(clf, X, y, cv=cv, scoring='accuracy')
        scores = cv_results['test_score']
        ablation_results[cfg_name] = (scores.mean(), scores.std())
        print(f"  {cfg_name:25s}  {scores.mean():.4f} ± {scores.std():.4f}")
    return ablation_results
