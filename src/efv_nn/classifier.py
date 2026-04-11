import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class EFVClassifier(BaseEstimator, ClassifierMixin):
    """
    Energy-Frequency-Vibration Classifier.
    
    ENERGY:     Cross-entropy as free energy; Boltzmann probability assignment
    FREQUENCY:  Multi-scale Random Fourier Features for spectral representation  
    VIBRATION:  Cosine-annealed learning rate with damped noise (exploration→settling)
    """
    
    def __init__(self, n_frequencies=200, frequency_scales=None,
                 n_epochs=100, learning_rate=0.5,
                 vibration_amplitude=0.3, vibration_periods=4,
                 vibration_noise_init=0.05, vibration_noise_decay=0.97,
                 l2_reg=0.0001, random_state=None):
        self.n_frequencies = n_frequencies
        self.frequency_scales = frequency_scales or [0.1, 0.5, 1.0, 2.0, 5.0]
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.vibration_amplitude = vibration_amplitude
        self.vibration_periods = vibration_periods
        self.vibration_noise_init = vibration_noise_init
        self.vibration_noise_decay = vibration_noise_decay
        self.l2_reg = l2_reg
        self.random_state = random_state
    
    def _fourier_transform(self, X):
        """FREQUENCY: Multi-scale Fourier feature mapping."""
        features = []
        n_per = self.n_freq_per_scale_
        for s_idx, scale in enumerate(self.frequency_scales):
            W = self.W_[s_idx * n_per:(s_idx + 1) * n_per]
            proj = X @ (W * scale).T
            features.append(np.cos(2 * np.pi * proj))
            features.append(np.sin(2 * np.pi * proj))
        # Also include raw (scaled) features for direct linear separation
        features.append(X)
        return np.hstack(features)
    
    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    
    def _vibration_lr(self, epoch):
        """VIBRATION: Cosine annealing with warm restarts."""
        t = epoch / self.n_epochs
        # Cosine annealing component
        cosine = 0.5 * (1 + np.cos(2 * np.pi * self.vibration_periods * t))
        # Damping envelope
        envelope = 1.0 - 0.5 * t  # Linear decay from 1.0 to 0.5
        lr = self.learning_rate * (1.0 - self.vibration_amplitude + 
                                    self.vibration_amplitude * cosine * envelope)
        return max(lr, self.learning_rate * 0.01)
    
    def _vibration_noise(self, epoch):
        """VIBRATION: Damped noise for exploration."""
        return self.vibration_noise_init * (self.vibration_noise_decay ** epoch)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Map y to indices
        y_map = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([y_map[c] for c in y])
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), yi] = 1.0
        
        # Scale
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        
        # FREQUENCY: Initialize Fourier projection
        self.n_freq_per_scale_ = self.n_frequencies // len(self.frequency_scales)
        total = self.n_freq_per_scale_ * len(self.frequency_scales)
        self.W_ = rng.randn(total, n_features)
        
        # Transform
        Xf = self._fourier_transform(Xs)
        d = Xf.shape[1]
        
        # ENERGY: Initialize weights (Xavier)
        scale = np.sqrt(2.0 / (d + n_classes))
        self.theta_ = rng.randn(n_classes, d) * scale
        self.bias_ = np.zeros(n_classes)
        
        # Training
        self.history_ = {'loss': [], 'accuracy': [], 'lr': [], 'noise': []}
        
        for epoch in range(self.n_epochs):
            # Forward
            logits = Xf @ self.theta_.T + self.bias_
            probs = self._softmax(logits)
            
            # ENERGY: Cross-entropy (free energy)
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-10), axis=1))
            loss += 0.5 * self.l2_reg * np.sum(self.theta_ ** 2)
            
            # Gradient
            err = (probs - Y) / n_samples
            grad_w = err.T @ Xf + self.l2_reg * self.theta_
            grad_b = err.sum(axis=0)
            
            # VIBRATION: Adaptive LR + noise
            lr = self._vibration_lr(epoch)
            noise_scale = self._vibration_noise(epoch)
            
            # Update with noise injection
            self.theta_ -= lr * grad_w + noise_scale * rng.randn(*self.theta_.shape)
            self.bias_ -= lr * grad_b
            
            # Metrics
            preds = self.classes_[np.argmax(logits, axis=1)]
            acc = np.mean(preds == y)
            
            self.history_['loss'].append(loss)
            self.history_['accuracy'].append(acc)
            self.history_['lr'].append(lr)
            self.history_['noise'].append(noise_scale)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        Xs = self.scaler_.transform(check_array(X))
        Xf = self._fourier_transform(Xs)
        return self.classes_[np.argmax(Xf @ self.theta_.T + self.bias_, axis=1)]
    
    def predict_proba(self, X):
        check_is_fitted(self)
        Xs = self.scaler_.transform(check_array(X))
        Xf = self._fourier_transform(Xs)
        return self._softmax(Xf @ self.theta_.T + self.bias_)
