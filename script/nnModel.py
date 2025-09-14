import tensorflow as tf
from tensorflow.keras import Input, layers, Model
from tcn import TCN
from sklearn.utils import class_weight
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from sklearn.metrics import roc_curve
import os

def RNNmodel(time_steps):
    input_ts = Input(shape=(time_steps,1))
    x = layers.LSTM(3,dropout=0.3)(input_ts)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(input_ts, output), Model(input_ts,x)

def TCNmodel(time_steps):
    input_ts = Input(shape=(time_steps,1))
    x = TCN(dropout_rate=0.3)(input_ts)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(input_ts, output), Model(input_ts,x)

def RNNmodel_multivariate(num_features, time_steps=63):
    input_ts = Input(shape=(time_steps,num_features))
    x = layers.LSTM(3,dropout=0.5)(input_ts)
    x = layers.Flatten()(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(input_ts, output), Model(input_ts,x)

def TCNmodel_multivariate(num_features, time_steps=63):
    input_ts = Input(shape=(time_steps,num_features))
    x = TCN(dropout_rate=0.5)(input_ts)
    x = layers.Flatten()(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model(input_ts, output), Model(input_ts,x)

class NNmodel(BaseEstimator, ClassifierMixin):
    def __init__(self, time_steps=63, model_name="TCN", best_model_filepath="best_model.keras", num_features=1):
        self.time_steps = time_steps
        self.model_name = model_name
        self.best_model_filepath = best_model_filepath
        self.num_features = num_features
        # Initialize internal attributes
        self.model = None
        self.feature_extractor = None
        self.checkpoint = None
        self.th = None
        self.classes_ = None
        self.is_fitted_ = False
        
    def _build_model(self):
        """Build the neural network model."""
        if self.num_features == 1:
            if self.model_name == "TCN":
                self.model, self.feature_extractor = TCNmodel(self.time_steps)
            elif self.model_name == "RNN":
                self.model, self.feature_extractor = RNNmodel(self.time_steps)
            else:
                raise ValueError("Unsupported model name. Choose 'TCN' or 'RNN'.")
        else:
            if self.model_name == "TCN":
                self.model, self.feature_extractor = TCNmodel_multivariate(self.num_features, self.time_steps)
            elif self.model_name == "RNN":
                self.model, self.feature_extractor = RNNmodel_multivariate(self.num_features, self.time_steps)
            else:
                raise ValueError("Unsupported model name. Choose 'TCN' or 'RNN'.")
        
        self.model.compile(optimizer=tf.keras.optimizers.AdamW(), 
                          loss='binary_crossentropy', 
                          metrics=['accuracy', 'AUC'])
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.best_model_filepath,
            monitor='val_AUC',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
    
    def fit(self, X, y, epochs=200, batch_size=32, validation_split=0.1):
        """Fit the neural network model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, time_steps, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        epochs : int, default=200
            Number of training epochs.
        batch_size : int, default=32
            Batch size for training.
        validation_split : float, default=0.1
            Fraction of training data to use for validation.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X, y = check_X_y(X, y, allow_nd=True)
        
        # Store classes
        self.classes_ = np.unique(y)
        
        # Build model if not already built
        if self.model is None:
            self._build_model()
        
        # Compute class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Train the model
        self.model.fit(X, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=validation_split,
                       callbacks=[self.checkpoint],
                       class_weight=class_weight_dict,
                       verbose=2)
        
        # Load best weights if available
        if os.path.exists(self.checkpoint.filepath):
            self.model.load_weights(self.checkpoint.filepath)
        else:
            print(f"Warning: Checkpoint file {self.checkpoint.filepath} not found. Using final model weights.")

        # Compute optimal threshold using geometric mean
        baseline_prob_train = self.model.predict(X).flatten()
        fpr, tpr, thresholds = roc_curve(y, baseline_prob_train)
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        self.th = thresholds[ix]
        
        # Mark as fitted
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, time_steps, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, 'is_fitted_')
        
        # Validate input
        X = check_array(X, allow_nd=True)
        
        # Predict probabilities and apply threshold
        y_pred_raw = self.model.predict(X).flatten()
        return (y_pred_raw >= self.th).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, time_steps, n_features)
            Input data.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, 2)
            Class probabilities for each sample.
        """
        # Check if fitted
        check_is_fitted(self, 'is_fitted_')
        
        # Validate input
        X = check_array(X, allow_nd=True)
        
        # Get probabilities for positive class
        proba_pos = self.model.predict(X).flatten()
        proba_neg = 1 - proba_pos
        
        # Return probabilities for both classes
        return np.column_stack([proba_neg, proba_pos])
    
    def decision_function(self, X):
        """Predict confidence scores for samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, time_steps, n_features)
            Input data.
            
        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Confidence scores per sample.
        """
        # Check if fitted
        check_is_fitted(self, 'is_fitted_')
        
        # Validate input
        X = check_array(X, allow_nd=True)
        
        return self.model.predict(X).flatten()
    
    def get_features(self, X):
        """Extract features using the feature extractor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, time_steps, n_features)
            Input data.
            
        Returns
        -------
        features : ndarray
            Extracted features.
        """
        # Check if fitted
        check_is_fitted(self, 'is_fitted_')
        
        # Validate input
        X = check_array(X, allow_nd=True)
        
        return self.feature_extractor.predict(X)
    
    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, time_steps, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'time_steps': self.time_steps,
            'model_name': self.model_name,
            'best_model_filepath': self.best_model_filepath,
            'num_features': self.num_features
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        
        # Reset fitted state when parameters change
        self.is_fitted_ = False
        self.model = None
        self.feature_extractor = None
        self.checkpoint = None
        self.th = None
        self.classes_ = None
        
        return self
    
# Simple transformer to reshape data for neural networks
class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_steps, num_features):
        self.time_steps = time_steps
        self.num_features = num_features
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if len(X.shape) == 2:
            return X.reshape(X.shape[0], self.time_steps, self.num_features)
        return X
        
    def get_params(self, deep=True):
        return {'time_steps': self.time_steps, 'num_features': self.num_features}
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self