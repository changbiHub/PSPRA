import tensorflow as tf
from tensorflow.keras import Input, layers, Model
from tcn import TCN
from sklearn.utils import class_weight
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

class NNmodel():
    def __init__(self, time_steps=63, model_name="TCN", best_model_filepath="best_model.keras", num_features=1):
        self.time_steps = time_steps
        self.model_name = model_name
        if num_features==1:
            if model_name=="TCN":
                self.model, self.feature_extractor = TCNmodel(time_steps)
            elif model_name=="RNN":
                self.model, self.feature_extractor = RNNmodel(time_steps)
            else:
                raise ValueError("Unsupported model name. Choose 'TCN' or 'RNN'.")
        else:
            if model_name=="TCN":
                self.model, self.feature_extractor = TCNmodel_multivariate(num_features, time_steps)
            elif model_name=="RNN":
                self.model, self.feature_extractor = RNNmodel_multivariate(num_features, time_steps)
            else:
                raise ValueError("Unsupported model name. Choose 'TCN' or 'RNN'.")
        
        self.model.compile(optimizer = tf.keras.optimizers.AdamW(), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model_filepath,
            monitor='val_AUC',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
    
    def fit(self, X_train, y_train, epochs=200, batch_size=32, validation_split=0.1):
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        self.model.fit(X_train,y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=validation_split,
                       callbacks=[self.checkpoint],
                       class_weight=class_weight_dict,
                       verbose=2)
        
        # Only load weights if the checkpoint file exists
        if os.path.exists(self.checkpoint.filepath):
            self.model.load_weights(self.checkpoint.filepath)
        else:
            print(f"Warning: Checkpoint file {self.checkpoint.filepath} not found. Using final model weights.")

        baseline_prob_train = self.model.predict(X_train).flatten()
        fpr, tpr, thresholds = roc_curve(y_train,baseline_prob_train)
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        self.th = thresholds[ix]

    
    def predict(self, X):
        y_pred_raw = self.model.predict(X).flatten()
        return (y_pred_raw>=self.th).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict(X).flatten()
    
    def get_features(self, X):
        return self.feature_extractor.predict(X)