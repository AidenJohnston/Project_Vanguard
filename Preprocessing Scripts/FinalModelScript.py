import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
import joblib
import matplotlib.pyplot as plt

# this was on my personal pc directories don't matter
output_dir = r"C:\Users\Aiden Johnston\Desktop\Project Vanguard"
model_dir = os.path.join(output_dir, "model")
os.makedirs(model_dir, exist_ok=True)

file_path = os.path.join(output_dir, "final_dataset.csv")
df = pd.read_csv(file_path)

if 'Score' in df.columns:
    y = df['Score'].values
    X = df.drop('Score', axis=1).values
else:
    print("Error: 'Score' column not found in the DataFrame. Please check your data.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pk1'))

#model architecture
def build_model(input_dim, l2_lambda=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', 
                              input_shape=(input_dim,),
                              kernel_regularizer=regularizers.l2(l2_lambda)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_lambda)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_lambda)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mse', 'mae'])
    return model


#training settings
epochs = 1000 
batch_size = 16
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# k-fold cause google said it was good
num_bins = 5
y_binned = pd.qcut(y_train, q=num_bins, labels=False)
skf = StratifiedKFold(n_splits=num_bins, shuffle=True, random_state=42)
fold_no = 1
cv_mse_scores = []
cv_mae_scores = []

for train_index, val_index in skf.split(X_train_scaled, y_binned):
    print(f"Starting fold {fold_no}...")
    X_fold_train, X_fold_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    
    model = build_model(input_dim=X_train_scaled.shape[1], l2_lambda=0.001)
    history_cv = model.fit(X_fold_train, y_fold_train,
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(X_fold_val, y_fold_val),
                           callbacks=[early_stop, lr_reduction],
                           verbose=0)
    
    scores = model.evaluate(X_fold_val, y_fold_val, verbose=0)
    print(f"Fold {fold_no} - Validation Loss (MSE): {scores[0]:.4f}, MSE: {scores[1]:.4f}, MAE: {scores[2]:.4f}\n")
    cv_mse_scores.append(scores[1])
    cv_mae_scores.append(scores[2])
    fold_no += 1


print("K-Fold Cross Validation Results:")
print(f"Average Validation MSE over {num_bins} folds: {np.mean(cv_mse_scores):.4f}")
print(f"Average Validation MAE over {num_bins} folds: {np.mean(cv_mae_scores):.4f}\n")

final_model = build_model(input_dim=X_train_scaled.shape[1], l2_lambda=0.001)
history = final_model.fit(X_train_scaled, y_train, 
                          epochs=epochs, 
                          batch_size=batch_size, 
                          validation_split=0.1,
                          callbacks=[early_stop, lr_reduction],
                          verbose=1)

print("Final Training Metrics on full training set:")
print("Loss (Huber):", history.history['loss'][-1])
print("Training MSE:", history.history['mse'][-1])
print("Training MAE:", history.history['mae'][-1])

test_metrics = final_model.evaluate(X_test_scaled, y_test, verbose=0)
print("\nTest Set Metrics:")
print("Test Loss (Huber):", test_metrics[0])
print("Test MSE:", test_metrics[1])
print("Test MAE:", test_metrics[2])

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Final Model Loss over Epochs')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('Final Model MAE over Epochs')
plt.show()

model_save_path = os.path.join(output_dir, "Model.keras")
final_model.save(model_save_path)
predictions = final_model.predict(X_test_scaled)
print("\nFirst 10 Predictions:")
for i in range(10):
    print(f"Predicted: {predictions[i][0]:.2f}, Actual: {y_test[i]:.2f}")
