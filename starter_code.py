import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from sklearn.metrics import f1_score, accuracy_score

print("--- Step 1: Loading Data from CSV Files ---")

# 1. Load the 4 CSV files
# Note: These files must be in the same folder as this script
df_train_val = pd.read_csv('.../train_val_labeled.csv')
df_adj = pd.read_csv('.../adjacency_matrix.csv', header=None).values.astype(np.float32)

# --- Step 2: Reconstruct Full Feature & Label Matrices ---
num_nodes = 2708

# Determine number of features (columns starting with 'word_')
feature_columns = [c for c in df_train_val.columns if c.startswith('word_')]
num_features = len(feature_columns)

# Initialize empty matrices
features = np.zeros((num_nodes, num_features), dtype=np.float32)
labels = np.zeros((num_nodes,), dtype=np.int32)

# 1. Fill Train/Val data
tv_ids = df_train_val['id'].values
features[tv_ids] = df_train_val[feature_columns].values
labels[tv_ids] = df_train_val['target'].values


print(f"Reconstructed Features: {features.shape}")
print(f"Reconstructed Labels: {labels.shape}")

# --- Step 3: Normalize Adjacency Matrix ---

def normalize_adj(adj):
    adj = adj + np.eye(len(adj))
    degree = adj.sum(axis=1, keepdims=True)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    adj_normalized = degree_inv_sqrt * adj * degree_inv_sqrt.T
    return adj_normalized

adj_norm = normalize_adj(df_adj)
print("Adjacency matrix normalized successfully!")

# --- Step 4: Define Splits (Standard Cora) ---
train_idx = np.array(range(140), dtype=np.int32)
val_idx = np.array(range(140, 640), dtype=np.int32)
num_classes = len(np.unique(labels[:640])) # Count classes based on Train+Val

# --- Step 5: Model Definition ---

def gcn_op(inputs_list):
    feat, adj = inputs_list[0], inputs_list[1]
    return tf.matmul(adj, feat)

inputs = layers.Input(shape=(num_nodes, num_features), name='node_features')
adj_input = layers.Input(shape=(num_nodes, num_nodes), name='adjacency_matrix')

# GCN Layer 1
x = layers.Lambda(gcn_op, name='gcn1_aggregate')([inputs, adj_input])
x = layers.Dense(64, activation='relu', name='gcn1_transform')(x)
x = layers.Dropout(0.5, name='gcn1_dropout')(x)

# GCN Layer 2
x = layers.Lambda(gcn_op, name='gcn2_aggregate')([x, adj_input])
x = layers.Dense(32, activation='relu', name='gcn2_transform')(x)
x = layers.Dropout(0.5, name='gcn2_dropout')(x)

# Output Layer
x = layers.Lambda(gcn_op, name='gcn_out_aggregate')([x, adj_input])
outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

model = Model(inputs=[inputs, adj_input], outputs=outputs)
model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(f"Model built successfully! Parameters: {model.count_params():,}")

# --- Step 6: Prepare Data for Keras fit() ---

# Reshape data to have a batch dimension: (1, Nodes, Features)
feat_input = features[np.newaxis, :, :]
adj_input = adj_norm[np.newaxis, :, :]
labels_input = labels[np.newaxis, :]

# Create Masks (Sample Weights)
num_samples = labels.shape[0]
train_mask = np.zeros(num_samples, dtype=np.float32)
train_mask[train_idx] = 1.0
train_mask_input = train_mask[np.newaxis, :]

val_mask = np.zeros(num_samples, dtype=np.float32)
val_mask[val_idx] = 1.0
val_mask_input = val_mask[np.newaxis, :]

# --- Step 7: Training ---

print("Starting training with model.fit()...")
history = model.fit(
    x=[feat_input, adj_input],
    y=labels_input,
    sample_weight=train_mask_input,
    validation_data=([feat_input, adj_input], labels_input, val_mask_input),
    epochs=200,
    batch_size=1,
    verbose=1,
    callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

print("Training completed!")


# 8. Load Test Features
df_test_feat = pd.read_csv('.../test_features_only.csv')
test_ids = df_test_feat['id'].values

# 9. Update the Full Feature Matrix
# In Transductive GCNs, we need to include the test node features in the graph
# to perform aggregation before the final prediction step.
features[test_ids] = df_test_feat[feature_columns].values

print(f"Added test features to graph. Total features shape: {features.shape}")

# 10. Predict
# We predict on the full graph (train + val + test) because GCN layers aggregate info
pred = model.predict([features[np.newaxis, :, :], adj_norm[np.newaxis, :, :]])[0]

# Extract predictions for the test nodes
test_pred_indices = np.argmax(pred[test_ids], axis=1)

# 11. Create Submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'target': test_pred_indices
})

# 12. Save to CSV
output_filename = 'submission.csv'
submission.to_csv(output_filename, index=False)

print(f"✅ Successfully saved predictions to '{output_filename}'")
