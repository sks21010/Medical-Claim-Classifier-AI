import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, num_samples=None, p_1=None, p_0=None, entropy=None):
        self.feature_index = feature_index 
        self.threshold = threshold 
        self.left = left
        self.right = right
        self.value = value
        self.num_samples = num_samples 
        self.p_1 = p_1
        self.p_0 = p_0 
        self.entropy = entropy

print("Node class loaded")

class DecisionTreeClassifier:
    def __init__(self, max_depth=7, max_features=None):
        self.max_depth = max_depth 
        self.max_features = max_features
        self.root = None 

    def calculate_entropy(self, y, w):
        if len(y) == 0:
            return 0.0 
        weighted_counts = np.bincount(y, weights=w)
        total_weight = np.sum(w)
        probs = weighted_counts[weighted_counts > 0] / total_weight
        return -np.sum(probs * np.log2(probs))

    def calculate_information_gain(self, y_parent, w_parent, y_left, w_left, y_right, w_right):
        E_parent = self.calculate_entropy(y_parent, w_parent)
        prop_left = np.sum(w_left) / np.sum(w_parent)
        prop_right = np.sum(w_right) / np.sum(w_parent)
        E_children = (prop_left * self.calculate_entropy(y_left, w_left) + 
                    prop_right * self.calculate_entropy(y_right, w_right))
        return E_parent - E_children

    def find_best_split(self, X, y, w, max_features=None):
        best_gain = -1
        best_split = {} 
        n_features = X.shape[1]

        if max_features is None:
            max_features = int(np.sqrt(n_features))
        elif max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(max_features, float):
            max_features = int(max_features * n_features)
        
        feature_indices = np.random.choice(n_features, size=min(max_features, n_features), replace=False)
        
        for feature_index in feature_indices:
            col = X[:, feature_index]
            mask = col > 0.5
            
            y_left, w_left = y[mask], w[mask]
            y_right, w_right = y[~mask], w[~mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            gain = self.calculate_information_gain(y, w, y_left, w_left, y_right, w_right)

            if gain > best_gain:
                best_gain = gain 
                best_split = {
                    "feature_index": feature_index,
                    "threshold": 0.5,
                    "gain": gain
                }
        return best_split
    
    def build_tree(self, X, y, w, depth=0):
        num_samples = len(y)
        total_weight = np.sum(w)
        
        weighted_counts = np.bincount(y.astype(int), weights=w)
        p_1 = weighted_counts[1] / total_weight if len(weighted_counts) > 1 else 0.0
        p_0 = weighted_counts[0] / total_weight if len(weighted_counts) > 0 else 0.0
        entropy = self.calculate_entropy(y, w)
        
        if depth >= self.max_depth or len(y) == 0 or len(np.unique(y)) == 1:
            if len(y) > 0:
                leaf_value = np.argmax(weighted_counts)
            else:
                leaf_value = None
            return Node(value=leaf_value, num_samples=num_samples, p_1=p_1, p_0=p_0, entropy=entropy)
            
        split = self.find_best_split(X, y, w, self.max_features) 
        
        if split.get("gain", 0) <= 0:
            leaf_value = np.argmax(weighted_counts)
            return Node(value=leaf_value, num_samples=num_samples, p_1=p_1, p_0=p_0, entropy=entropy)
            
        best_f_idx = split["feature_index"]
        best_thresh = split["threshold"]
        
        left_mask = X[:, best_f_idx] <= best_thresh
        
        X_left, y_left, w_left = X[left_mask], y[left_mask], w[left_mask]
        X_right, y_right, w_right = X[~left_mask], y[~left_mask], w[~left_mask]
        
        left_node = self.build_tree(X_left, y_left, w_left, depth + 1)
        right_node = self.build_tree(X_right, y_right, w_right, depth + 1)
        
        return Node(feature_index=best_f_idx, threshold=best_thresh, left=left_node, right=right_node,
                    num_samples=num_samples, p_1=p_1, p_0=p_0, entropy=entropy)

    def fit(self, X, y, sample_weights=None):
        self.X_cols = X.columns
        
        if hasattr(X, 'to_numpy'):
            X_np = X.to_numpy()
            y_np = y.to_numpy()
        else:
            X_np = X
            y_np = y
            
        if sample_weights is None:
            w_np = np.ones(len(y_np)) 
        else:
            w_np = sample_weights.to_numpy() if hasattr(sample_weights, 'to_numpy') else sample_weights
            
        self.root = self.build_tree(X_np, y_np, w_np, depth=0) 
        return self
    
    def predict_sample(self, x, tree):
        if tree.value is not None:
            return tree.value 
        
        feature_value = x[tree.feature_index]

        if feature_value <= tree.threshold:
            return self.predict_sample(x, tree.left)
        else:
            return self.predict_sample(x, tree.right)
        
    def predict(self, X):
        if not self.root:
            raise Exception("Fit model first before calling predict")
        
        X_numpy = X.to_numpy()
        predictions = []

        for row in X_numpy:
            prediction = self.predict_sample(row, self.root)
            predictions.append(prediction)
        
        return pd.Series(predictions, index=X.index)

print("DecisionTreeClassifier class loaded")