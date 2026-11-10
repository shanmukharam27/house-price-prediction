import os
import numpy as np
import joblib

def test_data_shapes():
    assert os.path.exists('data/processed/X_train_final.npy'), "X_train_final.npy missing"
    assert os.path.exists('data/processed/X_test_final.npy'), "X_test_final.npy missing"
    assert os.path.exists('data/processed/y_train.npy'), "y_train.npy missing"
    assert os.path.exists('data/processed/y_test.npy'), "y_test.npy missing"
    
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"
    assert X_train.shape[0] == y_train.shape[0], "Train sample mismatch"
    assert X_test.shape[0] == y_test.shape[0], "Test sample mismatch"
    assert not np.isnan(X_train).any(), "NaNs detected in X_train"
    assert not np.isnan(X_test).any(), "NaNs detected in X_test"

def test_model_inference():
    assert os.path.exists('models/random_forest_baseline.pkl'), "random_forest_baseline.pkl missing"
    model = joblib.load('models/random_forest_baseline.pkl')
    X_test = np.load('data/processed/X_test_final.npy')
    preds = model.predict(X_test)
    assert len(preds) == len(X_test), "Prediction shape mismatch"
    assert not np.isnan(preds).any(), "NaNs in predictions"

if __name__ == "__main__":
    test_data_shapes()
    test_model_inference()
    print("[SUCCESS] All pipeline tests passed successfully!")
