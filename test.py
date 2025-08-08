import joblib
import sklearn

print(f"scikit-learn version: {sklearn.__version__}")

try:
    model = joblib.load("model/heat_diease_model.pkl", mmap_mode=None)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")