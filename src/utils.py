import os
import joblib


def save_object(file_path, obj):
    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(obj, file_path)
    print("Object saved successfully")


def load_object(file_path):
    obj = joblib.load(file_path)
    print("Object loaded successfully")
    return obj


def test_utils():
    data = {
        "name": "Kavya",
        "role": "Data Scientist"
    }

    save_object("artifacts/sample.pkl", data)

    loaded = load_object("artifacts/sample.pkl")

    print(loaded)

    if loaded == data:
        print("✅ Utils Test Passed")
    else:
        print("❌ Utils Test Failed")


if __name__ == "__main__":
    test_utils()