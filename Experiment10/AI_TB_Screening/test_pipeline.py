"""
Comprehensive Integration & Verification Test Suite for AI TB Screening Flask App
"""

import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
import config

def run_tests():
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE FLASK INTEGRATION TESTS")
    print("=" * 60)

    app.app.config["TESTING"] = True
    client = app.app.test_client()

    routes = [
        ("/", "Dashboard"),
        ("/dataset", "Dataset Explorer"),
        ("/preprocessing", "Preprocessing Pipeline"),
        ("/features", "Feature Extraction"),
        ("/training", "Training Interface"),
        ("/evaluation", "Model Evaluation"),
        ("/comparison", "Model Comparison"),
        ("/prediction", "Prediction Interface"),
        ("/explainability", "Explainable AI (Grad-CAM)"),
        ("/history", "Screening History"),
        ("/about", "About & Documentation"),
        ("/api/training-status", "API: Training Status")
    ]

    all_passed = True
    for route, label in routes:
        response = client.get(route)
        if response.status_code in [200, 302]:
            print(f" [PASS] {route:<24} Status: {response.status_code} ({label})")
        else:
            print(f" [FAIL] {route:<24} Status: {response.status_code} ({label})")
            all_passed = False

    # Test Prediction Route with a sample image
    sample_img = next(config.TB_DIR.glob("*.jpg"), None)
    if sample_img:
        print(f"\nTesting Prediction POST with: {sample_img.name}...")
        with open(sample_img, "rb") as f:
            data = {
                "file": (io.BytesIO(f.read()), sample_img.name),
                "model_name": "EfficientNetB0"
            }
            pred_resp = client.post("/prediction", data=data, content_type="multipart/form-data")
            if pred_resp.status_code == 200:
                print(f" [PASS] Prediction POST succeeded! Status: {pred_resp.status_code}")
                # Check if result was rendered in HTML
                html = pred_resp.get_data(as_text=True)
                if "SCREENING CLASSIFICATION" in html and "Model Output Probabilities" in html:
                    print(" [PASS] Prediction results rendered correctly in HTML!")
                else:
                    print(" [FAIL] Prediction results missing in HTML output.")
                    all_passed = False
            else:
                print(f" [FAIL] Prediction POST failed with status: {pred_resp.status_code}")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED - PLEASE REVIEW LOGS")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_tests()
