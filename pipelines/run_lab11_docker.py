"""
Lab 11: Containerization Pipeline Runner
Automates verifying and building the Docker container for the Ames House Price Prediction service.
"""

import os
import shutil
import subprocess
import sys

def check_docker_available():
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    try:
        res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False

def run_lab11():
    print("[INFO] ========================================================")
    print("[INFO] Starting Lab 11: Docker Containerization Pipeline (House Price)")
    print("[INFO] ========================================================")

    # 1. Verify necessary files
    required_files = [
        "Dockerfile",
        ".dockerignore",
        "requirements-api.txt",
        "docker-compose.yml",
        "deployment/model.pkl",
        "deployment/preprocessor.pkl",
        "config.yaml",
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            print(f"[ERROR] Missing required deployment file: {rf}")
            sys.exit(1)
        print(f"[OK] Found required file: {rf}")

    print("\n[INFO] Checking Docker daemon status...")
    if not check_docker_available():
        print("[WARNING] Docker daemon is not active or accessible.")
        print("[INFO] Skipping runtime container build test. Docker deployment files are intact and verified.")
        return

    # 2. Build Docker image
    image_tag = "house-price-prediction-api:latest"
    print(f"\n[INFO] Building Docker image '{image_tag}'...")
    build_cmd = ["docker", "build", "-t", image_tag, "."]
    build_res = subprocess.run(build_cmd)
    if build_res.returncode != 0:
        print("[ERROR] Docker build failed.")
        sys.exit(build_res.returncode)

    print(f"[SUCCESS] Docker image '{image_tag}' built successfully.")

if __name__ == "__main__":
    run_lab11()
