#!/bin/bash
# Package the Lambda for arm64 (Graviton) without Docker: pinned manylinux aarch64 wheels + our code.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build && mkdir -p build
python3 -m pip install --quiet --target build --only-binary=:all: \
  --platform manylinux2014_aarch64 --python-version 3.12 --implementation cp \
  -r requirements-lambda.txt
cp -r riverwatch lambda_function.py cameras.json build/
mkdir -p build/profiles && cp -r profiles/* build/profiles/
find build -name "__pycache__" -prune -exec rm -rf {} +
# boto3 is provided by the Lambda runtime; drop test folders to keep the package small
rm -rf build/numpy/*/tests build/cv2/data 2>/dev/null || true
du -sh build
