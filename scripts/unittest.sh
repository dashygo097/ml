#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
UNITTEST_DIR="$BASE_DIR/tests"

cd "$UNITTEST_DIR" || exit
for file in $(find . -name "test_*.py"); do
    echo "[INFO] Running $file"
    python3 "$file"
done
