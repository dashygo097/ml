#!/bin/bash

TEST_DIR=$(pwd)
BASE_DIR=$(dirname "$TEST_DIR")
APP_DIR="$BASE_DIR/3rdparty/trtrt/examples/cornellbox"

cd $APP_DIR || exit
python3 main.py

cd $TEST_DIR || exit
