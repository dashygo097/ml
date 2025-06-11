rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  ./core ./benchmark ./README.md ./requirements.txt lcxie@222.20.98.159:~/dev/ml
rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' --exclude='attempts' --exclude="valid_*"\
  ./demo/Hust-IC-LipSync lcxie@222.20.98.159:~/dev/ml/demo
