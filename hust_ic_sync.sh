rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  ./core ./benchmark ./README.md ./requirements.txt ./hust_ic_init.sh lcxie@222.20.98.159:~/dev/ml
rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  ./demo/ResNet-Pretrain-Faces lcxie@222.20.98.159:~/dev/ml/demo
rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  ./demo/Wav2Lip lcxie@222.20.98.159:~/dev/ml/demo

