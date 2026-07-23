#!/bin/bash

# 下载 GQA 问题文件
GQA_DIR="./playground/data/eval/gqa/data/questions"
mkdir -p ${GQA_DIR}

cd ${GQA_DIR}

echo "Downloading GQA questions..."
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip

echo "Extracting..."
unzip -q questions1.2.zip

echo "Cleaning up..."
rm questions1.2.zip

echo "Done! Files:"
ls -lh *.json
