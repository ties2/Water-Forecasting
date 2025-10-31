#!/bin/bash

echo "cloning mmdet config folder from mmdetection repo"
mkdir mmdet
cd mmdet
git init
git remote add -f origin https://github.com/open-mmlab/mmdetection.git
git config core.sparseCheckout true
echo "configs/" >> .git/info/sparse-checkout
git pull origin master
[ ! -d "configs" ] && echo "configs folder missing, clone failed" && exit 1
cd ..
export MM_CONFIG_FOLDER="mmdet/configs"
echo "clone complete"

