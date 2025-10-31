#!/bin/bash

echo "cloning mmseg config folder from mmsegmentation repo"
mkdir mmseg
cd mmseg
git init
git remote add -f origin https://github.com/open-mmlab/mmsegmentation.git
git config core.sparseCheckout true
echo "configs/" >> .git/info/sparse-checkout
git pull origin master
[ ! -d "configs" ] && echo "configs folder missing, clone failed" && exit 1
cd ..
export MM_CONFIG_FOLDER_SEG="mmseg/configs"
echo "clone complete"
