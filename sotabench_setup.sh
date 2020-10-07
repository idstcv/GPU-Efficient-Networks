#!/usr/bin/env bash -x
#
# Copyright 2020 Tal Ridnik, Hussam Lawen, Asaf Noy, Itamar Friedman  (https://github.com/hussam789/TResNet)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
export DEBIAN_FRONTEND=noninteractive
# REPO="$( cd "$(dirname "$0")" ; cd .. ; pwd -P )"
# cd $REPO

$PYTHON -m pip install -e .
$PYTHON -m pip install torch==1.4.0
$PYTHON -m pip install torchvision==0.5.0
$PYTHON -m pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12
$PYTHON -m pip install gdown

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --upgrade Pillow

### Pillow-simd with libjpeg turbo
$PYTHON -m pip uninstall -y pillow
apt-get update
apt-get install -y libjpeg-dev zlib1g-dev libpng-dev libwebp-dev
CFLAGS="${CFLAGS} -mavx2" $PYTHON -m pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all:--compile https://github.com/mrT23/pillow-simd/zipball/simd/7.0.x
$PYTHON -c "from PIL import Image; print(Image.PILLOW_VERSION)"

apt-get install wget
wget https://onedrive.hyper.ai/down/ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12.tar.gz -P ./.data/vision/imagenet
wget https://onedrive.hyper.ai/down/ImageNet/data/ImageNet2012/ILSVRC2012_img_val.tar -P ./.data/vision/imagenet

# gdown https://drive.google.com/file/d/1pySDEdfvLRROaNXi-fWrd_aV9N6W9Yjo/view?usp=sharing
#gdown https://drive.google.com/uc?id=1cTOwVmxLqWhNl8zg2ZZJfiyejEcOq1RR
#gdown https://drive.google.com/uc?id=1jgKXMoJpZ6Sow_sJhgi5tec7AyeQCLbH
#gdown https://drive.google.com/uc?id=1IIPU77tcS83cBRTAfKeGvAmcBL5D196r
