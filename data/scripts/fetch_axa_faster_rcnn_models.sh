#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=axa_poc_faster_rcnn_final.caffemodel
URL=s3://axa.fr.ecm.exchange/OCR/faster-rcnn/$FILE

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  exit 0
fi

echo "Downloading AXA Faster R-CNN demo models (350M)..."

#wget $URL -O $FILE
aws s3 cp $URL .

echo "Done. Please run  ./tools/axademo.py to launch demo"
