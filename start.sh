#!/bin/bash

if [ ! -e  deepimage/core/style_transfer/models ]; then
    mkdir deepimage/core/style_transfer/models
    wget -O deepimage/core/style_transfer/models/21styles.model https://www.dropbox.com/s/2iz8orqqubrfrpo/21styles.model?dl=1
fi
app="deep-image"
docker build -t ${app} .
docker run -d -p 5000:5000 \
  --name=${app} \
  -v $PWD:/app ${app}
