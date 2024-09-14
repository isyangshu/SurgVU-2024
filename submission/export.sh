#!/usr/bin/env bash

bash build.sh

docker save surgvu_cat2 | gzip -c > surgformer.tar.gz
