#!/bin/bash

set -e

################################################################################
# Environment
################################################################################

# Constants
SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"
LOCAL_DATA_DIR="${PROJECT_DIR}/data/"

if [ "$1" == "" ]; then
  echo "Usage: $0 <DATA-DIR>"
  exit
fi

DATA_DIR=$(echo $1 | sed 's:/*$::')

echo Creating symlink from "$LOCAL_DATA_DIR" to "$DATA_DIR"
ln -s "$DATA_DIR" "$PROJECT_DIR"

################################################################################
# Download data (~14G)
################################################################################

# RIR / Noise Database (3G)
echo "downloading RIR/noise dataset..."
mkdir -p "$DATA_DIR"/rir-database/real
mkdir -p "$DATA_DIR"/rir-database/synthetic
mkdir -p "$DATA_DIR"/noise-database/room
mkdir -p "$DATA_DIR"/noise-database/pointsource

wget -O "$DATA_DIR"/rirs-noises.zip \
  https://www.openslr.org/resources/28/rirs_noises.zip
unzip "$DATA_DIR"/rirs-noises.zip -d "$DATA_DIR"/
rm -f "$DATA_DIR"/rirs-noises.zip

cp -a "$DATA_DIR"/RIRS_NOISES/pointsource_noises/. \
  "$DATA_DIR"/noise-database/pointsource
cp -a "$DATA_DIR"/RIRS_NOISES/simulated_rirs/. \
  "$DATA_DIR"/rir-database/synthetic

room_noises=($(find "$DATA_DIR"/RIRS_NOISES/real_rirs_isotropic_noises/ -maxdepth 1 -name '*noise*' -type f))
cp -- "${room_noises[@]}" "$DATA_DIR"/noise-database/room

rirs=($(find "$DATA_DIR"/RIRS_NOISES/real_rirs_isotropic_noises/ ! -name '*noise*' ))
cp -- "${rirs[@]}" "$DATA_DIR"/rir-database/real

rm -rf "$DATA_DIR"/RIRS_NOISES/

# LibriTTS-R (11G)
echo "downloading LibriTTS-R test_clean..."
wget -O "$DATA_DIR"/test_clean.tar.gz \
 https://www.openslr.org/resources/141/test_clean.tar.gz
tar -xzvf "$DATA_DIR"/test_clean.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/test_clean.tar.gz

echo "downloading LibriTTS-R train_clean_100..."
wget -O "$DATA_DIR"/train_clean_100.tar.gz \
 https://www.openslr.org/resources/141/train_clean_100.tar.gz
tar -xzvf "$DATA_DIR"/train_clean_100.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/train_clean_100.tar.gz
