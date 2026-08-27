#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets
wget -O datasets/mip_nerf_360.zip https://skhpersonal.s3.amazonaws.com/mip_nerf_360.zip
unzip datasets/mip_nerf_360.zip -d datasets
rm datasets/mip_nerf_360.zip
