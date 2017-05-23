#!/bin/bash
set -e

HDD_DIR="/media/snhryt/Data/Research_Master"
SITUATION="100fonts_200class"
WORK_DIR="./MyWork/${SITUATION}"
NETWORK="caffenet"
ITER_NUM="30000"
TEST_DIRNAME="11S01-Black-Tuesday-Offset"

mkdir -pv $WORK_DIR

if [ ! -d "${WORK_DIR}/train_lmdb" ]; then
  ./build/tools/convert_imageset \
      / \
      "${WORK_DIR}/train.txt" \
      "${WORK_DIR}/train_lmdb" \
      -gray
  echo ""
fi
if [ ! -d "${WORK_DIR}/val_lmdb" ]; then
  ./build/tools/convert_imageset \
      / \
      "${WORK_DIR}/validation.txt" \
      "${WORK_DIR}/val_lmdb" \
      -gray
  echo ""
fi
if [ ! -e "${WORK_DIR}/mean.npy" ]; then
  if [ ! -e "${WORK_DIR}/mean.binaryproto" ]; then
    ./build/tools/compute_image_mean \
        "${WORK_DIR}/train_lmdb" \
        "${WORK_DIR}/mean.binaryproto"
  fi
  python ./python/ConvertBinaryprotoToNpy.py \
      "${WORK_DIR}/mean.binaryproto" \
      "${WORK_DIR}/mean.npy"
  echo ""
fi
if [ ! -e "${WORK_DIR}/${NETWORK}_iter_${ITER_NUM}.caffemodel" ]; then
  ./build/tools/caffe train \
      -solver \
      "${WORK_DIR}/${NETWORK}_solver.prototxt"
  echo ""
fi

python ./python/MakeRecogResultsTable.py \
    "${HDD_DIR}/Syn_AlphabetImages/font/${TEST_DIRNAME}" \
    "${WORK_DIR}/${NETWORK}_deploy.prototxt" \
    "${WORK_DIR}/${NETWORK}_iter_${ITER_NUM}.caffemodel" \
    "${HDD_DIR}/Syn_AlphabetImages/selected/100fonts_Clustering/SelectedFonts_200class.txt" \
    "${HDD_DIR}/RecogResults/Clustering/${SITUATION}/${TEST_DIRNAME}.csv" \
    --mean_img_filepath="${WORK_DIR}/mean.npy" 

echo "\n=== ALL DONE! =============================="