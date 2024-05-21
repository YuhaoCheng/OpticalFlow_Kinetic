#!/bin/bash
if [ $# -eq 1 ]; then
    LOCAL_ROOT_PATH=$1
    echo "Local dataset root path: ${LOCAL_ROOT_PATH}"
else
    LOCAL_ROOT_PATH='/dfs/data/Optical_Flow_Dataset'
fi


DIRECTORY='datasets'
if [ ! -d "${DIRECTORY}" ]
then
    mkdir "$DIRECTORY"
else
    echo "Directory $DIRECTORY already exists"
fi
cd "$DIRECTORY"
 
CHAIRS_DIR='FlyingChairs_release'
if [[ -L "${CHAIRS_DIR}" && -d "${CHAIRS_DIR}" ]]; then  # if ${CHAIRS_DIR} exists, and is a symbolic link(-L) and a directory(-d) as created by `ln -s`
    echo "${CHAIRS_DIR} exists"
else
    ln -s ${LOCAL_ROOT_PATH}/${CHAIRS_DIR} "${CHAIRS_DIR}"
    echo "Link ${CHAIRS_DIR}"
fi


THINGS_DIR='FlyingThings3D'
if [[ -L "${THINGS_DIR}" && -d "${THINGS_DIR}" ]]; then
    echo "${THINGS_DIR} exists"
else
    ln -s ${LOCAL_ROOT_PATH}/${THINGS_DIR} "${THINGS_DIR}"
    echo "Link ${THINGS_DIR}"
fi


SINTEL_DIR='Sintel'
if [[ -L "${SINTEL_DIR}" && -d "${SINTEL_DIR}" ]]; then
    echo "${SINTEL_DIR} exists"
else
    ln -s ${LOCAL_ROOT_PATH}/${SINTEL_DIR} "${SINTEL_DIR}"
    echo "Link MPI Sintel"
fi


KITTI_DIR='KITTI'
if [[ -L "${KITTI_DIR}" && -d "${KITTI_DIR}" ]]; then
    echo "${KITTI_DIR} exists"
else
    ln -s ${LOCAL_ROOT_PATH}/${KITTI_DIR} "${KITTI_DIR}"
    echo "Link ${KITTI_DIR}"
fi


SFSINTEL_DIR='high_speed_sintel'
if [[ -L "${SFSINTEL_DIR}" && -d "${SFSINTEL_DIR}" ]]; then
    echo "${SFSINTEL_DIR} exists"
else
    ln -s ${LOCAL_ROOT_PATH}/${SFSINTEL_DIR} "${SFSINTEL_DIR}"
    echo "Link High Speed Sintel"
fi


DAVIS='DAVIS'
if [[ -L "${DAVIS}" && -d "${DAVIS}" ]]; then
    echo "${DAVIS} exists"
else
    ln -s $LOCAL_ROOT_PATH/DAVIS "${DAVIS}"
    echo "Link DAVIS"
fi


HD1K_DIR='HD1K'
if [[ -L "${HD1K_DIR}" && -d "${HD1K_DIR}" ]]; then
    echo "${HD1K_DIR} exists"
else
    ln -s ${LOCAL_ROOT_PATH}/${HD1K_DIR} "${HD1K_DIR}"
    echo "Link HD1K"
fi