#!/bin/bash

## f4
URL_NYX=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
URL_QMC=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/QMCPack/SDRBENCH-QMCPack.tar.gz
## f8
URL_MIRANDA=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz
URL_S3D=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/S3D/SDRBENCH-S3D.tar.gz

if [ $# -eq 1 ]; then
    echo "setting data dir root to $1"
else
    echo "bash setup-data.sh [root dir to put data]"
    exit 1
fi

RED='\033[0;31m'
GRAY='\033[0;37m'
NOCOLOR='\033[0m'
BOLDRED='\033[1;31m'

export DATAPATH=$1
mkdir -p $DATAPATH
pushd $DATAPATH

for URL in $URL_NYX $URL_QMC $URL_MIRANDA $URL_S3D; do
    FILE=$(basename $URL)
    TAR_XDIR=$(basename $FILE .tar.gz)
    
    echo -e "\n${BOLDRED}${FILE}${NOCOLOR}"
    if [ ! -f $FILE ]; then
        echo "    downloading $FILE"
        wget $URL
    else
        echo "    $FILE exists...skip downloading"
    fi
    
    if [ -d dataset ] || [ -d ${TAR_XDIR} ]; then
        echo "    ${FILE} has been untar'ed...skip"
    else
        echo "    untaring $FILE"
        echo -e "${GRAY}"
        tar zxvf $FILE
        echo -e "${NOCOLOR}"
    fi
done

## special fix to QMC
if [ -d dataset ]; then
    EXISTING=dataset
    FILE=$(basename $URL_QMC)
    SUPPOSED=$(basename $FILE .tar.gz)
    echo -e "${BOLDRED}linking QMCPack dir...${NOCOLOR}"
    if [ ! -f ${SUPPOSED} ]; then
        ln -s ${EXISTING} ${SUPPOSED}
    fi
fi

## covert f8 to f4, Miranda
MIRANDA_TAR_FILE=$(basename $URL_MIRANDA)
MIRANDA_DIR=$(basename $MIRANDA_TAR_FILE .tar.gz)

pushd ${MIRANDA_DIR}
for F8_DATA in *.d64; do
    F4_DATA=$(basename ${F8_DATA} .d64).f32
    if [ ! -f ${F4_DTA} ]; then 
        echo -e "    ${RED}coverting ${F8_DATA} to ${F4_DATA}${NOCOLOR}"
        convertDoubleToFloat ${F8_DATA} ${F4_DATA} >/dev/null
    else
        echo -e "    ${RED}${F4_DATA} exists...skip converting${NOCOLOR}"
    fi
done

popd

## TODO JHTDB

popd
