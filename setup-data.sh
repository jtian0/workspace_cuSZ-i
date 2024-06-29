#!/bin/bash

## f4
URL_NYX=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
URL_QMC=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/QMCPack/SDRBENCH-QMCPack.tar.gz
## f8
URL_MIRANDA=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz
URL_S3D=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/S3D/SDRBENCH-S3D.tar.gz

if [ $# -eq 1 ]; then
    if [[ "$1" = "purgesoftlink" ]]; then
        echo "purging soft/symbolic links..."
        rm -f SDRBENCH-QMCPack
        exit 0
    fi
fi

RED='\033[0;31m'
GRAY='\033[0;37m'
NOCOLOR='\033[0m'
BOLDRED='\033[1;31m'

mkdir -p ${DATAPATH}
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
    mkdir -p $SUPPOSED
    mv $EXISTING/288x115x69x69/einspline_288_115_69_69.pre.f32 $SUPPOSED
fi

## covert f8 to f4, Miranda
MIRANDA_TAR_FILE=$(basename $URL_MIRANDA)
MIRANDA_DIR=$(basename $MIRANDA_TAR_FILE .tar.gz)
echo -e "${BOLDRED}converting Miranda data...${NOCOLOR}"
pushd ${MIRANDA_DIR}
for F8_DATA in *.d64; do
    F4_DATA=$(basename ${F8_DATA} .d64).f32
    if [ ! -f ${F4_DTA} ]; then
        echo -e "    ${RED}coverting ${F8_DATA} to ${F4_DATA} (overwrite)${NOCOLOR}"
        convertDoubleToFloat ${F8_DATA} ${F4_DATA} >/dev/null
    else
        echo -e "    ${RED}${F4_DATA} exists...skip converting${NOCOLOR}"
    fi
done
rm -f *.d64
popd

# split S3D
S3D_FIELDS=("stat_planar.1.1000E-03.CH4.f32" "stat_planar.1.7000E-03.CH4.f32" "stat_planar.2.3500E-03.CH4.f32" "stat_planar.2.9000E-03.CH4.f32" "stat_planar.2.9950E-03.CH4.f32" "stat_planar.1.1000E-03.CO.f32"  "stat_planar.1.7000E-03.CO.f32" "stat_planar.2.3500E-03.CO.f32" "stat_planar.2.9000E-03.CO.f32" "stat_planar.2.9950E-03.CO.f32" "stat_planar.1.1000E-03.CO2.f32" "stat_planar.1.7000E-03.CO2.f32" "stat_planar.2.3500E-03.CO2.f32" "stat_planar.2.9000E-03.CO2.f32" "stat_planar.2.9950E-03.CO2.f32" "stat_planar.1.1000E-03.H2O.f32" "stat_planar.1.7000E-03.H2O.f32" "stat_planar.2.3500E-03.H2O.f32" "stat_planar.2.9000E-03.H2O.f32" "stat_planar.2.9950E-03.H2O.f32" "stat_planar.1.1000E-03.N2.f32" "stat_planar.1.7000E-03.N2.f32" "stat_planar.2.3500E-03.N2.f32" "stat_planar.2.9000E-03.N2.f32" "stat_planar.2.9950E-03.N2.f32" "stat_planar.1.1000E-03.O2.f32" "stat_planar.1.7000E-03.O2.f32" "stat_planar.2.3500E-03.O2.f32" "stat_planar.2.9000E-03.O2.f32" "stat_planar.2.9950E-03.O2.f32" "stat_planar.1.1000E-03.PRES.f32" "stat_planar.1.7000E-03.PRES.f32" "stat_planar.2.3500E-03.PRES.f32" "stat_planar.2.9000E-03.PRES.f32" "stat_planar.2.9950E-03.PRES.f32" "stat_planar.1.1000E-03.T.f32" "stat_planar.1.7000E-03.T.f32" "stat_planar.2.3500E-03.T.f32" "stat_planar.2.9000E-03.T.f32" "stat_planar.2.9950E-03.T.f32" "stat_planar.1.1000E-03.U.f32" "stat_planar.1.7000E-03.U.f32" "stat_planar.2.3500E-03.U.f32" "stat_planar.2.9000E-03.U.f32" "stat_planar.2.9950E-03.U.f32" "stat_planar.1.1000E-03.V.f32" "stat_planar.1.7000E-03.V.f32" "stat_planar.2.3500E-03.V.f32" "stat_planar.2.9000E-03.V.f32" "stat_planar.2.9950E-03.V.f32" "stat_planar.1.1000E-03.W.f32" "stat_planar.1.7000E-03.W.f32" "stat_planar.2.3500E-03.W.f32" "stat_planar.2.9000E-03.W.f32" "stat_planar.2.9950E-03.W.f32")

S3D_DONE=0
S3D_TAR_FILE=$(basename $URL_S3D)
S3D_DIR=$(basename $S3D_TAR_FILE .tar.gz)
pushd ${S3D_DIR}
for file in "${S3D_FIELDS[@]}"; do
    if [ -e "$file" ]; then
        echo -e "    ${GRAY}${file} exists.${NOCOLOR}"
        S3D_DONE=1
    else
        echo -e "    ${RED}$file does not exist.${NOCOLOR}"
        S3D_DONE=0
    fi
done
if [ $S3D_DONE -eq 0 ]; then
    echo -e "${BOLDRED}spliting S3D data...${NOCOLOR}"
    python ${WORKSPACE}/split-S3D.py
fi

rm -f \
stat_planar.1.1000E-03.CH4.f32 \
stat_planar.1.1000E-03.CO2.f32 \
stat_planar.1.1000E-03.CO.f32 \
stat_planar.1.1000E-03.H2O.f32 \
stat_planar.1.1000E-03.N2.f32 \
stat_planar.1.1000E-03.O2.f32 \
stat_planar.1.1000E-03.PRES.f32 \
stat_planar.1.1000E-03.T.f32 \
stat_planar.1.1000E-03.U.f32 \
stat_planar.1.1000E-03.V.f32 \
stat_planar.1.1000E-03.W.f32 \
stat_planar.1.7000E-03.CH4.f32 \
stat_planar.1.7000E-03.CO2.f32 \
stat_planar.1.7000E-03.CO.f32 \
stat_planar.1.7000E-03.H2O.f32 \
stat_planar.1.7000E-03.N2.f32 \
stat_planar.1.7000E-03.O2.f32 \
stat_planar.1.7000E-03.PRES.f32 \
stat_planar.1.7000E-03.T.f32 \
stat_planar.1.7000E-03.U.f32 \
stat_planar.1.7000E-03.V.f32 \
stat_planar.1.7000E-03.W.f32 \
stat_planar.2.3500E-03.N2.f32 \
stat_planar.2.3500E-03.PRES.f32 \
stat_planar.2.9000E-03.CH4.f32 \
stat_planar.2.9000E-03.CO2.f32 \
stat_planar.2.9000E-03.CO.f32 \
stat_planar.2.9000E-03.H2O.f32 \
stat_planar.2.9000E-03.N2.f32 \
stat_planar.2.9000E-03.O2.f32 \
stat_planar.2.9000E-03.PRES.f32 \
stat_planar.2.9000E-03.T.f32 \
stat_planar.2.9000E-03.U.f32 \
stat_planar.2.9000E-03.V.f32 \
stat_planar.2.9000E-03.W.f32 \
stat_planar.2.9950E-03.CH4.f32 \
stat_planar.2.9950E-03.CO2.f32 \
stat_planar.2.9950E-03.CO.f32 \
stat_planar.2.9950E-03.H2O.f32 \
stat_planar.2.9950E-03.N2.f32 \
stat_planar.2.9950E-03.O2.f32 \
stat_planar.2.9950E-03.PRES.f32 \
stat_planar.2.9950E-03.T.f32 \
stat_planar.2.9950E-03.U.f32 \
stat_planar.2.9950E-03.V.f32 \
stat_planar.2.9950E-03.W.f32

popd

## TODO JHTDB


## end of file
popd
