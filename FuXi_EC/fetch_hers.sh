#!/bin/bash
# bash fetch_hers.sh 2024013000
# fetch data from  server
rsync -avzhe ssh ***@*.*.*.*:/* /home/*
#DateStr_start=`date  +%Y%m%d%H` # init date
#DateStr_start='2023012912'
DateStr_start=$1
#prefix_files=('A1S' 'A1D' 'A2S' 'A2D')
# A1S 18点气压层
# A2S 18点地面
# A1D 12点气压层
# A2D 12点地面
if [ ${DateStr_start:8:10} == "12" -o ${DateStr_start:8:10} == "00" ];then
  prefix_files=('A1D' 'A2D')
elif [ ${DateStr_start:8:10} == "06" -o  ${DateStr_start:8:10} == "18" ];then
  prefix_files=('A1S' 'A2S')
else
  echo "Invalid date"
fi

hour=0 # forcast time

for((i=0;i<${#prefix_files[@]};i++))
do
  prefix_file=${prefix_files[i]}
  FileTime=`date -d "${DateStr_start:0:8} ${DateStr_start:8:10} ${hour}hour" +%m%d%H` # pred time
  if [ ${hour} == 0 ];then
    FileName="${prefix_file}${DateStr_start:4:10}00${FileTime}011"
  else
    FileName="${prefix_file}${DateStr_start:4:10}00${FileTime}001"
  fi
#  echo "${FileName}"
#  echo "${prefix_file:1:1}"
#  echo "${DateStr_start:8:10}"
  origin_path="/home/*/DataStore/data/EC_UK/${FileName}"

  if [ $((${DateStr_start:8:10}%6)) == 0 -a ${prefix_file:1:1} == '1' \
  -a -f ${origin_path} ];then
    target_path="/home/*/workspace/FuXi-infer-online/FuXi_EC/input_meta/pl/${DateStr_start:8:10}/${DateStr_start:0:8}.grib"
    echo ${origin_path}
    echo ${target_path}
    ln -sfn ${origin_path}  ${target_path}
  elif [ $((${DateStr_start:8:10}%6)) == 0 -a ${prefix_file:1:1} == '2' \
  -a -f ${origin_path} ];then
    target_path="/home/*/workspace/FuXi-infer-online/FuXi_EC/input_meta/sfc/${DateStr_start:8:10}/${DateStr_start:0:8}.grib"
    echo ${origin_path}
    echo ${target_path}
    ln -sfn ${origin_path}  ${target_path}
  else
    echo "input_meta fetch failed :${origin_path}"
  fi
done
