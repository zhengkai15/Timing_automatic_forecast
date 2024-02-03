#!/bin/bash
# https://mp.weixin.qq.com/s/cuzwph-k9biHaPEE5QK5yQ
# https://blog.csdn.net/weixin_37696997/article/details/78578128

# nohup bash ./FuXi_EC/autorun.sh > inference.log 2>&1 &
# export PATH="$PATH:/home/*/ossutil"
# ossutil ls oss://application

# 配置环境
source ~/.bashrc
conda activate FuXi-Sparse
workdir=/home/*/workspace/FuXi-infer-online
cd $workdir
export PYTHONPATH='/home/*/workspace/FuXi-infer-online'
# 脚本路径
shell_path=/home/*/workspace/FuXi-infer-online/FuXi_EC/
# 自动运行和推理开始时间和结束时间
start_date=`date +%Y%m%d`
end_date="20241229"
flag=1
get_gpu_memery(){
	echo `nvidia-smi | grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | sed "s/^|//" | awk '{print ($8" "$10)}' | sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | awk '{print $2 - $1}'`
}
# 开始下载数据
while [ "$start_date" -le "$end_date" ];
do
# 设置23点30到9点30这个时间段不运行脚本
    while [ 1 == 1 ]
    do  # can't run on mac
        # 设定脚本需要传入的时间格式 D日只能获取D-1的输入，不可能获取到D的输入
        V_DAY=`date -d '1 days ago' +%Y%m%d`
#       V_DAY=`date -d '2 days ago' +%Y%m%d` # debug
        time_now=`date '+%Y-%m-%d %H:%M:%S'`
        date_now=`date '+%Y%m%d'`
#       date_now=`date -d '1 days ago' +%Y%m%d` # debug

        time_now_s=$(date -d "$time_now" +%s)
        start_date_s=$(date -d "$start_date" +%s)
        # check if file exists
        if [ $flag -eq 1 -a $time_now_s -gt $start_date \
        -a ! -f ${shell_path}input_meta/pl/18/${V_DAY}.grib \
        -o ! -f ${shell_path}input_meta/pl/00/${date_now}.grib ]; then
            # download input meta data
            echo ${shell_path}fetch_hers.sh
            bash ${shell_path}fetch_hers.sh ${V_DAY}18
            bash ${shell_path}fetch_hers.sh ${date_now}00
#            break
            # if file exists, break and inference
            if [ $flag -eq 1 -a $time_now_s -gt $start_date \
            -a -f ${shell_path}input_meta/pl/18/${V_DAY}.grib \
            -a -f ${shell_path}input_meta/pl/00/${date_now}.grib \
            -a ! -d ${shell_path}output_onnx_c92/${date_now}-00_input_grib-output ]; then
              echo 'input meta data is available 00'
              echo 'break inner loop 00'
              break
            else
              echo waiting for fetching 00 input...
              sleep 300
            fi
        # check if file exists
        elif [ $flag -eq 2 -a $time_now_s -gt $start_date \
        -a ! -f ${shell_path}input_meta/sfc/06/${date_now}.grib \
        -o ! -f ${shell_path}input_meta/sfc/12/${date_now}.grib ]; then
            # download input meta data
            echo ${shell_path}fetch_hers.sh
            bash ${shell_path}fetch_hers.sh ${date_now}06
            bash ${shell_path}fetch_hers.sh ${date_now}12
          # if file exists, break and inference
          if [ $flag -eq 2 -a $time_now_s -gt $start_date \
          -a -f ${shell_path}input_meta/sfc/06/${date_now}.grib \
          -a -f ${shell_path}input_meta/sfc/12/${date_now}.grib \
          -a ! -d ${shell_path}output_onnx_c92/${date_now}-12_input_grib-output ]; then
              echo 'input meta data is available 12'
              echo 'break inner loop 12'
              break
          else
              echo waiting for fetching 12 input...
              sleep 300
          fi
        else
            echo waiting ...
            sleep 300
        fi
    done
    if [ $flag -eq 1 ]; then
        while [ $(get_gpu_memery) -lt 20000 ]; do
          echo "wait for 20000MB gpu memory, sleep 60s"
          sleep 60
        done
        echo  "$date_now 00 inference started"
        python ./FuXi_EC/fuxi_EC.py --init_time ${V_DAY}  --model  ./FuXi_EC/onnx_c92/ --input_save_dir ./FuXi_EC/input_onnx_c92 --save_dir ./FuXi_EC/output_onnx_c92 --init_time_type case4  --raw_data_format_type grib &&
        echo "${date_now} 00 inference done"
        # submit data to oss
        echo "submit ${date_now} 00 data to oss"
        python ./FuXi_EC/submit2oss.py  --prefix inference_output_onnx_c92 --filename ./FuXi_EC/output_onnx_c92/$date_now-00_input_grib-output/ &&
        echo "submit ${date_now} 00 data done"
        flag=2
    elif [ $flag -eq 2 ]; then
        while [ $(get_gpu_memery) -lt 20000 ]; do
          echo "wait for 20000MB gpu memory, sleep 60s"
          sleep 60
        done
        echo  "$date_now 12 inference started"
        python ./FuXi_EC/fuxi_EC.py --init_time ${date_now} --model  ./FuXi_EC/onnx_c92/ --input_save_dir ./FuXi_EC/input_onnx_c92 --save_dir ./FuXi_EC/output_onnx_c92 --init_time_type case2 --raw_data_format_type grib &&
        echo "${date_now} 12 inference done"
        echo "submit ${date_now} 12 data to oss"
        # submit data to oss
        python ./FuXi_EC/submit2oss.py  --prefix inference_output_onnx_c92 --filename ./FuXi_EC/output_onnx_c92/$date_now-12_input_grib-output/ &&
        echo "submit ${date_now} 12 data done"
        echo "next date"
        # 下一天日期
        start_date=$(date -d next-day +%Y%m%d)
        flag=1
    else
        echo "error"
    fi
done
