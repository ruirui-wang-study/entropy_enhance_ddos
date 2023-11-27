#!/bin/bash
#################################################################################
#
# Description: this file takes as input a folder with pcap files, and converts them
# to CSV files. It extracts just specific features form the file, e.g., frame len,
# Ether type, IP protocols etc.
#
# You will need python and tshark installed to run this code
#
# frame.len: 数据包的长度。
# eth.type: 以太网帧的类型。
# ip.proto: IP 数据包的协议类型。
# ip.flags: IP 数据包的标志。
# ipv6.nxt: IPv6 数据包的下一个头部。
# ipv6.opt: IPv6 数据包的选项。
# tcp.srcport: TCP 数据包的源端口。
# tcp.dstport: TCP 数据包的目标端口。
# tcp.flags: TCP 数据包的标志。
# udp.srcport: UDP 数据包的源端口。
# udp.dstport: UDP 数据包的目标端口。
# eth.src: 以太网帧的源 MAC 地址。


FILES=$(find original_files/ -type f)
TARGET='csv_files'
echo "Usage: ./set_features.py <substitution file>"

for file in $FILES
do
    echo "${file}"
    rm tmp
    ofile="$(cut -d'/' -f2 <<<"$file" |cut -d'.' -f1)"
    # tshark -r ${file} -Tfields -E occurrence=f -E separator=, -e frame.len -e eth.type -e ip.proto -e ip.flags -e ipv6.nxt -e ipv6.opt -e tcp.srcport -e tcp.dstport -e tcp.flags -e udp.srcport -e udp.dstport  -e eth.src > ${ofile}.csv
    tshark -r ${file} -T fields -E occurrence=f -E separator=, -e frame.len -e eth.type -e ip.proto -e ip.flags -e ipv6.nxt -e ipv6.opt -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e tcp.flags -e udp.srcport -e udp.dstport -e eth.src > ${ofile}.csv

    cp ${ofile}.csv  else.csv
    while IFS= read -r line
    do
      echo "处理行: $line"
      MAC="$(echo $line |cut -d/ -f1)"
      echo "MAC值: $MAC"
      grep -i $MAC ${ofile}.csv |sed  "s/$line/gI" >>tmp
      grep -v -i $MAC else.csv >else1.csv
      mv else1.csv else.csv
    done < "$1"
    mv tmp ${TARGET}/${ofile}-labeled.csv
    rm ${ofile}.csv
    cat else.csv | sed 's/.\{17\}$/4/' >> ${TARGET}/${ofile}-labeled.csv
    cat ${TARGET}/${ofile}-labeled.csv |sed 's/,,/,-1,/gI' | sed 's/,,/,-1,/gI' > ${TARGET}/${ofile}-labeled.csv1
    mv ${TARGET}/${ofile}-labeled.csv1 ${TARGET}/${ofile}-labeled.csv 
done
