#!/bin/bash
#find /mmu-audio-ssd/frontend/audioSep/wanghualei/code/mtg-jamendo-dataset/Hollywood10 -type f -name "*.wav" > Holl.txt
#find /mmu-audio-ssd/frontend/audioSep/wanghualei/code/mtg-jamendo-dataset/Hollywood10 -type f -name "*.wav" > Holl.txt
# 创建目标目录
mkdir -p Hollywood10

# 读取txt文件中的每一行
while IFS= read -r file
do
  # 获取文件的目录、基本名称和扩展名
  dirpath=$(dirname "$file")
  filename=$(basename -- "$file")
  extension="${filename##*.}"
  filename="${filename%.*}"

  # 提取cd路径部分
  cdpath=$(echo "$dirpath" | grep -o 'cd[0-9]\+')

  # 获取音频的持续时间
  duration=$(ffmpeg -i "$file" 2>&1 | grep "Duration" | cut -d ' ' -f 4 | sed s/,//)
  duration_in_sec=$(echo "$duration" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')

  # 计算需要切割的片段数量
  num_segments=$(echo "$duration_in_sec / 10" | bc)

  # 循环切割音频
  for i in $(seq 0 $((num_segments - 1)))
  do
    start_time=$(echo "$i * 10" | bc)
    output_file="Hollywood10/${cdpath}_${filename}_part${i}.${extension}"

    # 检查文件是否已经存在，如果存在则跳过
    if [ -f "$output_file" ]; then
      echo "File $output_file already exists, skipping..."
      continue
    fi

    ffmpeg -i "$file" -ss "$start_time" -t 10 -c copy "$output_file"
  done
done < output.txt
