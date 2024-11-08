#!/bin/bash

complexity_height=700
final_width=1000

for dataset in kodak clic20-pro-valid
do
    # Concatenate the images horizontally and resize them if they don't have the same height
    convert +append $dataset/perf_complexity.png $dataset/perf_decoding_time.png -resize x$complexity_height $dataset/all_complexity.png
    convert -append $dataset/rd.png $dataset/all_complexity.png -resize $final_width $dataset/concat_img.png
done

dataset=jvet
for class in B C D E F BCDEF
do
    # Concatenate the images horizontally and resize them if they don't have the same height
    convert +append $dataset/perf_complexity_class$class.png $dataset/perf_decoding_time_class$class.png -resize x$complexity_height $dataset/all_complexity_class$class.png
    convert -append $dataset/rd_class$class.png $dataset/all_complexity_class$class.png -resize $final_width $dataset/concat_img_class$class.png
done
