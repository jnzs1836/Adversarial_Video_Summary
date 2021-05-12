#!/bin/bash
for filepath in ./videos/*.mp4; do
	filename=${filepath:9:-4}
	echo $filename
	mkdir ./images/$filename
	ffmpeg -i $filepath -qscale:v 24 ./images/$filename/out%d.jpg
done
