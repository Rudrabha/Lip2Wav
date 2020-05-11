#!/bin/bash
speaker=$(basename $1)
mkdir -p "$1/videos/"
echo "Downloading Train set of $speaker"
youtube-dl -f best -a $1/train.txt -o $1"/videos/%(id)s.%(ext)s"
echo "Downloading Val set of $speaker"
youtube-dl -f best -a $1/val.txt -o $1"/videos/%(id)s.%(ext)s"
echo "Downloading Test set of $speaker"
youtube-dl -f best -a $1/test.txt -o $1"/videos/%(id)s.%(ext)s"

FILES=$1/videos/*
DST_DIR=$1/intervals
mkdir -p $DST_DIR

for f in $FILES
do
	echo $f
	fname=$(basename $f)
	fname="${fname%.*}"
	mkdir "$DST_DIR/$fname"
	/usr/bin/ffmpeg -loglevel panic -i $f -acodec copy -f segment -vcodec copy \
			-reset_timestamps 1 -map 0 -segment_time 30 "$DST_DIR/$fname/cut-%d.mp4"
done