#!/bin/bash

# remap perception related topics to /tmp/*
# usage: ./play_bag_without_perception.sh <bag_name> <start offset>

BAG_NAME=$1
OFFSET=${2:-0.001}

# topics not to be renamed
SKIP_TOPICS=(
     "/sensing/radar/*" # keep this if you want radar objects to be remained
     "/perception/object_recognition/detection/rois*"
)

PERCEPTION_TOPIC=$(ros2 bag info "$BAG_NAME" | awk '{print $2}' | grep -e centerpoint -e objects -e prediction/ -e objects_raw)
COMMAND_OPTION='--remap '

for topic in ${PERCEPTION_TOPIC[@]}; do
    # skip specific topics
    skip=0
    for skip_pattern in "${SKIP_TOPICS[@]}"; do
        if [[ $topic == $skip_pattern ]]; then
            skip=1
            break
        fi
    done
    if [[ $skip -eq 1 ]]; then
        continue
    fi

    # remap to /tmp/*
    RENAMED_TARGET_PERCEPTION_TOPIC=$(echo "$topic" | sed 's,^/,\0tmp/,')
    COMMAND_OPTION+=" $topic:=$RENAMED_TARGET_PERCEPTION_TOPIC"
done

ros2 bag play "$BAG_NAME" $COMMAND_OPTION -r 0.2 --clock 200 -s sqlite3 --start-offset $OFFSET