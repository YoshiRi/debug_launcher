#!/bin/bash

# remap perception related topics to /tmp/*
# usage: ./play_bag_before_tracking.sh -b <bag_name> -o <start offset> -r <rate> --remap-radar --remap-detection

REMAP_RADAR=0
REMAP_DETECTION=0
OFFSET=0.001
RATE=0.2

while getopts "b:o:r:-:" opt; do
  case $opt in
    b)
      BAG_NAME=$OPTARG
      ;;
    o)
      OFFSET=$OPTARG
      ;;
    r)
      RATE=$OPTARG
      ;;
    -)
      case $OPTARG in
        remap-radar)
          REMAP_RADAR=1
          ;;
        remap-detection)
          REMAP_DETECTION=1
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

# topics to be renamed
BASIC_REMAP_TOPICS=(
    "/perception/object_recognition/tracking/near_objects"
    "/perception/object_recognition/tracking/objects"
    "/perception/object_recognition/objects"
)
DETECTION_REMAP_TOPICS=(
    "/perception/object_recognition/detection/objects"
    "/perception/object_recognition/detection/detection_by_tracker/objects"
)
RADAR_REMAP_TOPICS=(
    "/perception/object_recognition/tracking/radar_objects"
)

# Initialize the perception remap topic array with the basic topics
PERCEPTION_REMAP_TOPICS=("${BASIC_REMAP_TOPICS[@]}")

# Add radar topics if requested
if [ $REMAP_RADAR -eq 1 ]; then
    PERCEPTION_REMAP_TOPICS+=("${RADAR_REMAP_TOPICS[@]}")
fi

# Add detection topics if requested
if [ $REMAP_DETECTION -eq 1 ]; then
    PERCEPTION_REMAP_TOPICS+=("${DETECTION_REMAP_TOPICS[@]}")
fi

COMMAND_OPTION='--remap '

for topic in ${PERCEPTION_REMAP_TOPICS[@]}; do
    # remap to /tmp/*
    RENAMED_PERCEPTION_REMAP_TOPICS=$(echo "$topic" | sed 's,^/,\0tmp/,')
    COMMAND_OPTION+=" $topic:=$RENAMED_PERCEPTION_REMAP_TOPICS"
done

# switch sqlite3 or mcap depending on the bag file format
if [[ $(ros2 bag info "$BAG_NAME" | grep -c sqlite3) -eq 1 ]]; then
    COMMAND_OPTION+=" --storage sqlite3"
else
    COMMAND_OPTION+=" --storage mcap"
fi

ros2 bag play "$BAG_NAME" $COMMAND_OPTION -r $RATE --clock 200  --start-offset $OFFSET
