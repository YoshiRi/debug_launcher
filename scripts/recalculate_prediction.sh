#!/bin/bash

# usage ./recalculate_prediction.sh -b <bagfile> -o <offset> -r <rate> --rviz_config <rviz_config>

# variables
BAGFILE=""
OFFSET=0.001
RATE=0.2
RVIZ_CONFIG=""
while getopts "b:o:r:-:" opt; do
  case $opt in
    b)
      BAGFILE=$OPTARG
      ;;
    o)
      OFFSET=$OPTARG
      ;;
    r)
      RATE=$OPTARG
      ;;
    -)
      case $OPTARG in
        rviz_config)
          RVIZ_CONFIG=$OPTARG
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# 0. setup.bash path
SETUP_PATH=$COLCON_PREFIX_PATH/setup.bash
SETUP_COMMAND="source $SETUP_PATH"
echo "Run in SETUP_PATH: $SETUP_PATH"

# 1. launch autoware first
if [ -z "$RVIZ_CONFIG" ]; then
  LAUNCH_COMMAND="ros2 launch debug_launcher rerun_prediction_only.launch.xml"
else
  LAUNCH_COMMAND="ros2 launch debug_launcher rerun_prediction_only.launch.xml rviz_config:=$RVIZ_CONFIG"
fi

# 2. play bag file
PLAY_COMMAND="ros2 bag play $BAGFILE --clock --remap /perception/object_recognition/objects:=/orig_objects /perception/object_recognition/prediction/map_based_prediction/debug/processing_time_detail_ms:=/tmp/perception/object_recognition/prediction/map_based_prediction/debug/processing_time_detail_ms -r $RATE --start-offset $OFFSET"

# 3. visualize processing time
VISUALIZE_COMMAND="ros2 run autoware_debug_tools processing_time_visualizer -t /perception/object_recognition/prediction/map_based_prediction/debug/processing_time_detail_ms"


# 1. launch autoware first in a new terminal
gnome-terminal -- bash -c "$SETUP_COMMAND; $LAUNCH_COMMAND;" &
LAUNCH_PID=$!
echo "LAUNCH_PID: $LAUNCH_PID"

# 2. play bag file in another new terminal after 5 seconds
sleep 5
gnome-terminal -- bash -c "$SETUP_COMMAND; $PLAY_COMMAND;" &
PLAY_PID=$!

# 3. visualize processing time in another new terminal after 5 seconds
sleep 5
$VISUALIZE_COMMAND
# gnome-terminal -- bash -c "$SETUP_COMMAND; $VISUALIZE_COMMAND; exec bash" &
# VISUALIZE_PID=$!

# # 4. wait for the play to finish
# wait $PLAY_PID
# echo "Play finished"

# # 5. cleanup: kill the launch and visualize processes
# kill $LAUNCH_PID
# kill $VISUALIZE_PID
