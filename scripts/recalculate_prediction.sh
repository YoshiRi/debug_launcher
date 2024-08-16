#!/bin/bash
# usage ./recalculate_prediction.sh -b <bagfile> -o <offset> -r <rate> -v --map <map_name> --rviz_config <rviz_config> --no-visualization --run-tracking
#       description: recalculate prediction based on the bagfile and visualize the processing time
#       -b: bagfile path
#       -o: start offset
#       -r: play rate
#       -v: show debug message
#       --map: map name (look for the ~/autoware_map/<map_name> folder)
#       --rviz_config: rviz config file
#       --no-visualization: do not run the processing time visualization
#       --run-tracking: run tracking node

# variables
BAGFILE=""
OFFSET=0.001
RATE=0.2
VERBOSE=false
RVIZ_CONFIG=""
DO_NOT_RUN_VISUALIZATION=false
MAP_NAME="odaiba"
RUN_TRACKING=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -b)
      BAGFILE="$2"
      shift 2
      ;;
    -o)
      OFFSET="$2"
      shift 2
      ;;
    -r)
      RATE="$2"
      shift 2
      ;;
    -v)
      VERBOSE=true
      shift
      ;;
    --map)
      MAP_NAME="$2"
      shift 2
      ;;
    --rviz_config)
      RVIZ_CONFIG="$2"
      shift 2
      ;;
    --no-visualization)
      DO_NOT_RUN_VISUALIZATION=true
      shift
      ;;
    --run-tracking)
      RUN_TRACKING=true
      shift
      ;;
    *)
      echo "Invalid option: $1" >&2
      exit 1
      ;;
  esac
done


# 0. setup.bash path
SETUP_PATH=$COLCON_PREFIX_PATH/setup.bash
SETUP_COMMAND="source $SETUP_PATH"
echo "Run in SETUP_PATH: $SETUP_PATH"
echo "RUN_TRACKING: $RUN_TRACKING"
echo "MAP_NAME: $MAP_NAME"


# 1. launch autoware first
if [ "$RUN_TRACKING" = true ]; then
  BASE_LAUNCH_COMMAND="ros2 launch debug_launcher rerun_tracking_and_prediction.launch.xml"
else
  BASE_LAUNCH_COMMAND="ros2 launch debug_launcher rerun_prediction_only.launch.xml"
fi

if [ -z "$RVIZ_CONFIG" ]; then
  LAUNCH_OPTION="map_name:=$MAP_NAME"
else
  LAUNCH_OPTION="map_name:=$MAP_NAME rviz_config:=$RVIZ_CONFIG"
fi

LAUNCH_COMMAND="$BASE_LAUNCH_COMMAND $LAUNCH_OPTION"


# 2. play bag file
BASE_PLAY_COMMAND="ros2 bag play $BAGFILE"
PREDICTION_REMAP="/perception/object_recognition/objects:=/tmp/prediction/objects \
/perception/object_recognition/prediction/map_based_prediction/debug/processing_time_detail_ms:=/tmp/prediction/processing_time_detail_ms \
/perception/object_recognition/prediction/map_based_prediction/debug/processing_time_ms:=/tmp/prediction/processing_time_ms \
/perception/object_recognition/prediction/map_based_prediction/debug/cyclic_time_ms:=/tmp/prediction/cyclic_time_ms"
TRACKING_REMAP="/perception/object_recognition/tracking/objects:=/tmp/tracking/objects \
/perception/object_recognition/tracking/debug/processing_time_detail_ms:=/tmp/tracking/processing_time_detail_ms \
/perception/object_recognition/tracking/debug/processing_time_ms:=/tmp/tracking/processing_time_ms \
/perception/object_recognition/tracking/debug/cyclic_time_ms:=/tmp/tracking/cyclic_time_ms"
if [ "$RUN_TRACKING" = true ]; then
  PLAY_COMMAND="$BASE_PLAY_COMMAND --remap $TRACKING_REMAP $PREDICTION_REMAP --clock 200 -r $RATE --start-offset $OFFSET"
else
  PLAY_COMMAND="$BASE_PLAY_COMMAND --remap $PREDICTION_REMAP --clock 200 -r $RATE --start-offset $OFFSET"
fi


# debug message
if [ "$VERBOSE" = true ]; then
    echo "Launch command: $LAUNCH_COMMAND"
    echo "Play command: $PLAY_COMMAND"
fi


# 3. visualize processing time
VISUALIZE_COMMAND="ros2 run autoware_debug_tools processing_time_visualizer -t /perception/object_recognition/prediction/map_based_prediction/debug/processing_time_detail_ms"

RECORD_COMMAND="ros2 run debug_launcher record_processing_time.py -t /perception/object_recognition/prediction/map_based_prediction/debug/processing_time_ms"

# 4. run the commands
# 4-1. launch autoware first in a new terminal
gnome-terminal -- bash -c "$SETUP_COMMAND; $LAUNCH_COMMAND;" & # add exec bash; when you want to keep the terminal open

# 4-2. play bag file in another new terminal after 5 seconds
sleep 5
gnome-terminal -- bash -c "$SETUP_COMMAND; $PLAY_COMMAND;" &

# 4-2-b. record processing time in another new terminal after 5 seconds
gnome-terminal -- bash -c "$SETUP_COMMAND; $RECORD_COMMAND; exec bash;" & 

# 4-3. visualize processing time in another new terminal after 5 seconds
sleep 7 # need to wait until the topic is published
if [ "$DO_NOT_RUN_VISUALIZATION" = false ]; then
    $VISUALIZE_COMMAND
fi

# 4-4. get the launch pid and wait for the play to finish
while pgrep -f "$BASE_PLAY_COMMAND" > /dev/null; do
    sleep 1
done
echo "Play finished. Kill the launch"
LAUNCH_PID=$(pgrep -f "$LAUNCH_COMMAND")
RECORD_PID=$(pgrep -f "$RECORD_COMMAND")

# 5. kill the launch
kill -s SIGINT $LAUNCH_PID
kill -s SIGINT $RECORD_PID