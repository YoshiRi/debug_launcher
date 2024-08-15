#!/bin/bash
# usage ./recalculate_prediction.sh -b <bagfile> -o <offset> -r <rate> --map <map_name> --rviz_config <rviz_config> --do-not-run-visualization --run-tracking
#       description: recalculate prediction based on the bagfile and visualize the processing time
#       -b: bagfile path
#       -o: start offset
#       -r: play rate
#       --map: map name (look for the ~/autoware_map/<map_name> folder)
#       --rviz_config: rviz config file
#       --do-not-run-visualization: do not run the processing time visualization
#       --run-tracking: run tracking node


# variables
BAGFILE=""
OFFSET=0.001
RATE=0.2
RVIZ_CONFIG=""
DO_NOT_RUN_VISUALIZATION=false
MAP_NAME="odaiba"
RUN_TRACKING=false
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
        map)
          MAP_NAME=$OPTARG
          ;;
        rviz_config)
          RVIZ_CONFIG=$OPTARG
          ;;
        do-not-run-visualization)
            DO_NOT_RUN_VISUALIZATION=true
            ;;
        run-tracking)
            RUN_TRACKING=true
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
if [ "$RUN_TRACKING" = true ]; then
  LAUNCH_COMMAND="ros2 launch debug_launcher rerun_tracking_and_prediction.launch.xml"
else
  LAUNCH_COMMAND="ros2 launch debug_launcher rerun_prediction_only.launch.xml"
fi

if [ -z "$RVIZ_CONFIG" ]; then
  LAUNCH_OPTION="map:=$MAP_NAME"
else
  LAUNCH_OPTION="map:=$MAP_NAME rviz_config:=$RVIZ_CONFIG"
fi

LAUNCH_COMMAND="$LAUNCH_COMMAND $LAUNCH_OPTION"
echo "Launch command: $LAUNCH_COMMAND"

# 2. play bag file
BASE_PLAY_COMMAND="ros2 bag play $BAGFILE "
PREDICTION_REMAP="/perception/object_recognition/objects:=/tmp/prediction/objects \
/perception/object_recognition/prediction/map_based_prediction/debug/processing_time_detail_ms:=/tmp/prediction/processing_time_detail_ms \
/perception/object_recognition/prediction/map_based_prediction/debug/processing_time_ms:=/tmp/prediction/processing_time_ms \
/perception/object_recognition/prediction/map_based_prediction/debug/cyclic_time_ms:=/tmp/prediction/cyclic_time_ms"
TRACKING_REMAP="/perception/object_recognition/tracking/objects:=/tmp/tracking/objects \
/perception/object_recognition/tracking/debug/processing_time_detail_ms:=/tmp/tracking/processing_time_detail_ms \
/perception/object_recognition/tracking/debug/processing_time_ms:=/tmp/tracking/processing_time_ms \
/perception/object_recognition/tracking/debug/cyclic_time_ms:=/tmp/tracking/cyclic_time_ms"
if [ "$RUN_TRACKING" = true ]; then
  PLAY_COMMAND="$BASE_PLAY_COMMAND --remap $TRACKING_REMAP $PREDICTION_REMAP --clock 200  -r $RATE --start-offset $OFFSET"
else
  PLAY_COMMAND="$BASE_PLAY_COMMAND --remap $PREDICTION_REMAP --clock 200  -r $RATE --start-offset $OFFSET"
fi
# echo "Play command: $PLAY_COMMAND"


# 3. visualize processing time
VISUALIZE_COMMAND="ros2 run autoware_debug_tools processing_time_visualizer -t /perception/object_recognition/prediction/map_based_prediction/debug/processing_time_detail_ms"


# 1. launch autoware first in a new terminal
gnome-terminal -- bash -c "$SETUP_COMMAND; $LAUNCH_COMMAND;" & # add exec bash; when you want to keep the terminal open

# 2. play bag file in another new terminal after 5 seconds
sleep 5
gnome-terminal -- bash -c "$SETUP_COMMAND; $PLAY_COMMAND;" &

# 3. visualize processing time in another new terminal after 5 seconds
sleep 7 # need to wait until the topic is published
if [ "$DO_NOT_RUN_VISUALIZATION" = false ]; then
    $VISUALIZE_COMMAND
fi

LAUNCH_PID=$(pgrep -f "$LAUNCH_COMMAND")
# echo "LAUNCH_PID: $LAUNCH_PID"

# 4. wait for the play to finish
while pgrep -f "$BASE_PLAY_COMMAND" > /dev/null; do
    sleep 1
done
echo "Play finished. Kill the launch"

# 5. kill the launch
kill $LAUNCH_PID
