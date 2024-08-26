# Debug launcher

lightweight launcher for debugging

## installation

This clone and build this package in your autoware workspace.

```bash
cd src
git clone https://github.com/YoshiRi/debug_launcher.git
cd ..
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select debug_launcher
```

## launcher

### `map.launch.xml`: launch map and rviz only

Just show map information in rviz.


Usage:
```bash
ros2 launch debug_launcher map.launch.xml map_path:=$HOME/<your_map_path> rviz_config:=$HOME/<your_rviz_config>
```

or

```bash
ros2 launch debug_launcher map.launch.xml map_name:=<your_map_name> rviz_config:=$HOME/<your_rviz_config>
```

| arg | description |
| --- | --- |
| map_path | path to map directory. |
| map_name | name of map file. If you set this, the launcher will look for `$(env HOME)/autoware_map/$(var map_name)`.|
| rviz_config | path to rviz config file. default rviz config path is `$(this package)/rviz/debug.rviz`. |


Then you can just run ros2 bag play and see the results in rviz.

### `rerun_tracking_and_prediction.launch.xml`: rerun tracking and prediction from existing rosbag

This launcher runs lidar/radar based tracking and prediction based on existing rosbag detection messages.

Usage:
```bash
ros2 launch debug_launcher rerun_tracking_and_prediction.launch.xml map_name:=<your_map_name>
```

| arg | description |
| --- | --- |
| map_name | name of map file. If you set this, the launcher will look for `$(env HOME)/autoware_map/$(var map_name)`.|
| use_tracking_merger | if true, use tracking merger. Enable this option when you want to check with radar config. |
| use_universe_path | if true, use universe path for config for tracking/prediction. It makes easier for you to check your change quickly. If false, nodes use parameters in autoware_launch. |


When play rosbag, you need to remap existing tracking/prediction topics related to this launcher.

```bash
ros2 run debug_launcher play_bag_before_tracking.sh -b <rosbag> -o <offset>
```


## scripts

For general assumptions, map files are considered to be located in `$(env HOME)/autoware_map/$(var map_name)`.

### `play_bag_without_perception.sh`

This script plays rosbag witt perception topics remapped.
In the default, most perception topics and pointcloud topics are remapped to `/tmp/<original topic name>` namespace.

Usage:
```bash
ros2 run debug_launcher play_bag_without_perception.sh <your bag file>
```

### `play_bag_before_tracking.sh`

Compared to `play_bag_without_perception.sh`, this script remaps only and tracking/prediction topics.

Usage:
```bash
ros2 run debug_launcher play_bag_before_tracking.sh -b <rosbag> -o <offset>
```


<!-- ros2 run debug_launcher recalculate_prediction.sh -b $HOME/autoware_bag/SPkadai/perf/xx1p1/converted/ --map large_scale_odaiba_beta --run-tracking -r 2.5 -->
## `recalculate_prediction.sh`

This script runs tracking and prediction based on existing rosbag detection messages.
This will open three terminals and run `ros2 bag play`, `main launcher`, and `processing time recorder` and create `all_processing_time.csv` file to save results.

Usage:
```bash
ros2 run debug_launcher recalculate_prediction.sh -b <rosbag> -m <map_name> -r <replay rate>
```

main options:
| arg | description |
| --- | --- |
| -b | path to rosbag file. |
| -m | name of map file. If you set this, the launcher will look for `$(env HOME)/autoware_map/$(var map_name)`.|
| -r | replay rate. |
| -o | offset. |
| --run-tracking | if set, run tracking either. |
| --rviz_config | path to rviz config file. default rviz config path is `$(this package)/rviz/debug.rviz`. |