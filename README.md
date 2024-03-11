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

When play rosbag, you need to remap existing tracking/prediction topics related to this launcher.

```bash
ros2 run debug_launcher play_bag_before_tracking.sh -b <rosbag> -o <offset>
```


## scripts

### `play_bag_without_perception.sh`

This script plays rosbag witt perception topics remapped.

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