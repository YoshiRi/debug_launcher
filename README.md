# Debug launcher

lightweight launcher for debugging

## launch map and rviz only

- Just show map information in rviz


```
ros2 launch debug_launcher map.launch.xml map_path:=$HOME/<your_map_path> rviz_config:=$HOME/<your_rviz_config>
```

- default map path is `$HOME/autoware_map/odaiba`
- default rviz config path is `$(this package)/rviz/debug.rviz`