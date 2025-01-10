import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace

def generate_launch_description():
    # Get the package directory
    bringup_dir = get_package_share_directory('sim_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Paths for URDFs and SDFs
    urdf_paths = [
        os.path.join(bringup_dir, 'urdf', 'robot1.urdf'),
        os.path.join(bringup_dir, 'urdf', 'robot2.urdf'),
        os.path.join(bringup_dir, 'urdf', 'robot3.urdf'),
        os.path.join(bringup_dir, 'urdf', 'robot4.urdf'),
    ]
    
    sdf_paths = [
        os.path.join(bringup_dir, 'models', 'turtlebot3_waffle', 'mybot1.sdf'),
        os.path.join(bringup_dir, 'models', 'turtlebot3_waffle', 'mybot2.sdf'),
        os.path.join(bringup_dir, 'models', 'turtlebot3_waffle', 'mybot3.sdf'),
        os.path.join(bringup_dir, 'models', 'turtlebot3_waffle', 'mybot4.sdf'),
    ]
    
    # Load robot descriptions
    robot_descriptions = []
    for urdf_path in urdf_paths:
        with open(urdf_path, 'r') as urdf_file:
            robot_descriptions.append(urdf_file.read())
    
    yaml_file = os.path.join('src', 'params.yaml')
    with open(yaml_file, 'r') as file:
        robots_config = yaml.safe_load(file)

    spawn_robots = []
    def spawn_robot_group(robot_name, robot_type_index, x, y, z, Y, m):
        namespace = robot_name
        robot_description = robot_descriptions[robot_type_index]

        if m == 1:
            return GroupAction([
                PushRosNamespace(namespace),

                # Robot State Publisher
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    output='screen',
                    parameters=[{'use_sim_time': True, 'robot_description': robot_description}]
                ),

                # Spawn robot in Gazebo
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    name=f'spawn_{namespace}',
                    output='screen',
                    arguments=[
                        '-entity', namespace,
                        '-file', sdf_paths[robot_type_index],
                        '-robot_namespace', namespace,
                        '-x', str(x), '-y', str(y), '-z', str(z), '-Y', str(Y)
                    ]
                ),
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name=f'static_tf_pub1_{namespace}',
                    output='screen',
                    arguments=[
                        str(x), str(y), str(z), '0', '0', '0',
                        'odom',
                        f'{namespace}/base_footprint'
                    ]
                )
            ])
        else:
            return GroupAction([
                PushRosNamespace(namespace),
                 Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    name=f'spawn_{namespace}',
                    output='screen',
                    arguments=[
                        '-entity', namespace,
                        '-file', sdf_paths[robot_type_index],
                        '-robot_namespace', namespace,
                        '-x', str(x), '-y', str(y), '-z', str(z), '-Y', str(Y)
                    ]
                )
            ])
    for robot_type, robot_list in robots_config.items():
        if robot_type == 'robots':
            for i, robot in enumerate(robot_list):
                spawn_robots.append(spawn_robot_group(robot['name'], i, robot['x'], robot['y'], robot['z'], robot['Y'], 1))
    
    world = os.path.join(
        get_package_share_directory('sim_gazebo'),
        'worlds',
        'turtlebot3_factory_1.world'
    )
    
    # Start Gazebo
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    rviz_config_path = os.path.join(bringup_dir, 'rviz', '4bots.rviz')

    rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )

    custom_node = Node(
        package='turtlebot3_rl',  # Replace with your package name
        executable='det_obs',  # Replace with your node executable name
        name='det_obs',        # Optional: custom name for the node
        output='screen',)

    # Create launch description
    ld = LaunchDescription()

    # Declare arguments
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock'))

    # Add Gazebo and RVIZ
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(rviz_cmd)
    # ld.add_action(custom_node)

    # Add robot spawns
    for spawn_robot in spawn_robots:
        ld.add_action(spawn_robot)
    
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_map_to_odom',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    ))

    return ld
