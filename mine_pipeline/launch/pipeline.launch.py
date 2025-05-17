from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mine_pipeline',
            executable='sensor_sim_node',
            name='sensor_sim_node'
        ),
        Node(
            package='mine_pipeline',
            executable='preproc_node',
            name='preproc_node'
        ),
        Node(
            package='mine_pipeline',
            executable='ai_inference_node',
            name='ai_inference_node'
        ),
        Node(
            package='mine_pipeline',
            executable='alarm_node',
            name='alarm_node'
        ),
    ])