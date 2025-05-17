import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class SensorSim(Node):
    def __init__(self):
        super().__init__('sensor_sim_node')
        self.pub = self.create_publisher(Float32MultiArray, '/bscan_raw', 10)
        self.timer = self.create_timer(1.0, self.generate_bscan)
        self.counter = 0

    def generate_bscan(self):
        x = np.arange(512)
        with_mine = self.counter % 10 < 5  # каждые 5 из 10 итераций - мина
        self.get_logger().info(f"Publishing with_mine = {with_mine}")
        if with_mine:
            bscan = np.random.normal(0, 0.1, (256, 512))
            bscan[100:110] += np.exp(-((x - 256)**2) / 1000)
        else:
            bscan = np.random.normal(0, 0.02, (256, 512))
        self.counter += 1
        msg = Float32MultiArray()
        msg.data = bscan.flatten().tolist()
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()