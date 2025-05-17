import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class PreprocNode(Node):
    def __init__(self):
        super().__init__('preproc_node')
        self.sub = self.create_subscription(Float32MultiArray, '/bscan_raw', self.callback, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/bscan_clean', 10)
        self.bg = None

    def callback(self, msg):
        arr = np.array(msg.data, dtype=np.float32).reshape((256, 512))
        if self.bg is None:
            self.bg = arr
            self.get_logger().info("Background captured.")
            return
        cleaned = arr - self.bg
        normed = (cleaned - cleaned.min()) / (cleaned.max() - cleaned.min() + 1e-6)
        out = Float32MultiArray()
        out.data = normed.flatten().tolist()
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = PreprocNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()