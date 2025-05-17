import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

class AlarmNode(Node):
    def __init__(self):
        super().__init__('alarm_node')
        self.sub = self.create_subscription(Float32, '/alarm_score', self.callback, 10)
        self.pub = self.create_publisher(Bool, '/alarm', 10)
        self.threshold = 0.8

    def callback(self, msg):
        alarm = msg.data > self.threshold
        out = Bool()
        out.data = alarm
        self.pub.publish(out)
        self.get_logger().info(f'Score: {msg.data:.3f} → Alarm: {alarm}')

def main(args=None):
    rclpy.init(args=args)
    node = AlarmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()