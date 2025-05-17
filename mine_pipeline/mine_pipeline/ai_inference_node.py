from mine_pipeline.inference_logger import InferenceLogger
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np
import scipy.special
import cv2

class ONNXInferenceNode(Node):
    def __init__(self):
        super().__init__('onnx_inference_node')
        self.logger = InferenceLogger()
        self.sub = self.create_subscription(Float32MultiArray, '/bscan_clean', self.callback, 10)
        self.pub = self.create_publisher(Float32, '/alarm_score', 10)
        self.session = ort.InferenceSession("swin_mine_detector.onnx")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def callback(self, msg):
        arr = np.array(msg.data, dtype=np.float32).reshape((256, 512))
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.stack([arr]*3, axis=0)  # (3, 256, 512)
        arr = np.expand_dims(arr, axis=0)  # (1, 3, 256, 512)
        # корректный resize с сохранением пространственной структуры:
        arr = arr[0]  # убираем batch размерность
        resized = np.stack([
            cv2.resize(arr[c], (224, 224), interpolation=cv2.INTER_LINEAR)
            for c in range(3)
        ], axis=0)
        arr_resized = np.expand_dims(resized, axis=0).astype(np.float32)  # (1, 3, 224, 224)
        # --- инференс ---
        result = self.session.run([self.output_name], {self.input_name: arr_resized})[0]  # shape (1, 2)

        # --- логиты до softmax ---
        logits = result[0]
        self.get_logger().info(
            f'Logits before softmax: {logits.tolist()} '
            f'(min={logits.min():.3f}, max={logits.max():.3f})'
        )

        # --- softmax + score ---
        probs = scipy.special.softmax(logits)
        score = float(probs[1])  # вероятность класса "мина"

        self.logger.log(score)
        self.get_logger().info(f"[ONNX] Inference score: {score:.3f} → {'ALARM' if score > 0.95 else 'Clear'}")

        out = Float32()
        out.data = score
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = ONNXInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()