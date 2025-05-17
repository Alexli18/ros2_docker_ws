FROM ros:humble

# Установка зависимостей
RUN apt update && apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    python3-opencv

RUN pip3 install numpy tqdm scikit-learn torch torchvision timm onnx onnxruntime matplotlib

# Создаем рабочую директорию
WORKDIR /ros2_ws
