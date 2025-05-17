# FROM ros:humble

# # Установка зависимостей
# RUN apt update && apt install -y \
#     python3-pip \
#     python3-colcon-common-extensions \
#     ros-humble-rclpy \
#     ros-humble-std-msgs \
#     python3-opencv

# RUN pip3 install numpy tqdm scikit-learn torch torchvision timm onnx onnxruntime matplotlib

# # Создаем рабочую директорию
# WORKDIR /ros2_ws
# ---------- BASE ----------
    FROM ros:humble

    # ---------- System deps ----------
    RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends \
            python3-pip \
            python3-colcon-common-extensions \
            ros-humble-rclpy \
            ros-humble-std-msgs && \
        apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # ---------- Python deps ----------
    COPY requirements.txt /tmp/reqs.txt
    RUN pip3 install --no-cache-dir -r /tmp/reqs.txt
    
    # ---------- Copy workspace ----------
    COPY . /ros2_ws
    WORKDIR /ros2_ws
    RUN colcon build --symlink-install
