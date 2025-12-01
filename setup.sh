#!/bin/bash
set -e # Прекращаем выполнение при любой ошибке

echo "🚀 Начинаем установку зависимостей для подсистемы распознавания объектов"

# Шаг 1: Обновление системы и установка базовых зависимостей
echo "🔧 Шаг 1: Обновление системы и установка базовых зависимостей"
sudo apt update
sudo apt install -y python3-dev python3-dev libatlas-base-dev libopenblas-dev liblapack-dev gfortran \
  libsnappy-dev libsystemd-dev pkg-config build-essential curl wget libspatialindex-dev

# Шаг 2: Проверка архитектуры системы
ARCH=$(dpkg --print-architecture)
echo "🔍 Обнаружена архитектура системы: $ARCH"

# Шаг 3: Установка библиотек для Google Coral TPU
echo "🔌 Шаг 2: Установка библиотек для Google Coral TPU"
if [ "$ARCH" = "armhf" ]; then
  echo "⬇️  Скачивание 32-битной версии libedgetpu"
  wget -q https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-max_16.0tf2.17.1-1.bookworm_armhf.deb
  sudo dpkg -i libedgetpu1-max_16.0tf2.17.1-1.bookworm_armhf.deb || sudo apt --fix-broken install -y
elif [ "$ARCH" = "arm64" ]; then
  echo "⬇️  Скачивание 64-битной версии libedgetpu"
  wget -q https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-max_16.0tf2.17.1-1.bookworm_arm64.deb
  sudo dpkg -i libedgetpu1-max_16.0tf2.17.1-1.bookworm_arm64.deb || sudo apt --fix-broken install -y
else
  echo "❌ Неподдерживаемая архитектура: $ARCH"
  exit 1
fi

# Шаг 4: Установка системных Python-пакетов (быстрее и стабильнее для RPi)
echo "🐍 Шаг 3: Установка системных Python-пакетов"
sudo apt install -y python3-opencv python3-numpy python3-scipy python3-pil python3-sklearn python3-shapely

# Шаг 5: Установка ROS-зависимостей (предполагается, что ROS уже установлен)
echo "🤖 Шаг 4: Установка ROS-зависимостей"
sudo apt install -y ros-$(source /opt/ros/*/setup.sh && rosversion -d)-rospy \
  ros-$(source /opt/ros/*/setup.sh && rosversion -d)-sensor-msgs \
  ros-$(source /opt/ros/*/setup.sh && rosversion -d)-cv-bridge

# Шаг 6: Создание виртуального окружения
echo "🌱 Шаг 6: Создание виртуального окружения"
if [ ! -d "cow_env" ]; then
  uv venv cow_env --system-site-packages --python=python3.11
fi
source cow_env/bin/activate

# Шаг 7: Установка Python-зависимостей
echo "⚙️ Шаг 7: Установка Python-зависимостей"
uv pip install tflite-runtime
uv pip install "numpy<2" trimesh rtree tqdm
uv pip install --no-deps supervision alphashape

# Конец работы установщика
echo "🎉 Установка завершена успешно!"
exit 0
