# Update system packages
!apt update && apt upgrade -y

# Install necessary system dependencies
!apt install -y python3-opengl ffmpeg xvfb

# Upgrade pip
!pip install --upgrade pip setuptools wheel

# Install MuJoCo and Stable-Baselines3 with correct dependencies
!pip install mujoco==3.3.0
!pip install gymnasium[all] stable-baselines3[extra]
!pip install torch torchvision torchaudio transformers
!pip install numpy opencv-python imageio tqdm
!pip install google-colab
!pip install pickle5
!pip uninstall -y stable-baselines3
!pip install stable-baselines3[extra]
!pip install gymnasium[mujoco]
