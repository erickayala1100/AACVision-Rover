# Raspberry Pi Setup

## Existing AAC Vision environment

The tested rover used:

```text
/home/pi/aacvision-env
```

Activate it with:

```bash
source /home/pi/aacvision-env/bin/activate
```

Confirm key imports:

```bash
python -c "import cv2, numpy, pyrealsense2, RPi.GPIO; print('camera/GPIO imports OK')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "from breezyslam.algorithms import RMHC_SLAM; print('BreezySLAM OK')"
```

## Model

The main version uses:

```text
yolo26n.pt
```

The model class list must contain `sports ball`.

```bash
cd /home/pi
/home/pi/aacvision-env/bin/python -c "from ultralytics import YOLO; m=YOLO('yolo26n.pt'); print(m.names)"
```

## RPLIDAR SDK

The main program expects:

```text
/home/pi/rplidar_sdk/output/Linux/Release/ultra_simple
```

and normally connects to:

```text
/dev/ttyUSB0
```

These constants can be changed in the main Python source when the installation differs.

## Repository installation

```bash
cd /home/pi
git clone https://github.com/YOUR-GITHUB-USERNAME/AACVision-Rover.git
cd AACVision-Rover
chmod +x scripts/*.sh
./scripts/check_environment.sh
```

## Wheels-up test

```bash
./scripts/run_v8_3.sh
```

Keep the rover raised and place a stationary orange basketball approximately 1.5-3 m in front of the RealSense camera.
