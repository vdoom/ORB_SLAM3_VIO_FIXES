# RPi Camera Viewer

Qt6 application for viewing CSI camera and IMU data on Raspberry Pi 5, with ROS bag recording for Kalibr camera-IMU calibration.

## Features

- Real-time camera preview using libcamera
- Live 3D IMU orientation visualization
- **ROS bag recording** for Kalibr calibration (camera + IMU data)
- FPS and frame counter displayed in window title
- Fullscreen mode (F or F11)
- Pause/resume (Space)

## ROS Bag Recording

The application can record camera frames and IMU data to ROS1 bag format for use with [Kalibr](https://github.com/ethz-asl/kalibr) camera-IMU calibration tool.

### Recording Features

- **No ROS dependencies** - standalone implementation of ROS1 bag format v2.0
- Records grayscale camera frames (`sensor_msgs/Image`, mono8 encoding)
- Records IMU data (`sensor_msgs/Imu`) with accelerometer and gyroscope
- Compatible with `rosbag play` and Kalibr

### Topics

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/cam0/image_raw` | sensor_msgs/Image | Grayscale camera frames |
| `/imu0` | sensor_msgs/Imu | IMU accelerometer + gyroscope |

### Usage

1. Click the **"Record Bag"** button in the toolbar to start recording
2. Move the camera/IMU through various orientations for calibration
3. Click the button again to stop recording
4. The bag file is saved to `~/rosbag_YYYYMMDD_HHMMSS.bag`

### Using with Kalibr

```bash
# Verify the bag file
rosbag info ~/rosbag_*.bag

# Run Kalibr camera-IMU calibration
kalibr_calibrate_imu_camera \
    --target april_grid.yaml \
    --cam camchain.yaml \
    --imu imu.yaml \
    --bag ~/rosbag_*.bag
```

## Dependencies

Install on Raspberry Pi OS (Bookworm):

```bash
# Qt6 development packages
sudo apt install qt6-base-dev libqt6openglwidgets6 qt6-base-dev-tools

# libcamera (usually pre-installed on Pi OS)
sudo apt install libcamera-dev

# Build tools
sudo apt install cmake build-essential pkg-config
```

## Building

```bash
./build.sh
```

Or manually:

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Running

```bash
./build/rpi_camera_viewer
```

Or with specific display (for remote/SSH):
```bash
DISPLAY=:0 ./build/rpi_camera_viewer
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| Q / Esc | Quit |
| F / F11 | Toggle fullscreen |
| Space | Pause/resume camera |

## IMU Setup

The application reads IMU data from a serial port (default: `/dev/ttyACM0` at 115200 baud).

### Expected IMU Data Format

The IMU should send ASCII data in this format:
```
timestamp<ms>; [INT #<count>] A: <ax>,<ay>,<az> g | G: <gx>,<gy>,<gz> °/s
```

Example:
```
timestamp1234567890; [INT #123] A: 0.01,0.02,1.00 g | G: 1.5,0.5,-0.3 °/s
```

### Supported Hardware

- Raspberry Pi Pico with BMI160 IMU (see `BMI160+INNO-MAKER_CAM-OV9281RAW-V2+RPi3/RPi_Pico_code/`)
- Any microcontroller sending data in the expected format

## Camera Selection

By default, camera 0 (first CSI camera) is used. The application requests R8 (8-bit grayscale) format from libcamera for optimal performance with monochrome sensors like OV9281.

## Troubleshooting

### "No cameras found"
- Check camera ribbon cable connection
- Enable camera in `raspi-config`
- Verify with `libcamera-hello --list-cameras`

### Permission denied
```bash
sudo usermod -aG video $USER
sudo usermod -aG dialout $USER  # For serial/IMU access
# Log out and back in
```

### No IMU data
- Check serial port: `ls -l /dev/ttyACM*`
- Test with: `screen /dev/ttyACM0 115200`
- Verify Pico firmware is running

### Low FPS
- The default resolution is 1280x800
- Monochrome sensors (OV9281) perform better than color sensors
- Close other applications using the camera

## Architecture

```
ext_view_app/
├── src/
│   ├── main.cpp              # Application entry point
│   ├── MainWindow.h/cpp      # Main window with toolbar
│   ├── CameraController.h/cpp # libcamera interface
│   ├── CameraWidget.h/cpp    # Camera preview widget
│   ├── SerialIMUReader.h/cpp # Serial IMU data reader
│   ├── Axes3DWidget.h/cpp    # 3D IMU visualization (OpenGL)
│   ├── RosBagWriter.h/cpp    # Standalone ROS1 bag writer
│   └── RosbagRecorder.h/cpp  # Qt wrapper for bag recording
├── CMakeLists.txt
├── build.sh
└── README.md
```

## License

MIT License
