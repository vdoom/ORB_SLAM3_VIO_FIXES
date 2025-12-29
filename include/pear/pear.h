/**
 * Pear VIO Camera API
 *
 * A hardware-synchronized Visual-Inertial camera system combining:
 * - OV9281 global shutter camera (via libcamera)
 * - BMI160 IMU (via Raspberry Pi Pico over serial)
 * - Hardware trigger synchronization
 *
 * Main header - includes all Pear API components.
 *
 * Quick start:
 *   #include <pear/pear.h>
 *
 *   pear::PearVIOCamera camera;
 *   camera.open("/dev/ttyACM0");
 *   camera.start();
 *
 *   pear::VIOFrame frame;
 *   while (camera.getFrame(frame)) {
 *       // frame.image - grayscale cv::Mat
 *       // frame.timestamp - seconds since start
 *       // frame.imuData - vector of IMU measurements
 *   }
 */

#ifndef PEAR_H
#define PEAR_H

// Core types
#include "types.h"

// Configuration
#include "config.h"

// Low-level components (for advanced usage)
#include "camera.h"
#include "imu.h"

// High-level interface (recommended)
#include "pear_vio_camera.h"

#endif  // PEAR_H
