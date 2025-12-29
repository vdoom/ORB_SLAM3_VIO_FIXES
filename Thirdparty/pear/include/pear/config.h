/**
 * Pear VIO Camera API - Configuration
 *
 * Configuration structures for camera, IMU, and unified VIO camera.
 */

#ifndef PEAR_CONFIG_H
#define PEAR_CONFIG_H

#include <string>
#include <fstream>
#include <cstdlib>

namespace pear {

/**
 * Camera configuration for OV9281 sensor via libcamera.
 */
struct CameraConfig {
    // Camera device
    int cameraIndex = 0;

    // Resolution
    int width = 640;
    int height = 400;

    // Exposure settings
    bool autoExposure = true;
    int exposureTimeUs = 10000;   // microseconds (manual mode)
    float gain = 8.0f;            // analog gain (manual mode)

    // Frame rate control
    int64_t frameDurationUs = 50000;  // 50ms = 20 FPS (for triggered mode)

    // Trigger mode (external trigger from Pico)
    bool triggerMode = false;

    // I2C configuration for OV9281 trigger mode control
    int i2cBus = 10;              // I2C bus number (10 for RPi5 CAM1)
    int i2cAddress = 0x60;        // OV9281 I2C address

    /**
     * Load settings from INI file (compatible with viewer_app format).
     * @param path Path to INI file
     * @return true if loaded successfully
     */
    bool loadFromIniFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        std::string currentSection;

        while (std::getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            size_t end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;

            // Section header
            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.size() - 2);
                continue;
            }

            // Key=value pair
            size_t eqPos = line.find('=');
            if (eqPos == std::string::npos) continue;

            std::string key = line.substr(0, eqPos);
            std::string value = line.substr(eqPos + 1);

            // Trim key and value
            start = key.find_first_not_of(" \t");
            end = key.find_last_not_of(" \t");
            if (start != std::string::npos) key = key.substr(start, end - start + 1);

            start = value.find_first_not_of(" \t");
            end = value.find_last_not_of(" \t");
            if (start != std::string::npos) value = value.substr(start, end - start + 1);

            // Parse camera section
            if (currentSection == "camera") {
                if (key == "auto_exposure") {
                    autoExposure = (value == "true" || value == "1");
                } else if (key == "manual_exposure_us") {
                    exposureTimeUs = std::stoi(value);
                } else if (key == "manual_gain") {
                    gain = std::stof(value);
                } else if (key == "trigger_mode") {
                    triggerMode = (value == "true" || value == "1");
                } else if (key == "width") {
                    width = std::stoi(value);
                } else if (key == "height") {
                    height = std::stoi(value);
                } else if (key == "i2c_bus") {
                    i2cBus = std::stoi(value);
                } else if (key == "i2c_address") {
                    i2cAddress = std::stoi(value, nullptr, 0);  // supports hex
                }
            }
        }

        return true;
    }

    /**
     * Get default config path (~/.config/rpi_camera_viewer.ini).
     */
    static std::string getDefaultConfigPath() {
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + "/.config/rpi_camera_viewer.ini";
        }
        return "";
    }
};

/**
 * IMU configuration for BMI160 sensor via serial (Raspberry Pi Pico).
 */
struct IMUConfig {
    // Serial port
    std::string serialPort = "/dev/ttyACM0";
    int baudRate = 115200;

    // IMU rate (BMI160 is configured for 200Hz on both accel and gyro)
    int imuRateHz = 200;

    // Camera trigger configuration (Pico triggers camera every N IMU samples)
    int samplesPerTrigger = 10;  // 200Hz / 10 = 20Hz camera rate
};

/**
 * Unified VIO camera configuration.
 */
struct PearConfig {
    CameraConfig camera;
    IMUConfig imu;

    /**
     * Load configuration from INI file.
     * Camera settings from [camera] section, IMU from [imu] section.
     */
    bool loadFromIniFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        std::string currentSection;

        while (std::getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            size_t end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;

            // Section header
            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.size() - 2);
                continue;
            }

            // Key=value pair
            size_t eqPos = line.find('=');
            if (eqPos == std::string::npos) continue;

            std::string key = line.substr(0, eqPos);
            std::string value = line.substr(eqPos + 1);

            // Trim key and value
            start = key.find_first_not_of(" \t");
            end = key.find_last_not_of(" \t");
            if (start != std::string::npos) key = key.substr(start, end - start + 1);

            start = value.find_first_not_of(" \t");
            end = value.find_last_not_of(" \t");
            if (start != std::string::npos) value = value.substr(start, end - start + 1);

            // Parse sections
            if (currentSection == "camera") {
                if (key == "auto_exposure") {
                    camera.autoExposure = (value == "true" || value == "1");
                } else if (key == "manual_exposure_us") {
                    camera.exposureTimeUs = std::stoi(value);
                } else if (key == "manual_gain") {
                    camera.gain = std::stof(value);
                } else if (key == "trigger_mode") {
                    camera.triggerMode = (value == "true" || value == "1");
                } else if (key == "width") {
                    camera.width = std::stoi(value);
                } else if (key == "height") {
                    camera.height = std::stoi(value);
                } else if (key == "camera_index") {
                    camera.cameraIndex = std::stoi(value);
                } else if (key == "i2c_bus") {
                    camera.i2cBus = std::stoi(value);
                } else if (key == "i2c_address") {
                    camera.i2cAddress = std::stoi(value, nullptr, 0);
                }
            } else if (currentSection == "imu") {
                if (key == "serial_port") {
                    imu.serialPort = value;
                } else if (key == "baud_rate") {
                    imu.baudRate = std::stoi(value);
                } else if (key == "rate_hz") {
                    imu.imuRateHz = std::stoi(value);
                } else if (key == "samples_per_trigger") {
                    imu.samplesPerTrigger = std::stoi(value);
                }
            }
        }

        return true;
    }

    /**
     * Create default configuration.
     */
    static PearConfig createDefault() {
        return PearConfig();
    }
};

}  // namespace pear

#endif  // PEAR_CONFIG_H
