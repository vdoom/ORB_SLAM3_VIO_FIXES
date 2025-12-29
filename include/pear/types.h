/**
 * Pear VIO Camera API - Type Definitions
 *
 * Basic data types for IMU and camera data.
 */

#ifndef PEAR_TYPES_H
#define PEAR_TYPES_H

#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>

namespace pear {

/**
 * IMU measurement data from BMI160 sensor.
 *
 * Units:
 * - timestamp_ms: milliseconds (from Pico clock)
 * - accel_x/y/z: g (gravitational units)
 * - gyro_x/y/z: degrees per second
 */
struct IMUData {
    uint64_t timestamp_ms = 0;
    uint32_t interrupt_count = 0;
    float accel_x = 0, accel_y = 0, accel_z = 0;  // in g
    float gyro_x = 0, gyro_y = 0, gyro_z = 0;     // in deg/s

    // Convert acceleration to m/s²
    void getAccelSI(float& ax, float& ay, float& az) const {
        constexpr float G_TO_MS2 = 9.80665f;
        ax = accel_x * G_TO_MS2;
        ay = accel_y * G_TO_MS2;
        az = accel_z * G_TO_MS2;
    }

    // Convert gyroscope to rad/s
    void getGyroSI(float& gx, float& gy, float& gz) const {
        constexpr float DEG_TO_RAD = 0.0174532925f;  // pi/180
        gx = gyro_x * DEG_TO_RAD;
        gy = gyro_y * DEG_TO_RAD;
        gz = gyro_z * DEG_TO_RAD;
    }
};

/**
 * Thread-safe queue for IMU data.
 */
class IMUQueue {
public:
    void push(const IMUData& data) {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(data);
    }

    bool pop(IMUData& data) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.empty()) return false;
        data = queue_.front();
        queue_.pop();
        return true;
    }

    std::vector<IMUData> popAll() {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<IMUData> result;
        result.reserve(queue_.size());
        while (!queue_.empty()) {
            result.push_back(queue_.front());
            queue_.pop();
        }
        return result;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        while (!queue_.empty()) {
            queue_.pop();
        }
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.empty();
    }

private:
    mutable std::mutex mtx_;
    std::queue<IMUData> queue_;
};

}  // namespace pear

#endif  // PEAR_TYPES_H
