/**
 * Pear VIO Camera API - Serial IMU Reader
 *
 * Serial port reader for BMI160 IMU data from Raspberry Pi Pico.
 */

#ifndef PEAR_IMU_H
#define PEAR_IMU_H

#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>

#include "types.h"
#include "config.h"

namespace pear {

/**
 * Serial IMU reader for BMI160 sensor connected via Raspberry Pi Pico.
 *
 * Features:
 * - Parses IMU data from serial port (200Hz accel + gyro)
 * - Captures camera trigger timestamps from Pico
 * - Thread-safe data retrieval
 * - Supports hardware-synchronized camera triggering
 *
 * Expected serial format from Pico:
 * - IMU: "timestamp: 12345; [INT #1] A: 0.00,0.00,1.00 g | G: 0.00,0.00,0.00"
 * - Trigger: "Camera triggered timestamp: 12345"
 */
class SerialIMUReader {
public:
    SerialIMUReader();
    ~SerialIMUReader();

    // Non-copyable
    SerialIMUReader(const SerialIMUReader&) = delete;
    SerialIMUReader& operator=(const SerialIMUReader&) = delete;

    /**
     * Open serial port with given configuration.
     * @param config IMU configuration
     * @return true if successful
     */
    bool open(const IMUConfig& config);

    /**
     * Open serial port with default settings.
     * @param portName Serial port path (e.g., "/dev/ttyACM0")
     * @param baudRate Baud rate (default 115200)
     * @return true if successful
     */
    bool open(const std::string& portName, int baudRate = 115200);

    /**
     * Close serial port and stop reading thread.
     */
    void close();

    /**
     * Check if serial port is open.
     */
    bool isOpen() const { return isOpen_.load(); }

    /**
     * Get all accumulated IMU data and clear the buffer.
     * @return Vector of IMU measurements
     */
    std::vector<IMUData> getIMUData();

    /**
     * Get next camera trigger timestamp from queue.
     * @return Timestamp in milliseconds, or 0 if queue is empty
     */
    uint64_t getCameraTriggerTimestamp();

    /**
     * Check if there are pending camera trigger timestamps.
     */
    bool hasTriggerTimestamp();

    /**
     * Clear all pending camera trigger timestamps.
     */
    void clearTriggerQueue();

    /**
     * Clear all accumulated IMU data.
     */
    void clearIMUQueue();

    /**
     * Get current configuration.
     */
    const IMUConfig& config() const { return config_; }

private:
    void readLoop();

    // Configuration
    IMUConfig config_;

    // Serial port
    int fd_ = -1;
    std::atomic<bool> isOpen_{false};
    std::atomic<bool> shouldStop_{false};

    // Read thread
    std::thread readThread_;

    // IMU data queue
    IMUQueue imuQueue_;

    // Camera trigger queue
    std::mutex triggerMtx_;
    std::queue<uint64_t> triggerQueue_;
};

}  // namespace pear

#endif  // PEAR_IMU_H
