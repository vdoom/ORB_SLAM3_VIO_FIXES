#ifndef SERIAL_IMU_READER_H
#define SERIAL_IMU_READER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QVector3D>
#include <QString>
#include <atomic>

/**
 * IMU data structure
 * Contains parsed accelerometer and gyroscope data with timestamp
 */
struct IMUData {
    uint64_t timestamp_ms = 0;      // Timestamp in milliseconds
    uint32_t interrupt_count = 0;   // Interrupt counter from Pico

    // Accelerometer data in g
    float accel_x = 0.0f;
    float accel_y = 0.0f;
    float accel_z = 0.0f;

    // Gyroscope data in degrees/second
    float gyro_x = 0.0f;
    float gyro_y = 0.0f;
    float gyro_z = 0.0f;

    QVector3D accelVector() const { return QVector3D(accel_x, accel_y, accel_z); }
    QVector3D gyroVector() const { return QVector3D(gyro_x, gyro_y, gyro_z); }
};

/**
 * SerialIMUReader - Reads IMU data from serial port in a background thread
 *
 * Parses data in format:
 * timestamp: <ms>; [INT #<count>] A: <ax>,<ay>,<az> g | G: <gx>,<gy>,<gz> °/s
 *
 * Example:
 * timestamp: 1234567890; [INT #123] A: 0.01,0.02,1.00 g | G: 1.5,0.5,-0.3 °/s
 */
class SerialIMUReader : public QObject
{
    Q_OBJECT

public:
    explicit SerialIMUReader(QObject *parent = nullptr);
    ~SerialIMUReader() override;

    // Open serial port (e.g., "/dev/ttyACM0")
    bool open(const QString &portName, int baudRate = 115200);

    // Close serial port and stop reading
    void close();

    // Check if port is open and reading
    bool isOpen() const { return isOpen_; }

    // Get last received IMU data (thread-safe)
    IMUData lastData() const;

    // Get port name
    QString portName() const { return portName_; }

signals:
    // Emitted when new IMU data is received
    void dataReceived(const IMUData &data);

    // Emitted when camera trigger timestamp is received
    // This arrives BEFORE the actual camera frame from libcamera
    void cameraTriggerReceived(uint64_t timestamp_ms);

    // Emitted on error
    void errorOccurred(const QString &error);

    // Emitted when connection status changes
    void connectionChanged(bool connected);

private slots:
    void readLoop();

private:
    bool parseIMULine(const QString &line, IMUData &data);
    bool parseCameraTriggerLine(const QString &line, uint64_t &timestamp_ms);
    int configureSerialPort(int fd, int baudRate);

    QString portName_;
    int serialFd_ = -1;
    std::atomic<bool> isOpen_{false};
    std::atomic<bool> shouldStop_{false};

    QThread *readThread_ = nullptr;

    mutable QMutex dataMutex_;
    IMUData lastData_;
};

#endif // SERIAL_IMU_READER_H
