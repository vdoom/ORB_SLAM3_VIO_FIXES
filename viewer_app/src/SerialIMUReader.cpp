#include "SerialIMUReader.h"

#include <QDebug>
#include <QRegularExpression>

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <cstring>
#include <errno.h>

SerialIMUReader::SerialIMUReader(QObject *parent)
    : QObject(parent)
{
}

SerialIMUReader::~SerialIMUReader()
{
    close();
}

int SerialIMUReader::configureSerialPort(int fd, int baudRate)
{
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        return -1;
    }

    // Set baud rate
    speed_t speed;
    switch (baudRate) {
        case 9600:   speed = B9600; break;
        case 19200:  speed = B19200; break;
        case 38400:  speed = B38400; break;
        case 57600:  speed = B57600; break;
        case 115200: speed = B115200; break;
        case 230400: speed = B230400; break;
        case 460800: speed = B460800; break;
        default:     speed = B115200; break;
    }

    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    // 8N1 mode (8 data bits, no parity, 1 stop bit)
    tty.c_cflag &= ~PARENB;        // No parity
    tty.c_cflag &= ~CSTOPB;        // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;            // 8 data bits
    tty.c_cflag &= ~CRTSCTS;       // No hardware flow control
    tty.c_cflag |= CREAD | CLOCAL; // Enable reading, ignore modem controls

    // Non-canonical mode (raw input)
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;

    // Disable software flow control
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;

    // Set read timeout (0.1 seconds)
    tty.c_cc[VTIME] = 1;
    tty.c_cc[VMIN] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        return -1;
    }

    return 0;
}

bool SerialIMUReader::open(const QString &portName, int baudRate)
{
    if (isOpen_) {
        close();
    }

    portName_ = portName;

    // Open serial port
    serialFd_ = ::open(portName.toLocal8Bit().constData(), O_RDWR | O_NOCTTY);
    if (serialFd_ < 0) {
        QString error = QString("Failed to open %1: %2").arg(portName).arg(strerror(errno));
        emit errorOccurred(error);
        return false;
    }

    // Configure serial port
    if (configureSerialPort(serialFd_, baudRate) != 0) {
        QString error = QString("Failed to configure %1: %2").arg(portName).arg(strerror(errno));
        ::close(serialFd_);
        serialFd_ = -1;
        emit errorOccurred(error);
        return false;
    }

    // Flush any pending data
    tcflush(serialFd_, TCIOFLUSH);

    isOpen_ = true;
    shouldStop_ = false;

    // Start read thread
    readThread_ = QThread::create([this]() { readLoop(); });
    readThread_->start();

    qDebug() << "Serial port opened:" << portName << "at" << baudRate << "baud";
    emit connectionChanged(true);

    return true;
}

void SerialIMUReader::close()
{
    if (!isOpen_) {
        return;
    }

    shouldStop_ = true;

    if (readThread_) {
        readThread_->quit();
        readThread_->wait(1000);
        delete readThread_;
        readThread_ = nullptr;
    }

    if (serialFd_ >= 0) {
        ::close(serialFd_);
        serialFd_ = -1;
    }

    isOpen_ = false;
    emit connectionChanged(false);

    qDebug() << "Serial port closed:" << portName_;
}

IMUData SerialIMUReader::lastData() const
{
    QMutexLocker locker(&dataMutex_);
    return lastData_;
}

void SerialIMUReader::readLoop()
{
    QString lineBuffer;
    char buf[256];

    while (!shouldStop_ && serialFd_ >= 0) {
        int n = ::read(serialFd_, buf, sizeof(buf) - 1);

        if (n > 0) {
            buf[n] = '\0';

            // Append to line buffer and process complete lines
            lineBuffer += QString::fromLocal8Bit(buf, n);

            int newlineIdx;
            while ((newlineIdx = lineBuffer.indexOf('\n')) != -1) {
                QString line = lineBuffer.left(newlineIdx).trimmed();
                lineBuffer = lineBuffer.mid(newlineIdx + 1);

                if (!line.isEmpty()) {
                    // Try parsing as camera trigger first
                    uint64_t cameraTriggerTimestamp;
                    if (parseCameraTriggerLine(line, cameraTriggerTimestamp)) {
                        emit cameraTriggerReceived(cameraTriggerTimestamp);
                    } else {
                        // Try parsing as IMU data
                        IMUData data;
                        if (parseIMULine(line, data)) {
                            {
                                QMutexLocker locker(&dataMutex_);
                                lastData_ = data;
                            }
                            emit dataReceived(data);
                        }
                    }
                }
            }

            // Prevent buffer from growing too large
            if (lineBuffer.size() > 1024) {
                lineBuffer.clear();
            }
        } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            if (!shouldStop_) {
                emit errorOccurred(QString("Read error: %1").arg(strerror(errno)));
            }
            break;
        }

        // Small sleep to prevent busy waiting
        QThread::usleep(100);
    }
}

bool SerialIMUReader::parseIMULine(const QString &line, IMUData &data)
{
    // Parse format: timestamp: <ms>; [INT #<count>] A: <ax>,<ay>,<az> g | G: <gx>,<gy>,<gz> °/s
    // Example: timestamp: 1234567890; [INT #123] A: 0.01,0.02,1.00 g | G: 1.5,0.5,-0.3 °/s

    // Skip status lines and other non-data lines
    if (!line.startsWith("timestamp:")) {
        return false;
    }

    // Regular expression to parse the line
    static QRegularExpression regex(
        R"(timestamp:\s*(\d+);\s*\[INT\s*#(\d+)\]\s*A:\s*([-\d.]+),([-\d.]+),([-\d.]+)\s*g\s*\|\s*G:\s*([-\d.]+),([-\d.]+),([-\d.]+))"
    );

    QRegularExpressionMatch match = regex.match(line);
    if (!match.hasMatch()) {
        return false;
    }

    data.timestamp_ms = match.captured(1).toULongLong();
    data.interrupt_count = match.captured(2).toUInt();

    data.accel_x = match.captured(3).toFloat();
    data.accel_y = match.captured(4).toFloat();
    data.accel_z = match.captured(5).toFloat();

    data.gyro_x = match.captured(6).toFloat();
    data.gyro_y = match.captured(7).toFloat();
    data.gyro_z = match.captured(8).toFloat();

    return true;
}

bool SerialIMUReader::parseCameraTriggerLine(const QString &line, uint64_t &timestamp_ms)
{
    // Parse format: Camera triggered timestamp:  <ms>
    // Example: Camera triggered timestamp:  7128215

    if (!line.startsWith("Camera triggered timestamp:")) {
        return false;
    }

    // Regular expression to parse the timestamp
    static QRegularExpression regex(R"(Camera triggered timestamp:\s*(\d+))");

    QRegularExpressionMatch match = regex.match(line);
    if (!match.hasMatch()) {
        return false;
    }

    timestamp_ms = match.captured(1).toULongLong();
    return true;
}
