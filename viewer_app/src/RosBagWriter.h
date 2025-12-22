#ifndef ROS_BAG_WRITER_H
#define ROS_BAG_WRITER_H

/**
 * RosBagWriter - Standalone ROS1 bag file writer
 *
 * This is a custom implementation of the ROS1 bag format (version 2.0)
 * without any ROS dependencies. It can write sensor_msgs/Image and
 * sensor_msgs/Imu messages for use with Kalibr calibration tool.
 *
 * ROS1 bag format reference:
 * http://wiki.ros.org/Bags/Format/2.0
 *
 * Implementation notes:
 * - Uses uncompressed chunks for simplicity
 * - Writes proper connection records and indices
 * - Compatible with rosbag play and Kalibr
 */

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <cstdint>
#include <memory>

namespace rosbag {

// ROS Time structure
struct Time {
    uint32_t sec = 0;
    uint32_t nsec = 0;

    Time() = default;
    Time(uint32_t s, uint32_t ns) : sec(s), nsec(ns) {}

    // Create from nanoseconds since epoch
    static Time fromNsec(uint64_t nsec) {
        return Time(static_cast<uint32_t>(nsec / 1000000000ULL),
                    static_cast<uint32_t>(nsec % 1000000000ULL));
    }

    // Create from milliseconds since epoch
    static Time fromMsec(uint64_t msec) {
        return Time(static_cast<uint32_t>(msec / 1000),
                    static_cast<uint32_t>((msec % 1000) * 1000000));
    }

    uint64_t toNsec() const {
        return static_cast<uint64_t>(sec) * 1000000000ULL + nsec;
    }

    bool operator<(const Time& other) const {
        if (sec != other.sec) return sec < other.sec;
        return nsec < other.nsec;
    }
};

// Message header (std_msgs/Header)
struct Header {
    uint32_t seq = 0;
    Time stamp;
    std::string frame_id;
};

// IMU message data
struct ImuMessage {
    Header header;

    // Orientation quaternion (x, y, z, w)
    double orientation[4] = {0, 0, 0, 1};
    double orientation_covariance[9] = {-1, 0, 0, 0, 0, 0, 0, 0, 0}; // -1 means unknown

    // Angular velocity (rad/s)
    double angular_velocity[3] = {0, 0, 0};
    double angular_velocity_covariance[9] = {0};

    // Linear acceleration (m/s^2)
    double linear_acceleration[3] = {0, 0, 0};
    double linear_acceleration_covariance[9] = {0};
};

// Image message data
struct ImageMessage {
    Header header;
    uint32_t height = 0;
    uint32_t width = 0;
    std::string encoding;  // "mono8", "rgb8", "bgr8", etc.
    uint8_t is_bigendian = 0;
    uint32_t step = 0;     // Row stride in bytes
    std::vector<uint8_t> data;
};

// Connection info for a topic
struct ConnectionInfo {
    uint32_t id = 0;
    std::string topic;
    std::string datatype;
    std::string md5sum;
    std::string message_definition;
    std::string callerid;
    bool latching = false;
};

// Index entry for a message
struct IndexEntry {
    Time time;
    uint64_t chunk_pos = 0;
    uint32_t offset = 0;  // Offset within chunk
};

// Chunk info
struct ChunkInfo {
    uint64_t chunk_pos = 0;
    Time start_time;
    Time end_time;
    std::map<uint32_t, uint32_t> connection_counts;  // conn_id -> count
};

/**
 * RosBagWriter class
 *
 * Usage:
 *   RosBagWriter writer;
 *   writer.open("output.bag");
 *
 *   // Add topics
 *   writer.addImageTopic("/cam0/image_raw", "cam0");
 *   writer.addImuTopic("/imu0", "imu");
 *
 *   // Write messages
 *   writer.writeImage("/cam0/image_raw", timestamp, width, height, data);
 *   writer.writeImu("/imu0", timestamp, accel, gyro);
 *
 *   writer.close();
 */
class RosBagWriter {
public:
    RosBagWriter();
    ~RosBagWriter();

    // Open bag file for writing
    bool open(const std::string& filename);

    // Close bag file (writes indices and finalizes)
    void close();

    // Check if file is open
    bool isOpen() const { return file_.is_open(); }

    // Add a topic for image messages
    // Returns connection ID, or -1 on error
    int addImageTopic(const std::string& topic, const std::string& frame_id = "camera");

    // Add a topic for IMU messages
    // Returns connection ID, or -1 on error
    int addImuTopic(const std::string& topic, const std::string& frame_id = "imu");

    // Write an image message
    // data should be grayscale (mono8) with size width*height
    bool writeImage(const std::string& topic, const Time& stamp,
                    uint32_t width, uint32_t height,
                    const uint8_t* data, size_t dataSize,
                    const std::string& encoding = "mono8");

    // Write an IMU message
    // accel in m/s^2, gyro in rad/s
    bool writeImu(const std::string& topic, const Time& stamp,
                  const double accel[3], const double gyro[3]);

    // Get number of messages written
    uint64_t messageCount() const { return messageCount_; }

    // Get current file size
    uint64_t fileSize() const;

private:
    // Record opcodes
    static constexpr uint8_t OP_MESSAGE_DATA = 0x02;
    static constexpr uint8_t OP_BAG_HEADER = 0x03;
    static constexpr uint8_t OP_INDEX_DATA = 0x04;
    static constexpr uint8_t OP_CHUNK = 0x05;
    static constexpr uint8_t OP_CHUNK_INFO = 0x06;
    static constexpr uint8_t OP_CONNECTION = 0x07;

    // Write bag file header
    // indexPos = 0 means unindexed (used during initial write)
    // indexPos > 0 is the actual index position (used during final rewrite)
    void writeBagHeader(uint64_t indexPos = 0);

    // Write a connection record
    void writeConnectionRecord(const ConnectionInfo& conn);

    // Start a new chunk
    void startChunk();

    // End current chunk and write index
    void endChunk();

    // Write a message to current chunk
    void writeMessageToChunk(uint32_t connId, const Time& stamp,
                             const std::vector<uint8_t>& serializedMsg);

    // Serialize ROS messages
    std::vector<uint8_t> serializeImage(const ImageMessage& msg);
    std::vector<uint8_t> serializeImu(const ImuMessage& msg);

    // Write primitive types
    void writeUint8(std::vector<uint8_t>& buf, uint8_t val);
    void writeUint32(std::vector<uint8_t>& buf, uint32_t val);
    void writeUint64(std::vector<uint8_t>& buf, uint64_t val);
    void writeFloat64(std::vector<uint8_t>& buf, double val);
    void writeString(std::vector<uint8_t>& buf, const std::string& str);
    void writeTime(std::vector<uint8_t>& buf, const Time& t);
    void writeHeader(std::vector<uint8_t>& buf, const Header& h);

    // Write record header (field=value pairs)
    void writeRecordHeader(const std::map<std::string, std::vector<uint8_t>>& fields);
    void writeRecordData(const std::vector<uint8_t>& data);

    // File stream
    std::fstream file_;
    std::string filename_;

    // Connections (topics)
    std::map<std::string, ConnectionInfo> connections_;
    uint32_t nextConnId_ = 0;

    // Current chunk state
    bool inChunk_ = false;
    uint64_t chunkStartPos_ = 0;
    std::vector<uint8_t> chunkBuffer_;
    std::map<uint32_t, std::vector<IndexEntry>> chunkIndices_;  // conn_id -> indices
    Time chunkStartTime_;
    Time chunkEndTime_;

    // All chunk infos for final index
    std::vector<ChunkInfo> chunkInfos_;

    // Overall indices
    std::map<uint32_t, std::vector<IndexEntry>> connectionIndices_;

    // Statistics
    uint64_t messageCount_ = 0;

    // Sequence numbers per topic
    std::map<std::string, uint32_t> sequenceNumbers_;

    // Constants
    static constexpr size_t CHUNK_SIZE_THRESHOLD = 4 * 1024 * 1024;  // 4MB chunks
};

} // namespace rosbag

#endif // ROS_BAG_WRITER_H
