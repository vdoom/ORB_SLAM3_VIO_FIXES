#include "RosBagWriter.h"

#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace rosbag {

// MD5 sums for message types (pre-computed)
// These must match exactly what ROS expects
static const char* MD5_STD_MSGS_HEADER = "2176decaecbce78abc3b96ef049fabed";
static const char* MD5_SENSOR_MSGS_IMAGE = "060021388200f6f0f447d0fcd9c64743";
static const char* MD5_SENSOR_MSGS_IMU = "6a62c6daae103f4ff57a132d6f95cec2";

// Message definitions (simplified, but sufficient for Kalibr)
static const char* DEF_SENSOR_MSGS_IMAGE = R"(# This message contains an uncompressed image
# (0, 0) is at top-left corner of image

Header header

uint32 height
uint32 width

string encoding

uint8 is_bigendian
uint32 step
uint8[] data

================================================================================
MSG: std_msgs/Header
uint32 seq
time stamp
string frame_id
)";

static const char* DEF_SENSOR_MSGS_IMU = R"(# This is a message to hold data from an IMU (Inertial Measurement Unit)

Header header

geometry_msgs/Quaternion orientation
float64[9] orientation_covariance

geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance

geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance

================================================================================
MSG: std_msgs/Header
uint32 seq
time stamp
string frame_id
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
)";

RosBagWriter::RosBagWriter() = default;

RosBagWriter::~RosBagWriter() {
    if (isOpen()) {
        close();
    }
}

bool RosBagWriter::open(const std::string& filename) {
    if (isOpen()) {
        close();
    }

    filename_ = filename;
    // Need in|out for final header rewrite in close()
    file_.open(filename, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);

    if (!file_.is_open()) {
        return false;
    }

    // Reset state
    connections_.clear();
    nextConnId_ = 0;
    inChunk_ = false;
    chunkBuffer_.clear();
    chunkIndices_.clear();
    chunkInfos_.clear();
    connectionIndices_.clear();
    messageCount_ = 0;
    sequenceNumbers_.clear();

    // Write bag header
    writeBagHeader();

    return true;
}

void RosBagWriter::close() {
    if (!isOpen()) {
        return;
    }

    // End current chunk if any
    if (inChunk_) {
        endChunk();
    }

    // Remember position where index section starts (connection records + chunk info)
    // This is what index_pos in the bag header should point to
    uint64_t indexPos = static_cast<uint64_t>(file_.tellp());

    // Write all connection records
    for (const auto& [topic, conn] : connections_) {
        writeConnectionRecord(conn);
    }

    // Write chunk info records
    for (const auto& info : chunkInfos_) {
        // Build header fields
        std::map<std::string, std::vector<uint8_t>> fields;

        std::vector<uint8_t> opBuf;
        writeUint8(opBuf, OP_CHUNK_INFO);
        fields["op"] = opBuf;

        std::vector<uint8_t> verBuf;
        writeUint32(verBuf, 1);  // chunk info version
        fields["ver"] = verBuf;

        std::vector<uint8_t> posBuf;
        writeUint64(posBuf, info.chunk_pos);
        fields["chunk_pos"] = posBuf;

        std::vector<uint8_t> startBuf;
        writeTime(startBuf, info.start_time);
        fields["start_time"] = startBuf;

        std::vector<uint8_t> endBuf;
        writeTime(endBuf, info.end_time);
        fields["end_time"] = endBuf;

        std::vector<uint8_t> countBuf;
        writeUint32(countBuf, static_cast<uint32_t>(info.connection_counts.size()));
        fields["count"] = countBuf;

        writeRecordHeader(fields);

        // Data: connection count pairs
        std::vector<uint8_t> data;
        for (const auto& [connId, count] : info.connection_counts) {
            writeUint32(data, connId);
            writeUint32(data, count);
        }
        writeRecordData(data);
    }

    // Update bag header with connection and chunk counts and index position
    // Seek to start and rewrite bag header
    file_.seekp(0, std::ios::beg);
    writeBagHeader(indexPos);

    file_.close();
}

void RosBagWriter::writeBagHeader(uint64_t indexPos) {
    // Write magic
    const char* magic = "#ROSBAG V2.0\n";
    file_.write(magic, 13);

    // Build header fields
    std::map<std::string, std::vector<uint8_t>> fields;

    std::vector<uint8_t> opBuf;
    writeUint8(opBuf, OP_BAG_HEADER);
    fields["op"] = opBuf;

    std::vector<uint8_t> indexPosBuf;
    writeUint64(indexPosBuf, indexPos);  // 0 = unindexed, >0 = actual index position
    fields["index_pos"] = indexPosBuf;

    std::vector<uint8_t> connCountBuf;
    writeUint32(connCountBuf, static_cast<uint32_t>(connections_.size()));
    fields["conn_count"] = connCountBuf;

    std::vector<uint8_t> chunkCountBuf;
    writeUint32(chunkCountBuf, static_cast<uint32_t>(chunkInfos_.size()));
    fields["chunk_count"] = chunkCountBuf;

    writeRecordHeader(fields);

    // Bag header has no data, but we need padding to allow re-writing
    // Use 4096 bytes of padding
    std::vector<uint8_t> padding(4096, 0x20);  // space characters
    writeRecordData(padding);
}

int RosBagWriter::addImageTopic(const std::string& topic, const std::string& frame_id) {
    if (connections_.find(topic) != connections_.end()) {
        return connections_[topic].id;
    }

    ConnectionInfo conn;
    conn.id = nextConnId_++;
    conn.topic = topic;
    conn.datatype = "sensor_msgs/Image";
    conn.md5sum = MD5_SENSOR_MSGS_IMAGE;
    conn.message_definition = DEF_SENSOR_MSGS_IMAGE;
    conn.callerid = "/rosbag_writer";
    conn.latching = false;

    connections_[topic] = conn;
    sequenceNumbers_[topic] = 0;

    return conn.id;
}

int RosBagWriter::addImuTopic(const std::string& topic, const std::string& frame_id) {
    if (connections_.find(topic) != connections_.end()) {
        return connections_[topic].id;
    }

    ConnectionInfo conn;
    conn.id = nextConnId_++;
    conn.topic = topic;
    conn.datatype = "sensor_msgs/Imu";
    conn.md5sum = MD5_SENSOR_MSGS_IMU;
    conn.message_definition = DEF_SENSOR_MSGS_IMU;
    conn.callerid = "/rosbag_writer";
    conn.latching = false;

    connections_[topic] = conn;
    sequenceNumbers_[topic] = 0;

    return conn.id;
}

void RosBagWriter::writeConnectionRecord(const ConnectionInfo& conn) {
    // Build header fields
    std::map<std::string, std::vector<uint8_t>> fields;

    std::vector<uint8_t> opBuf;
    writeUint8(opBuf, OP_CONNECTION);
    fields["op"] = opBuf;

    std::vector<uint8_t> connBuf;
    writeUint32(connBuf, conn.id);
    fields["conn"] = connBuf;

    std::vector<uint8_t> topicBuf(conn.topic.begin(), conn.topic.end());
    fields["topic"] = topicBuf;

    writeRecordHeader(fields);

    // Data: connection header (another set of fields)
    std::vector<uint8_t> data;

    // Type field
    {
        std::string field = "type=" + conn.datatype;
        writeUint32(data, static_cast<uint32_t>(field.size()));
        data.insert(data.end(), field.begin(), field.end());
    }

    // MD5 field
    {
        std::string field = "md5sum=" + conn.md5sum;
        writeUint32(data, static_cast<uint32_t>(field.size()));
        data.insert(data.end(), field.begin(), field.end());
    }

    // Message definition field
    {
        std::string field = "message_definition=" + conn.message_definition;
        writeUint32(data, static_cast<uint32_t>(field.size()));
        data.insert(data.end(), field.begin(), field.end());
    }

    // Caller ID field
    {
        std::string field = "callerid=" + conn.callerid;
        writeUint32(data, static_cast<uint32_t>(field.size()));
        data.insert(data.end(), field.begin(), field.end());
    }

    // Latching field
    {
        std::string field = "latching=" + std::string(conn.latching ? "1" : "0");
        writeUint32(data, static_cast<uint32_t>(field.size()));
        data.insert(data.end(), field.begin(), field.end());
    }

    writeRecordData(data);
}

void RosBagWriter::startChunk() {
    if (inChunk_) {
        return;
    }

    // Just initialize state - don't write anything yet
    // We'll write everything when the chunk is complete
    chunkBuffer_.clear();
    chunkIndices_.clear();
    chunkStartTime_ = Time();
    chunkEndTime_ = Time();
    inChunk_ = true;
}

void RosBagWriter::endChunk() {
    if (!inChunk_) {
        return;
    }

    if (chunkBuffer_.empty()) {
        inChunk_ = false;
        return;
    }

    // Remember where this chunk starts in the file
    chunkStartPos_ = file_.tellp();

    // Build chunk header with known size
    std::map<std::string, std::vector<uint8_t>> fields;

    std::vector<uint8_t> opBuf;
    writeUint8(opBuf, OP_CHUNK);
    fields["op"] = opBuf;

    std::vector<uint8_t> compressionBuf;
    std::string compression = "none";
    compressionBuf.insert(compressionBuf.end(), compression.begin(), compression.end());
    fields["compression"] = compressionBuf;

    std::vector<uint8_t> sizeBuf;
    // Size field in chunk header refers to uncompressed size
    writeUint32(sizeBuf, static_cast<uint32_t>(chunkBuffer_.size()));
    fields["size"] = sizeBuf;

    // Write chunk header
    writeRecordHeader(fields);
    if (!file_.good()) {
        std::cerr << "RosBagWriter: Failed to write chunk header" << std::endl;
        return;
    }

    // Write chunk data length and data as record data
    uint32_t dataLen = static_cast<uint32_t>(chunkBuffer_.size());
    file_.write(reinterpret_cast<const char*>(&dataLen), sizeof(dataLen));

    // Write chunk buffer in smaller pieces to avoid blocking on large writes
    const size_t WRITE_CHUNK_SIZE = 64 * 1024;  // 64KB at a time
    const uint8_t* bufPtr = chunkBuffer_.data();
    size_t remaining = chunkBuffer_.size();

    while (remaining > 0) {
        size_t toWrite = std::min(remaining, WRITE_CHUNK_SIZE);
        file_.write(reinterpret_cast<const char*>(bufPtr), toWrite);
        if (!file_.good()) {
            std::cerr << "RosBagWriter: Failed to write chunk data (remaining: " << remaining << ")" << std::endl;
            return;
        }
        bufPtr += toWrite;
        remaining -= toWrite;
    }

    // Flush to ensure data is written
    file_.flush();
    if (!file_.good()) {
        std::cerr << "RosBagWriter: Failed to flush after chunk data" << std::endl;
        return;
    }

    // Write index records for this chunk
    for (const auto& [connId, indices] : chunkIndices_) {
        std::map<std::string, std::vector<uint8_t>> indexFields;

        std::vector<uint8_t> indexOpBuf;
        writeUint8(indexOpBuf, OP_INDEX_DATA);
        indexFields["op"] = indexOpBuf;

        std::vector<uint8_t> connBuf;
        writeUint32(connBuf, connId);
        indexFields["conn"] = connBuf;

        std::vector<uint8_t> verBuf;
        writeUint32(verBuf, 1);  // index version
        indexFields["ver"] = verBuf;

        std::vector<uint8_t> countBuf;
        writeUint32(countBuf, static_cast<uint32_t>(indices.size()));
        indexFields["count"] = countBuf;

        writeRecordHeader(indexFields);

        // Data: index entries
        std::vector<uint8_t> data;
        for (const auto& entry : indices) {
            writeTime(data, entry.time);
            writeUint32(data, entry.offset);
        }
        writeRecordData(data);
    }

    // Save chunk info
    ChunkInfo info;
    info.chunk_pos = chunkStartPos_;
    info.start_time = chunkStartTime_;
    info.end_time = chunkEndTime_;
    for (const auto& [connId, indices] : chunkIndices_) {
        info.connection_counts[connId] = static_cast<uint32_t>(indices.size());
    }
    chunkInfos_.push_back(info);

    // Merge chunk indices into connection indices
    for (const auto& [connId, indices] : chunkIndices_) {
        auto& connIndex = connectionIndices_[connId];
        for (const auto& entry : indices) {
            IndexEntry globalEntry = entry;
            globalEntry.chunk_pos = chunkStartPos_;
            connIndex.push_back(globalEntry);
        }
    }

    inChunk_ = false;
    chunkBuffer_.clear();
    chunkIndices_.clear();
}

void RosBagWriter::writeMessageToChunk(uint32_t connId, const Time& stamp,
                                        const std::vector<uint8_t>& serializedMsg) {
    if (!inChunk_) {
        startChunk();
    }

    // Record offset for index
    uint32_t offset = static_cast<uint32_t>(chunkBuffer_.size());

    // Update time bounds
    if (chunkStartTime_.sec == 0 && chunkStartTime_.nsec == 0) {
        chunkStartTime_ = stamp;
    }
    chunkEndTime_ = stamp;

    // Build message header
    std::map<std::string, std::vector<uint8_t>> fields;

    std::vector<uint8_t> opBuf;
    writeUint8(opBuf, OP_MESSAGE_DATA);
    fields["op"] = opBuf;

    std::vector<uint8_t> connBuf;
    writeUint32(connBuf, connId);
    fields["conn"] = connBuf;

    std::vector<uint8_t> timeBuf;
    writeTime(timeBuf, stamp);
    fields["time"] = timeBuf;

    // Calculate header size
    uint32_t headerSize = 0;
    for (const auto& [name, value] : fields) {
        headerSize += 4;  // field length
        headerSize += static_cast<uint32_t>(name.size());
        headerSize += 1;  // '='
        headerSize += static_cast<uint32_t>(value.size());
    }

    // Write header length
    writeUint32(chunkBuffer_, headerSize);

    // Write header fields
    for (const auto& [name, value] : fields) {
        uint32_t fieldLen = static_cast<uint32_t>(name.size() + 1 + value.size());
        writeUint32(chunkBuffer_, fieldLen);
        chunkBuffer_.insert(chunkBuffer_.end(), name.begin(), name.end());
        chunkBuffer_.push_back('=');
        chunkBuffer_.insert(chunkBuffer_.end(), value.begin(), value.end());
    }

    // Write data length and data
    writeUint32(chunkBuffer_, static_cast<uint32_t>(serializedMsg.size()));
    chunkBuffer_.insert(chunkBuffer_.end(), serializedMsg.begin(), serializedMsg.end());

    // Add index entry
    IndexEntry entry;
    entry.time = stamp;
    entry.offset = offset;
    chunkIndices_[connId].push_back(entry);

    messageCount_++;

    // Check if chunk should be closed
    if (chunkBuffer_.size() >= CHUNK_SIZE_THRESHOLD) {
        endChunk();
    }
}

bool RosBagWriter::writeImage(const std::string& topic, const Time& stamp,
                               uint32_t width, uint32_t height,
                               const uint8_t* data, size_t dataSize,
                               const std::string& encoding) {
    auto it = connections_.find(topic);
    if (it == connections_.end()) {
        return false;
    }

    ImageMessage msg;
    msg.header.seq = sequenceNumbers_[topic]++;
    msg.header.stamp = stamp;
    msg.header.frame_id = "cam0";
    msg.height = height;
    msg.width = width;
    msg.encoding = encoding;
    msg.is_bigendian = 0;
    msg.step = width;  // For mono8
    msg.data.assign(data, data + dataSize);

    std::vector<uint8_t> serialized = serializeImage(msg);
    writeMessageToChunk(it->second.id, stamp, serialized);

    return true;
}

bool RosBagWriter::writeImu(const std::string& topic, const Time& stamp,
                             const double accel[3], const double gyro[3]) {
    auto it = connections_.find(topic);
    if (it == connections_.end()) {
        return false;
    }

    ImuMessage msg;
    msg.header.seq = sequenceNumbers_[topic]++;
    msg.header.stamp = stamp;
    msg.header.frame_id = "imu0";

    // Orientation unknown
    msg.orientation[0] = 0;  // x
    msg.orientation[1] = 0;  // y
    msg.orientation[2] = 0;  // z
    msg.orientation[3] = 1;  // w
    msg.orientation_covariance[0] = -1;  // Unknown

    // Angular velocity
    msg.angular_velocity[0] = gyro[0];
    msg.angular_velocity[1] = gyro[1];
    msg.angular_velocity[2] = gyro[2];

    // Linear acceleration
    msg.linear_acceleration[0] = accel[0];
    msg.linear_acceleration[1] = accel[1];
    msg.linear_acceleration[2] = accel[2];

    std::vector<uint8_t> serialized = serializeImu(msg);
    writeMessageToChunk(it->second.id, stamp, serialized);

    return true;
}

std::vector<uint8_t> RosBagWriter::serializeImage(const ImageMessage& msg) {
    std::vector<uint8_t> buf;

    // Header
    writeHeader(buf, msg.header);

    // Image fields
    writeUint32(buf, msg.height);
    writeUint32(buf, msg.width);
    writeString(buf, msg.encoding);
    writeUint8(buf, msg.is_bigendian);
    writeUint32(buf, msg.step);

    // Data array (with length prefix)
    writeUint32(buf, static_cast<uint32_t>(msg.data.size()));
    buf.insert(buf.end(), msg.data.begin(), msg.data.end());

    return buf;
}

std::vector<uint8_t> RosBagWriter::serializeImu(const ImuMessage& msg) {
    std::vector<uint8_t> buf;

    // Header
    writeHeader(buf, msg.header);

    // Orientation (Quaternion: x, y, z, w)
    writeFloat64(buf, msg.orientation[0]);
    writeFloat64(buf, msg.orientation[1]);
    writeFloat64(buf, msg.orientation[2]);
    writeFloat64(buf, msg.orientation[3]);

    // Orientation covariance (9 doubles)
    for (int i = 0; i < 9; i++) {
        writeFloat64(buf, msg.orientation_covariance[i]);
    }

    // Angular velocity (Vector3: x, y, z)
    writeFloat64(buf, msg.angular_velocity[0]);
    writeFloat64(buf, msg.angular_velocity[1]);
    writeFloat64(buf, msg.angular_velocity[2]);

    // Angular velocity covariance (9 doubles)
    for (int i = 0; i < 9; i++) {
        writeFloat64(buf, msg.angular_velocity_covariance[i]);
    }

    // Linear acceleration (Vector3: x, y, z)
    writeFloat64(buf, msg.linear_acceleration[0]);
    writeFloat64(buf, msg.linear_acceleration[1]);
    writeFloat64(buf, msg.linear_acceleration[2]);

    // Linear acceleration covariance (9 doubles)
    for (int i = 0; i < 9; i++) {
        writeFloat64(buf, msg.linear_acceleration_covariance[i]);
    }

    return buf;
}

void RosBagWriter::writeUint8(std::vector<uint8_t>& buf, uint8_t val) {
    buf.push_back(val);
}

void RosBagWriter::writeUint32(std::vector<uint8_t>& buf, uint32_t val) {
    buf.push_back(static_cast<uint8_t>(val & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
}

void RosBagWriter::writeUint64(std::vector<uint8_t>& buf, uint64_t val) {
    for (int i = 0; i < 8; i++) {
        buf.push_back(static_cast<uint8_t>((val >> (i * 8)) & 0xFF));
    }
}

void RosBagWriter::writeFloat64(std::vector<uint8_t>& buf, double val) {
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    writeUint64(buf, bits);
}

void RosBagWriter::writeString(std::vector<uint8_t>& buf, const std::string& str) {
    writeUint32(buf, static_cast<uint32_t>(str.size()));
    buf.insert(buf.end(), str.begin(), str.end());
}

void RosBagWriter::writeTime(std::vector<uint8_t>& buf, const Time& t) {
    writeUint32(buf, t.sec);
    writeUint32(buf, t.nsec);
}

void RosBagWriter::writeHeader(std::vector<uint8_t>& buf, const Header& h) {
    writeUint32(buf, h.seq);
    writeTime(buf, h.stamp);
    writeString(buf, h.frame_id);
}

void RosBagWriter::writeRecordHeader(const std::map<std::string, std::vector<uint8_t>>& fields) {
    // Calculate header size
    uint32_t headerSize = 0;
    for (const auto& [name, value] : fields) {
        headerSize += 4;  // field length
        headerSize += static_cast<uint32_t>(name.size());
        headerSize += 1;  // '='
        headerSize += static_cast<uint32_t>(value.size());
    }

    // Write header length
    file_.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));

    // Write header fields
    for (const auto& [name, value] : fields) {
        uint32_t fieldLen = static_cast<uint32_t>(name.size() + 1 + value.size());
        file_.write(reinterpret_cast<const char*>(&fieldLen), sizeof(fieldLen));
        file_.write(name.c_str(), name.size());
        file_.put('=');
        file_.write(reinterpret_cast<const char*>(value.data()), value.size());
    }
}

void RosBagWriter::writeRecordData(const std::vector<uint8_t>& data) {
    uint32_t dataLen = static_cast<uint32_t>(data.size());
    file_.write(reinterpret_cast<const char*>(&dataLen), sizeof(dataLen));
    if (!data.empty()) {
        file_.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
}

uint64_t RosBagWriter::fileSize() const {
    if (!file_.is_open()) {
        return 0;
    }
    auto pos = const_cast<std::fstream&>(file_).tellp();
    return static_cast<uint64_t>(pos);
}

} // namespace rosbag
