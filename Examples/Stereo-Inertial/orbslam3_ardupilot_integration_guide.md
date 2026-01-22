# ORB-SLAM3 to ArduPilot VIO Integration Guide

## Executive Summary

This document provides a complete implementation specification for integrating ORB-SLAM3 stereo-inertial VIO with ArduPilot via MAVLink VISION_POSITION_ESTIMATE messages. The key challenge is handling tracking loss and new map creation without causing position jumps or triggering EKF failsafes.

**Critical Finding:** Sending zero positions with an incremented `reset_counter` causes ArduPilot to snap the drone's position to the origin (home point). The solution is to apply coordinate frame transformations to maintain position continuity.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Understanding ArduPilot's Reset Counter Mechanism](#understanding-ardupilots-reset-counter-mechanism)
3. [ORB-SLAM3 Multi-Map System Behavior](#orb-slam3-multi-map-system-behavior)
4. [Solution Architecture](#solution-architecture)
5. [Implementation Specifications](#implementation-specifications)
6. [ArduPilot Configuration](#ardupilot-configuration)
7. [Testing and Validation](#testing-and-validation)
8. [Troubleshooting](#troubleshooting)

---

## Problem Statement

### Current Behavior

When ORB-SLAM3 loses tracking and creates a new map:
- The new map starts at position (0,0,0) in its local coordinate frame
- Sending these zero coordinates with an incremented `reset_counter` to ArduPilot
- ArduPilot's EKF3 interprets this as "the drone has moved to position (0,0,0) in NED frame"
- Result: Position estimate jumps to the home point, causing loss of control

### Root Cause

The `reset_counter` field in VISION_POSITION_ESTIMATE is designed for **valid relocalization events**, not tracking loss. When incremented, it triggers an immediate position state reset in EKF3:

```cpp
// From ArduPilot source: AP_NavEKF3_PosVelFusion.cpp
if (extNavDataToFuse && extNavDataDelayed.posReset) {
    stateStruct.position.x = extNavDataDelayed.pos.x;  // Direct assignment!
    stateStruct.position.y = extNavDataDelayed.pos.y;  // No filtering!
}
```

---

## Understanding ArduPilot's Reset Counter Mechanism

### What Reset Counter Actually Means

The `reset_counter` field signals to ArduPilot: **"The VIO system has relocated to a new reference frame, and here is the vehicle's valid position in that new frame."**

### When to Increment Reset Counter

✅ **DO increment** when:
- ORB-SLAM3 successfully relocalizes to a previously mapped area (existing map in Atlas)
- Loop closure occurs with a valid corrected position
- The new position represents a real measurement in a known coordinate system

❌ **DON'T increment** when:
- Tracking is lost and you have no valid position
- A new map is created at an arbitrary origin (0,0,0)
- Temporarily losing tracking within the same map

### EKF3 Behavior Summary

| Message Type | EKF3 Response |
|--------------|---------------|
| Normal VISION_POSITION_ESTIMATE | Fuses position using Kalman filter innovation |
| Changed reset_counter + position | **Immediately snaps state** to reported position |
| No messages for >1 second | Continues with IMU dead-reckoning, triggers timeout warning |
| No messages for >5 seconds (default) | Triggers EKF failsafe (FS_EKF_ACTION) |

---

## ORB-SLAM3 Multi-Map System Behavior

### Atlas Multi-Map Architecture

ORB-SLAM3's Atlas system maintains multiple independent maps:
- Each map has its own coordinate system with arbitrary origin
- Maps are initially unrelated (no spatial connection)
- Place recognition can later merge maps via Sim(3) transformation

### Tracking States and Map Creation

```
TRACKING_OK          → Normal operation, position valid
↓ (features lost)
RECENTLY_LOST        → Short-term loss, IMU prediction active
↓ (>5 seconds, controlled by time_recently_lost parameter)
LOST                 → Extended loss, attempting relocalization
↓ (relocalization fails)
NEW_MAP_CREATED      → Fresh map spawned at origin (0,0,0)
↓ (place recognition succeeds)
RELOCALIZED          → Jumped to existing map, position may be valid
```

### Coordinate Frame Behavior

**Visual-Inertial Mode (Stereo-Inertial):**
- Initial keyframe placed at origin with identity rotation
- After ~2 seconds of IMU data, map is rotated to align Z-axis with gravity
- Position remains at first keyframe location (arbitrary in world space)

**Key Insight:** When a new map spawns, the position (0,0,0) is meaningless in the drone's actual flight coordinate system. The drone hasn't moved - only the VIO's internal reference frame has changed.

---

## Solution Architecture

### Approach: Coordinate Frame Bridging

Instead of passing coordinate frame discontinuities to ArduPilot, handle them in the integration layer by maintaining a persistent offset between VIO coordinates and NED coordinates.

### Core Principle

```
NED_position = VIO_position + vio_to_ned_offset

Where:
- NED_position: What we send to ArduPilot
- VIO_position: What ORB-SLAM3 reports
- vio_to_ned_offset: Calculated at each map transition
```

### Architecture Diagram

```
┌─────────────┐
│ ORB-SLAM3   │
│ Atlas       │
└──────┬──────┘
       │ VIO pose (in current map frame)
       │ Map ID
       │ Tracking state
       ▼
┌─────────────────────────────┐
│  Coordinate Transform Layer  │
│  - Detect map changes        │
│  - Calculate offsets         │
│  - Apply transformations     │
│  - Manage reset_counter      │
└──────┬──────────────────────┘
       │ NED pose (continuous frame)
       │ reset_counter (only for valid relocalization)
       ▼
┌─────────────────────────────┐
│  MAVLink Interface          │
│  VISION_POSITION_ESTIMATE   │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────┐
│ ArduPilot   │
│ EKF3        │
└─────────────┘
       ▲
       │ LOCAL_POSITION_NED (feedback)
       │
```

---

## Implementation Specifications

### Required State Variables

```cpp
struct VIOBridgeState {
    // Coordinate transformation
    Eigen::Vector3d vio_to_ned_offset;      // Offset from VIO frame to NED frame
    Eigen::Quaterniond vio_to_ned_rotation; // Rotation from VIO frame to NED frame
    
    // Map tracking
    int current_map_id;                     // Current ORB-SLAM3 map ID
    int previous_map_id;                    // Previous map ID for change detection
    
    // EKF feedback
    Eigen::Vector3d last_ekf_position;      // Last known position from ArduPilot
    uint32_t last_ekf_update_ms;            // Timestamp of last EKF feedback
    
    // Reset counter management
    uint8_t reset_counter;                  // Increments only for valid relocalization
    
    // Safety flags
    bool first_pose_received;               // Has any pose been received?
    bool vio_initialized;                   // Is VIO system initialized?
    uint32_t last_pose_time_ms;             // For timeout detection
};
```

### Main Processing Loop

```cpp
void processORBSLAMPose(const Sophus::SE3d& T_world_camera, 
                        int map_id,
                        ORB_SLAM3::Tracking::eTrackingState tracking_state,
                        float tracker_confidence) {
    
    uint32_t current_time_ms = getCurrentTimeMs();
    
    // 1. Extract VIO position and orientation
    Eigen::Vector3d vio_position = T_world_camera.translation();
    Eigen::Quaterniond vio_orientation = T_world_camera.unit_quaternion();
    
    // 2. Handle tracking state
    switch (tracking_state) {
        case ORB_SLAM3::Tracking::OK:
            handleTrackingOK(vio_position, vio_orientation, map_id);
            break;
            
        case ORB_SLAM3::Tracking::RECENTLY_LOST:
            handleRecentlyLost();
            break;
            
        case ORB_SLAM3::Tracking::LOST:
        case ORB_SLAM3::Tracking::NOT_INITIALIZED:
            handleTrackingLost();
            return; // Don't send any messages
            
        default:
            return;
    }
    
    // 3. Apply coordinate transformation
    Eigen::Vector3d ned_position = applyCoordinateTransform(vio_position);
    Eigen::Quaterniond ned_orientation = applyRotationTransform(vio_orientation);
    
    // 4. Send to ArduPilot
    sendVisionPositionEstimate(ned_position, ned_orientation, 
                               tracker_confidence, current_time_ms);
    
    // 5. Update state
    last_pose_time_ms = current_time_ms;
    updateEKFPosition();
}
```

### Map Change Detection and Handling

```cpp
void handleTrackingOK(const Eigen::Vector3d& vio_position,
                     const Eigen::Quaterniond& vio_orientation,
                     int map_id) {
    
    // Detect map change
    if (map_id != current_map_id) {
        
        if (!first_pose_received) {
            // Very first map - initialize with no offset
            current_map_id = map_id;
            first_pose_received = true;
            vio_initialized = true;
            logInfo("VIO initialized with map %d", map_id);
            return;
        }
        
        // Map has changed - determine type of change
        handleMapTransition(vio_position, vio_orientation, map_id);
        current_map_id = map_id;
    }
}

void handleMapTransition(const Eigen::Vector3d& new_vio_position,
                        const Eigen::Quaterniond& new_vio_orientation,
                        int new_map_id) {
    
    // Check if this is relocalization to existing map vs new map creation
    bool is_relocalization = checkIfRelocalization(new_map_id);
    
    if (is_relocalization) {
        // SCENARIO 1: Relocalized to existing map with known relationship
        handleRelocalization(new_vio_position, new_vio_orientation, new_map_id);
    } else {
        // SCENARIO 2: New map created at arbitrary origin
        handleNewMapCreation(new_vio_position, new_vio_orientation, new_map_id);
    }
}

void handleRelocalization(const Eigen::Vector3d& new_vio_position,
                         const Eigen::Quaterniond& new_vio_orientation,
                         int new_map_id) {
    
    // For relocalization to a previously visited map, we can increment reset_counter
    // BUT only if the maps share a known coordinate system
    
    // Check if maps are merged in Atlas
    if (mapsAreMerged(current_map_id, new_map_id)) {
        // Maps share coordinate system - safe to increment reset_counter
        reset_counter++;
        
        logInfo("Relocalized from map %d to map %d (merged), reset_counter=%d",
                current_map_id, new_map_id, reset_counter);
        
        // Update offset based on known map relationship
        updateOffsetForMergedMaps(new_map_id);
    } else {
        // Maps are not merged - treat as new map creation
        handleNewMapCreation(new_vio_position, new_vio_orientation, new_map_id);
    }
}

void handleNewMapCreation(const Eigen::Vector3d& new_vio_position,
                         const Eigen::Quaterniond& new_vio_orientation,
                         int new_map_id) {
    
    // NEW MAP at arbitrary origin - calculate offset to maintain continuity
    
    // Get current EKF position (where drone actually is)
    Eigen::Vector3d current_ekf_position = last_ekf_position;
    
    // Calculate offset: where EKF thinks we are minus where VIO thinks we are
    vio_to_ned_offset = current_ekf_position - new_vio_position;
    
    logWarning("New map %d created at origin. Applying position offset [%.2f, %.2f, %.2f]",
               new_map_id, 
               vio_to_ned_offset.x(), 
               vio_to_ned_offset.y(), 
               vio_to_ned_offset.z());
    
    // TODO: Handle yaw alignment if needed
    // New map may have arbitrary yaw in visual-only mode
    // or gravity-aligned yaw in VI mode
    
    // CRITICAL: Do NOT increment reset_counter for new map creation
    // Keep reset_counter unchanged to avoid position snap
}

bool checkIfRelocalization(int new_map_id) {
    // Check if this map ID has been seen before
    // This requires maintaining a history of map IDs
    
    // Simple implementation: check if it's a previously seen ID
    return (map_history.find(new_map_id) != map_history.end());
}
```

### Coordinate Transformation Application

```cpp
Eigen::Vector3d applyCoordinateTransform(const Eigen::Vector3d& vio_position) {
    // Apply position offset
    Eigen::Vector3d ned_position = vio_position + vio_to_ned_offset;
    
    // Apply any rotation transform (for yaw alignment)
    ned_position = vio_to_ned_rotation * ned_position;
    
    return ned_position;
}

Eigen::Quaterniond applyRotationTransform(const Eigen::Quaterniond& vio_orientation) {
    // Apply rotation offset (for yaw alignment)
    return vio_to_ned_rotation * vio_orientation;
}
```

### EKF Position Feedback

```cpp
void updateEKFPosition() {
    // Subscribe to LOCAL_POSITION_NED message from ArduPilot
    mavlink_message_t msg;
    
    if (receiveMAVLinkMessage(msg, MAVLINK_MSG_ID_LOCAL_POSITION_NED, 
                             /*blocking=*/false, /*timeout_ms=*/10)) {
        
        mavlink_local_position_ned_t pos_ned;
        mavlink_msg_local_position_ned_decode(&msg, &pos_ned);
        
        last_ekf_position = Eigen::Vector3d(pos_ned.x, pos_ned.y, pos_ned.z);
        last_ekf_update_ms = getCurrentTimeMs();
        
    } else {
        // No recent EKF feedback - check timeout
        uint32_t time_since_update = getCurrentTimeMs() - last_ekf_update_ms;
        if (time_since_update > 1000) {
            logWarning("No EKF position feedback for %d ms", time_since_update);
        }
    }
}
```

### Tracking Loss Handling

```cpp
void handleRecentlyLost() {
    // Short-term tracking loss (< 5 seconds)
    // Continue sending last valid position with high covariance
    
    // Inflate covariance significantly
    float position_covariance = 10.0f;  // meters (vs nominal ~0.1-0.5m)
    
    sendVisionPositionEstimate(last_valid_ned_position,
                              last_valid_ned_orientation,
                              /*confidence=*/1,  // Low confidence
                              getCurrentTimeMs(),
                              position_covariance);
    
    logWarning("Tracking recently lost, sending high-uncertainty estimate");
}

void handleTrackingLost() {
    // Extended tracking loss
    // STOP sending VISION_POSITION_ESTIMATE messages
    // Let ArduPilot's EKF timeout trigger failsafe
    
    logError("Tracking lost, suspending vision messages. EKF will use IMU propagation.");
    
    // Optionally send GCS status message
    sendStatusTextToGCS("VIO tracking lost", MAV_SEVERITY_WARNING);
    
    // Do NOT send any position messages
    // Do NOT increment reset_counter
    // Do NOT send zeros
}
```

### MAVLink Message Construction

```cpp
void sendVisionPositionEstimate(const Eigen::Vector3d& position,
                               const Eigen::Quaterniond& orientation,
                               float confidence,
                               uint32_t timestamp_ms,
                               float position_covariance = 0.1f) {
    
    mavlink_message_t msg;
    
    // Convert timestamp to microseconds
    uint64_t usec = timestamp_ms * 1000ULL;
    
    // Position covariance array (6x6 matrix, upper triangle)
    // For simplicity, use diagonal values only
    float covariance[21] = {0};
    covariance[0] = position_covariance * position_covariance;  // x variance
    covariance[6] = position_covariance * position_covariance;  // y variance
    covariance[11] = position_covariance * position_covariance; // z variance
    covariance[15] = 0.1f * 0.1f;  // roll variance
    covariance[18] = 0.1f * 0.1f;  // pitch variance
    covariance[20] = 0.1f * 0.1f;  // yaw variance
    
    // Pack message
    mavlink_msg_vision_position_estimate_pack(
        system_id,
        component_id,
        &msg,
        usec,                    // usec: Timestamp (UNIX Epoch time)
        position.x(),            // x: Global X position (NED)
        position.y(),            // y: Global Y position (NED)
        position.z(),            // z: Global Z position (NED)
        orientation.x(),         // roll: Roll angle
        orientation.y(),         // pitch: Pitch angle
        orientation.z(),         // yaw: Yaw angle
        covariance,              // covariance: Row-major representation
        reset_counter            // reset_counter: Estimate reset counter
    );
    
    // Send message
    sendMAVLinkMessage(msg);
}
```

---

## ArduPilot Configuration

### Essential Parameters

```python
# Basic VIO Setup
SERIAL2_PROTOCOL = 2          # MAVLink2 on telemetry port
VISO_TYPE = 1                 # MAVLink (use 2 for T265-specific features)

# EKF3 Source Configuration
EK3_SRC1_POSXY = 6           # ExternalNav for XY position
EK3_SRC1_POSZ = 1            # Barometer for Z (recommended over VIO height)
EK3_SRC1_VELXY = 0           # None (or 6 if sending velocity)
EK3_SRC1_YAW = 6             # ExternalNav for yaw

# VIO Noise and Quality Parameters
VISO_POS_M_NSE = 0.1         # Position measurement noise floor (meters)
VISO_YAW_M_NSE = 0.1         # Yaw measurement noise floor (radians)
VISO_QUAL_MIN = 1            # Minimum quality to accept (0-100)

# Position offset from IMU to camera (if applicable)
VISO_POS_X = 0.0             # X offset in meters
VISO_POS_Y = 0.0             # Y offset in meters
VISO_POS_Z = 0.0             # Z offset in meters

# Camera orientation (0=forward, 2=downward, etc.)
VISO_ORIENT = 0              # Forward-facing

# EKF Failsafe
FS_EKF_ACTION = 1            # Land on EKF failsafe
FS_EKF_THRESH = 0.8          # EKF variance threshold

# Optional: RC switch for yaw realignment
RC7_OPTION = 80              # Viso Align function
```

### Recommended Settings for ORB-SLAM3

```python
# Since ORB-SLAM3 height estimates can be noisy
EK3_SRC1_POSZ = 1            # Use barometer for altitude

# Conservative position noise (ORB-SLAM3 less accurate than T265)
VISO_POS_M_NSE = 0.3         # 30cm noise floor

# Higher quality threshold (ORB-SLAM3 confidence less calibrated than T265)
VISO_QUAL_MIN = 25           # Only accept medium+ confidence

# Glitch protection
EK3_GLITCH_RAD = 25          # Position glitch radius (meters)
```

### Pre-Flight Checklist

1. **Set EKF Origin:**
   ```
   # In Mission Planner, right-click map:
   "Set Home Here" → "Set EKF Origin Here"
   ```

2. **Verify Message Reception:**
   ```
   # Mission Planner → Ctrl+F → MAVLink Inspector
   # Check for VISION_POSITION_ESTIMATE messages at ~20-30Hz
   ```

3. **Check Confidence Level:**
   ```
   # Monitor VISION_POSITION_DELTA.confidence field
   # Should be > VISO_QUAL_MIN
   ```

4. **Align Yaw (if using RC7_OPTION=80):**
   ```
   # Toggle RC channel 7 before takeoff to align VIO yaw with AHRS
   ```

---

## Testing and Validation

### Unit Tests

```cpp
// Test coordinate offset calculation
void testCoordinateOffsetCalculation() {
    VIOBridgeState state;
    state.last_ekf_position = Eigen::Vector3d(10.0, 5.0, -2.0);
    
    // Simulate new map at origin
    Eigen::Vector3d new_vio_position(0.0, 0.0, 0.0);
    
    // Calculate offset
    Eigen::Vector3d offset = state.last_ekf_position - new_vio_position;
    
    assert(offset.x() == 10.0);
    assert(offset.y() == 5.0);
    assert(offset.z() == -2.0);
    
    // Apply offset
    Eigen::Vector3d transformed = new_vio_position + offset;
    assert((transformed - state.last_ekf_position).norm() < 0.001);
}

// Test reset_counter management
void testResetCounterManagement() {
    VIOBridgeState state;
    state.reset_counter = 5;
    uint8_t initial = state.reset_counter;
    
    // New map creation should NOT increment
    handleNewMapCreation(Eigen::Vector3d(0,0,0), 
                        Eigen::Quaterniond::Identity(), 42);
    assert(state.reset_counter == initial);
    
    // Relocalization to merged map SHOULD increment
    handleRelocalization(Eigen::Vector3d(1,1,1),
                        Eigen::Quaterniond::Identity(), 43);
    assert(state.reset_counter == initial + 1);
}
```

### Integration Tests

**Test 1: Normal Flight Operation**
```
1. Arm drone, takeoff to 2m
2. Hover for 10 seconds
3. Verify position stable (< 1m drift)
4. Perform gentle maneuvers
5. Land
```

**Test 2: Tracking Loss Recovery (Same Map)**
```
1. Arm drone, takeoff to 2m
2. Cover cameras briefly (simulate tracking loss)
3. Uncover cameras
4. Verify tracking recovers without position jump
5. Expected: reset_counter unchanged, smooth recovery
```

**Test 3: New Map Creation (Extended Loss)**
```
1. Arm drone, takeoff to 2m
2. Cover cameras for >5 seconds (force new map)
3. Uncover cameras
4. Verify position continuity maintained
5. Expected: reset_counter unchanged, coordinate offset applied
6. Land successfully
```

**Test 4: Failsafe Behavior**
```
1. Arm drone, takeoff to 2m
2. Disable VIO messages completely
3. Verify EKF timeout triggers
4. Expected: Controlled landing via FS_EKF_ACTION
```

### Monitoring and Logging

**Key Metrics to Log:**
```cpp
struct VIOTelemetry {
    uint32_t timestamp_ms;
    
    // ORB-SLAM3 state
    int map_id;
    ORB_SLAM3::Tracking::eTrackingState tracking_state;
    Eigen::Vector3d vio_position;
    float tracker_confidence;
    
    // Coordinate transformation
    Eigen::Vector3d vio_to_ned_offset;
    Eigen::Vector3d ned_position;
    uint8_t reset_counter;
    
    // EKF feedback
    Eigen::Vector3d ekf_position;
    Eigen::Vector3d position_error;  // |ned_position - ekf_position|
    
    // Health metrics
    uint32_t messages_sent;
    uint32_t messages_dropped;
    uint32_t map_transitions;
    uint32_t tracking_loss_events;
};
```

**Health Checks:**
```cpp
bool performHealthCheck() {
    bool healthy = true;
    
    // Check 1: Recent pose data
    if (getCurrentTimeMs() - last_pose_time_ms > 500) {
        logError("No pose data for >500ms");
        healthy = false;
    }
    
    // Check 2: EKF feedback
    if (getCurrentTimeMs() - last_ekf_update_ms > 1000) {
        logWarning("No EKF feedback for >1s");
        healthy = false;
    }
    
    // Check 3: Position error
    float position_error = (ned_position - last_ekf_position).norm();
    if (position_error > 5.0) {
        logError("Large position error: %.2fm", position_error);
        healthy = false;
    }
    
    // Check 4: Excessive map changes
    if (map_transitions_last_minute > 10) {
        logWarning("Excessive map changes: %d/min", map_transitions_last_minute);
        healthy = false;
    }
    
    return healthy;
}
```

---

## Troubleshooting

### Issue: Position jumps to home after tracking recovery

**Symptoms:**
- Drone position on GCS map jumps back to home location
- Occurs after ORB-SLAM3 tracking loss/recovery

**Diagnosis:**
```bash
# Check logs for:
- "New map with id: X" messages from ORB-SLAM3
- reset_counter incrementing in telemetry
- Large position offset being applied
```

**Solution:**
- Verify `reset_counter` is NOT incremented for new map creation
- Ensure coordinate offset is calculated correctly
- Check that EKF position feedback is being received

### Issue: Toilet bowling (circular drift)

**Symptoms:**
- Drone drifts in circles during position hold
- Yaw oscillation

**Diagnosis:**
- Yaw alignment mismatch between VIO and EKF

**Solution:**
```python
# Enable yaw alignment on RC switch
RC7_OPTION = 80  # Viso Align

# Or implement yaw offset in coordinate transform layer
vio_to_ned_rotation = getYawAlignmentRotation();
```

### Issue: EKF variance exceeds threshold

**Symptoms:**
- "EKF3 IMU0 ext nav variance" warnings
- FS_EKF_ACTION triggered

**Diagnosis:**
```bash
# Check:
- VISION_POSITION_ESTIMATE message rate (should be 20-30Hz)
- Covariance values being sent
- VISO_POS_M_NSE parameter value
```

**Solution:**
```python
# Reduce minimum noise floor
VISO_POS_M_NSE = 0.05  # Was 0.3

# Increase EKF glitch radius
EK3_GLITCH_RAD = 50    # Was 25

# Check actual VIO quality/accuracy
```

### Issue: Height drift over time

**Symptoms:**
- Altitude estimate drifts up or down
- Discrepancy between VIO and barometer

**Diagnosis:**
- VIO height estimates accumulating error
- Barometer not being used for height

**Solution:**
```python
# Use barometer for height
EK3_SRC1_POSZ = 1  # Barometer

# VIO only provides XY position
EK3_SRC1_POSXY = 6  # ExternalNav
```

### Issue: Messages not reaching ArduPilot

**Symptoms:**
- MAVLink Inspector shows no VISION_POSITION_ESTIMATE
- "EKF3 IMU0 waiting for external nav data"

**Diagnosis:**
```bash
# Check serial connection
ls /dev/ttyUSB* /dev/ttyACM*

# Check baud rate
# Companion: 921600
# ArduPilot: SERIAL2_BAUD = 921

# Test MAVLink connection
mavproxy.py --master=/dev/ttyUSB0 --baudrate=921600
```

**Solution:**
- Verify physical serial connection
- Check SERIAL2_PROTOCOL = 2
- Verify message is properly packed (check system_id, component_id)
- Use MAVLink sniffer to debug: `mavproxy.py --master=/dev/ttyUSB0 --out=udp:127.0.0.1:14550`

---

## Additional Resources

### Reference Implementation
The Intel T265 integration script provides a working example of coordinate transforms:
- GitHub: [thien94/vision_to_mavros](https://github.com/thien94/vision_to_mavros)
- Key file: `scripts/t265_to_mavlink.py`

### ArduPilot Documentation
- [Non-GPS Position Estimation](https://ardupilot.org/dev/docs/mavlink-nongps-position-estimation.html)
- [Intel T265 Setup](https://ardupilot.org/copter/docs/common-vio-tracking-camera.html)
- [EKF Source Selection](https://ardupilot.org/copter/docs/common-ekf-sources.html)

### ORB-SLAM3 Resources
- [ORB-SLAM3 Paper (arXiv:2007.11898)](https://arxiv.org/abs/2007.11898)
- [ORB-SLAM3 GitHub](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- Atlas Multi-Map System documentation in paper Section III

### MAVLink Protocol
- [VISION_POSITION_ESTIMATE Message Definition](https://mavlink.io/en/messages/common.html#VISION_POSITION_ESTIMATE)
- [LOCAL_POSITION_NED Message](https://mavlink.io/en/messages/common.html#LOCAL_POSITION_NED)

---

## Implementation Checklist

- [ ] Set up MAVLink communication with ArduPilot
- [ ] Implement VIOBridgeState structure with all required variables
- [ ] Create map change detection logic
- [ ] Implement coordinate offset calculation
- [ ] Apply coordinate transformations before sending to ArduPilot
- [ ] Set up EKF position feedback loop (LOCAL_POSITION_NED)
- [ ] Implement proper reset_counter management
- [ ] Handle tracking loss scenarios (RECENTLY_LOST, LOST)
- [ ] Add telemetry logging
- [ ] Configure ArduPilot parameters
- [ ] Write unit tests for coordinate transforms
- [ ] Perform integration testing
- [ ] Implement health monitoring and failsafes
- [ ] Document operational procedures

---

## Conclusion

The key to successful ORB-SLAM3 + ArduPilot integration is **never sending discontinuous position data to the flight controller**. Handle all coordinate frame changes in your integration layer by:

1. Maintaining a coordinate offset between VIO and NED frames
2. Only incrementing `reset_counter` for valid relocalization events
3. Stopping messages during extended tracking loss (let EKF failsafe handle it)
4. Monitoring EKF feedback to stay synchronized with ArduPilot's position estimate

This approach allows the drone to continue flying through VIO map transitions while maintaining position control safety.
