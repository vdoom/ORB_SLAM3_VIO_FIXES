# VIO Status Reporting via MAVLink

## Context
The VIO system (`mono_inertial_pear.cc`) currently sends pose data to ArduPilot via MAVLink but has no way to report system lifecycle status (loading, tracking state, IMU initialization stages). The flight controller and GCS have no visibility into what the VIO system is doing. We'll add dual reporting: STATUSTEXT for human-readable GCS display + NAMED_VALUE_INT for programmatic use by ArduPilot Lua scripts.

**Design goal**: The MAVLink status reporting layer must be VIO-backend-agnostic. It will be reused for OpenVINS integration later. The status events and MAVLink sending code live in the application layer (mono_inertial_pear.cc), not inside ORB-SLAM3 core. ORB-SLAM3 only exposes a generic event queue that the application consumes.

## Status Events (Generic VIO Status Codes)

| Event | STATUSTEXT message | NAMED_VALUE_INT | Severity |
|-------|-------------------|-----------------|----------|
| VIO loading | `VIO: Loading dictionary` | 1 | INFO |
| VIO started | `VIO: System started` | 2 | INFO |
| IMU initialized | `VIO: IMU initialized` | 3 | NOTICE |
| VIBA1 start | `VIO: VIBA1 started` | 4 | NOTICE |
| VIBA1 end | `VIO: VIBA1 complete` | 5 | NOTICE |
| VIBA2 start | `VIO: VIBA2 started` | 6 | NOTICE |
| VIBA2 end | `VIO: VIBA2 complete` | 7 | NOTICE |
| Tracking recently lost | `VIO: Tracking lost briefly` | 8 | WARNING |
| Tracking lost + restart | `VIO: Tracking lost, restarting` | 9 | ERROR |
| System crashed | `VIO: Crashed` | 10 | CRITICAL |

NAMED_VALUE_INT name: `"VIO_STAT"` (max 10 chars in MAVLink spec)

Status codes 1-3 and 8-10 are generic (apply to any VIO backend).
Status codes 4-7 (VIBA) are ORB-SLAM3-specific; OpenVINS will define its own equivalent events in the same range.

## Architecture

```
+------------------+     +------------------+     +------------------+
| VIO Backend      |     | Application      |     | MAVLink/ArduPilot|
| (ORB-SLAM3 or   |---->| (mono_inertial_  |---->| (STATUSTEXT +   |
|  OpenVINS)       |     |  pear.cc)        |     |  NAMED_VALUE_INT)|
| Push VIOEvents   |     | Poll events,     |     |                  |
|                  |     | map to MAVLink   |     |                  |
+------------------+     +------------------+     +------------------+
```

The **VIOStatusReporter** class in the application layer:
- Receives events from any VIO backend (via a generic event queue)
- Maintains a state machine to avoid duplicate messages
- Sends STATUSTEXT + NAMED_VALUE_INT via MAVLinkInterface
- Is backend-agnostic — same class reused for OpenVINS

## Files Modified

### 1. `include/System.h` — VIO event queue
- `VIOEvent` enum: IMU_INITIALIZED, VIBA1_START, VIBA1_END, VIBA2_START, VIBA2_END
- Thread-safe queue: `PushVIOEvent()` / `PopVIOEvent()`
- Private: `std::queue<VIOEvent>` + `std::mutex`

### 2. `src/System.cc` — Event queue implementation
- `PushVIOEvent()`: lock mutex, push to queue
- `PopVIOEvent()`: lock mutex, pop front if available, return bool

### 3. `src/LocalMapping.cc` — Event sources
- IMU_INITIALIZED pushed after first successful `InitializeIMU()` when `isImuInitialized()` becomes true
- VIBA1_START/END pushed around VIBA1's `InitializeIMU()` call
- VIBA2_START/END pushed around VIBA2's `InitializeIMU()` call

### 4. `Examples/Monocular-Inertial/mono_inertial_pear.cc` — Main integration

**MAVLinkInterface additions:**
- `sendStatusText(severity, text)` — packs and sends STATUSTEXT (50 char max)
- `sendNamedValueInt(name, value)` — packs and sends NAMED_VALUE_INT

**VIOStatusReporter class (backend-agnostic):**
- `report(code, severity, text)` — sends both STATUSTEXT + NAMED_VALUE_INT
- `processORBSLAM3Event(event)` — maps ORB-SLAM3 VIOEvents to status codes
- `processTrackingState(state)` — tracks state transitions, reports RECENTLY_LOST and LOST

**Initialization sequence:**
1. MAVLinkInterface created first, connection established
2. VIOStatusReporter created
3. Report "VIO: Loading dictionary" (code 1)
4. ORB_SLAM3::System constructor (blocks during dictionary load)
5. Report "VIO: System started" (code 2)
6. VIOBridge created with shared MAVLinkInterface
7. Main loop: poll VIOEvents + track state changes

**Crash detection:**
- Signal handlers (SIGSEGV, SIGABRT, SIGFPE) send "VIO: Crashed" via global MAVLink pointer
- try/catch around main loop sends crash status on exceptions

## OpenVINS Extensibility
When OpenVINS is integrated later:
- OpenVINS provides its own event mechanism (callbacks or polling)
- Application maps OpenVINS events to the same status codes (1-3, 8-10 identical; 4-7 replaced with OpenVINS-specific init stages)
- VIOStatusReporter and MAVLinkInterface are reused unchanged
- Only the event mapping function (`processORBSLAM3Event` equivalent) needs a new implementation

## Verification
1. Build: `cmake --build build`
2. Run on device with ArduPilot connected
3. Check Mission Planner "Messages" tab for STATUSTEXT messages
4. Check MAVLink Inspector for NAMED_VALUE_INT with name "VIO_STAT"
5. Verify lifecycle: Loading → Started → IMU init → VIBA1 start/end → VIBA2 start/end
6. Test tracking loss by covering camera — verify warning/error messages
7. Test crash handling — verify crash message sent

## ArduPilot Lua Script Example
```lua
-- Read VIO status from NAMED_VALUE_INT
function update()
    local vio_stat = named_value_int:get("VIO_STAT")
    if vio_stat then
        if vio_stat >= 9 then
            -- VIO tracking lost or crashed
            gcs:send_text(0, "VIO ALERT: status=" .. tostring(vio_stat))
        end
    end
    return update, 1000  -- check every 1s
end
return update()
```
