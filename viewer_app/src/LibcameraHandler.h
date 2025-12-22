#ifndef LIBCAMERA_HANDLER_H
#define LIBCAMERA_HANDLER_H

// This file wraps libcamera to avoid Qt MOC conflicts
// Qt defines 'slots' and 'emit' as macros which conflict with libcamera

// Temporarily undefine Qt macros
#ifdef slots
#undef slots
#define QT_SLOTS_DEFINED
#endif

#ifdef emit
#undef emit
#define QT_EMIT_DEFINED
#endif

#ifdef signals
#undef signals
#define QT_SIGNALS_DEFINED
#endif

// Now include libcamera
#include <libcamera/libcamera.h>

// Restore Qt macros
#ifdef QT_SLOTS_DEFINED
#define slots Q_SLOTS
#undef QT_SLOTS_DEFINED
#endif

#ifdef QT_EMIT_DEFINED
#define emit Q_EMIT
#undef QT_EMIT_DEFINED
#endif

#ifdef QT_SIGNALS_DEFINED
#define signals Q_SIGNALS
#undef QT_SIGNALS_DEFINED
#endif

#endif // LIBCAMERA_HANDLER_H
