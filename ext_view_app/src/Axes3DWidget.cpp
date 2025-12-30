#include "Axes3DWidget.h"

#include <QPainter>
#include <QtMath>

Axes3DWidget::Axes3DWidget(QWidget *parent)
    : QWidget(parent)
{
    setMinimumSize(150, 150);
}

Axes3DWidget::~Axes3DWidget() = default;

void Axes3DWidget::setRotation(float pitch, float roll, float yaw)
{
    pitch_ = pitch;
    roll_ = roll;
    yaw_ = yaw;
    update();
}

void Axes3DWidget::setDataVector(const QVector3D &vector)
{
    dataVector_ = vector;
    hasDataVector_ = true;
    update();
}

void Axes3DWidget::clearDataVector()
{
    hasDataVector_ = false;
    update();
}

QPointF Axes3DWidget::project3Dto2D(const QVector3D &point3D, const QSize &widgetSize)
{
    // Simple orthographic-like projection with perspective hint
    float scale = qMin(widgetSize.width(), widgetSize.height()) * 0.35f;
    float centerX = widgetSize.width() / 2.0f;
    float centerY = widgetSize.height() / 2.0f;

    // Apply base camera rotation for isometric-like view (so all axes are visible)
    // Then apply user rotation (pitch, roll, yaw)
    QMatrix4x4 rotation;

    // Base camera view: tilt down 25° and rotate 35° to the right
    rotation.rotate(25.0f, 1.0f, 0.0f, 0.0f);   // Tilt down
    rotation.rotate(-35.0f, 0.0f, 1.0f, 0.0f);  // Rotate right

    // Apply user-controlled rotation on top
    rotation.rotate(pitch_, 1.0f, 0.0f, 0.0f);
    rotation.rotate(roll_, 0.0f, 0.0f, 1.0f);
    rotation.rotate(yaw_, 0.0f, 1.0f, 0.0f);

    QVector3D rotated = rotation.map(point3D);

    // Simple projection (Y is up in 3D, but down in screen coords)
    float x = centerX + rotated.x() * scale;
    float y = centerY - rotated.y() * scale;  // Flip Y for screen coordinates

    // Add slight depth effect
    float depthFactor = 1.0f + rotated.z() * 0.1f;
    x = centerX + (x - centerX) * depthFactor;
    y = centerY + (y - centerY) * depthFactor;

    return QPointF(x, y);
}

void Axes3DWidget::drawArrow(QPainter &painter, const QPointF &from, const QPointF &to, const QColor &color, float lineWidth)
{
    QPen pen(color, lineWidth);
    pen.setCapStyle(Qt::RoundCap);
    painter.setPen(pen);

    // Draw main line
    painter.drawLine(from, to);

    // Draw arrowhead
    QPointF direction = to - from;
    float length = qSqrt(direction.x() * direction.x() + direction.y() * direction.y());
    if (length < 1.0f) return;

    // Normalize direction
    direction /= length;

    // Arrowhead size proportional to line width
    float arrowSize = 8.0f + lineWidth * 1.5f;

    // Calculate arrowhead points
    QPointF perpendicular(-direction.y(), direction.x());
    QPointF arrowBase = to - direction * arrowSize;
    QPointF arrowLeft = arrowBase + perpendicular * (arrowSize * 0.5f);
    QPointF arrowRight = arrowBase - perpendicular * (arrowSize * 0.5f);

    // Draw arrowhead
    QPolygonF arrowHead;
    arrowHead << to << arrowLeft << arrowRight;
    painter.setBrush(color);
    painter.drawPolygon(arrowHead);
}

void Axes3DWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Dark background
    painter.fillRect(rect(), QColor(38, 38, 38));

    // Project 3D axis endpoints to 2D
    QPointF origin = project3Dto2D(QVector3D(0, 0, 0), size());
    QPointF xEnd = project3Dto2D(QVector3D(1.0f, 0, 0), size());
    QPointF yEnd = project3Dto2D(QVector3D(0, 1.0f, 0), size());
    QPointF zEnd = project3Dto2D(QVector3D(0, 0, 1.0f), size());

    // Calculate depths for sorting (draw back-to-front)
    // Must match the rotation in project3Dto2D
    QMatrix4x4 rotation;
    rotation.rotate(25.0f, 1.0f, 0.0f, 0.0f);   // Base tilt down
    rotation.rotate(-35.0f, 0.0f, 1.0f, 0.0f);  // Base rotate right
    rotation.rotate(pitch_, 1.0f, 0.0f, 0.0f);
    rotation.rotate(roll_, 0.0f, 0.0f, 1.0f);
    rotation.rotate(yaw_, 0.0f, 1.0f, 0.0f);

    QVector3D xDir = rotation.map(QVector3D(1, 0, 0));
    QVector3D yDir = rotation.map(QVector3D(0, 1, 0));
    QVector3D zDir = rotation.map(QVector3D(0, 0, 1));

    // Sort axes by depth (z component after rotation)
    struct AxisInfo {
        QPointF end;
        QColor color;
        QString label;
        float depth;
        float lineWidth;
        bool isDataVector;
    };

    QVector<AxisInfo> axes = {
        {xEnd, Qt::red, "X", xDir.z(), 2.5f, false},
        {yEnd, Qt::green, "Y", yDir.z(), 2.5f, false},
        {zEnd, QColor(100, 149, 237), "Z", zDir.z(), 2.5f, false}  // Cornflower blue for better visibility
    };

    // Add data vector if present
    if (hasDataVector_) {
        float magnitude = dataVector_.length();
        if (magnitude > 0.001f) {
            // Normalize vector for display, scale to unit length
            QVector3D normalizedVec = dataVector_.normalized();

            // Project the data vector endpoint
            QPointF dataEnd = project3Dto2D(normalizedVec, size());

            // Calculate depth for sorting
            QVector3D dataDir = rotation.map(normalizedVec);

            // Line width based on magnitude (thicker for larger values)
            float lineWidth = qBound(2.0f, magnitude * 2.0f, 5.0f);

            axes.append({dataEnd, Qt::white, "", dataDir.z(), lineWidth, true});
        }
    }

    // Sort by depth (draw furthest first)
    std::sort(axes.begin(), axes.end(), [](const AxisInfo &a, const AxisInfo &b) {
        return a.depth < b.depth;
    });

    // Draw axes back-to-front
    for (const auto &axis : axes) {
        drawArrow(painter, origin, axis.end, axis.color, axis.lineWidth);

        // Draw axis label (not for data vector)
        if (!axis.isDataVector && !axis.label.isEmpty()) {
            painter.setPen(axis.color);
            painter.setFont(QFont("Sans", 9, QFont::Bold));
            painter.drawText(axis.end + QPointF(5, 5), axis.label);
        }
    }

    // Draw title
    painter.setPen(Qt::white);
    painter.setFont(QFont("Sans", 10, QFont::Bold));
    painter.drawText(5, 15, title_);

    // Draw data vector magnitude if present
    if (hasDataVector_) {
        float magnitude = dataVector_.length();
        painter.setPen(QColor(200, 200, 200));
        painter.setFont(QFont("Sans", 8));
        QString magText = QString("|v|=%1").arg(magnitude, 0, 'f', 2);
        painter.drawText(5, height() - 5, magText);
    }
}
