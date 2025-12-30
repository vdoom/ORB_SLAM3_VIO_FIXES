#ifndef AXES3D_WIDGET_H
#define AXES3D_WIDGET_H

#include <QWidget>
#include <QMatrix4x4>
#include <QVector3D>

/**
 * Axes3DWidget - 3D visualization widget showing X, Y, Z axes
 *
 * Uses QPainter for 2D rendering with 3D projection (no OpenGL required).
 * Used as base for GyroView and AccelView to display IMU orientation/data.
 * Displays three colored arrows:
 *   - X axis: Red
 *   - Y axis: Green
 *   - Z axis: Blue
 * Plus an optional white data vector representing sensor reading.
 */
class Axes3DWidget : public QWidget
{
    Q_OBJECT

public:
    explicit Axes3DWidget(QWidget *parent = nullptr);
    ~Axes3DWidget() override;

    void setTitle(const QString &title) { title_ = title; update(); }
    QString title() const { return title_; }

    // Set rotation for the axes (will be used for IMU visualization)
    void setRotation(float pitch, float roll, float yaw);

    // Set the data vector to display (white arrow showing sensor reading)
    // The vector is automatically normalized for display, but original magnitude affects thickness
    void setDataVector(const QVector3D &vector);
    void clearDataVector();

    // Get current data vector
    QVector3D dataVector() const { return dataVector_; }
    bool hasDataVector() const { return hasDataVector_; }

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QPointF project3Dto2D(const QVector3D &point3D, const QSize &widgetSize);
    void drawArrow(QPainter &painter, const QPointF &from, const QPointF &to, const QColor &color, float lineWidth = 2.5f);

    QString title_;

    float pitch_ = 0.0f;
    float roll_ = 0.0f;
    float yaw_ = 0.0f;

    // Data vector for sensor reading visualization
    QVector3D dataVector_;
    bool hasDataVector_ = false;
};

#endif // AXES3D_WIDGET_H
