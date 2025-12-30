#ifndef CAMERA_WIDGET_H
#define CAMERA_WIDGET_H

#include <QWidget>
#include <QImage>
#include <QMutex>

/**
 * CameraWidget - Displays camera frames
 * 
 * This widget displays camera frames scaled to fill the available space
 * while maintaining aspect ratio. It's designed to be placed in a layout
 * alongside other widgets (e.g., future IMU 3D views).
 */
class CameraWidget : public QWidget
{
    Q_OBJECT

public:
    explicit CameraWidget(QWidget *parent = nullptr);
    ~CameraWidget() override = default;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

public slots:
    void updateFrame(const QImage &frame);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    QImage currentFrame_;
    QImage scaledFrame_;
    QMutex frameMutex_;
    
    bool needsScaling_ = true;
    QRect drawRect_;
    
    void updateScaledFrame();
};

#endif // CAMERA_WIDGET_H
