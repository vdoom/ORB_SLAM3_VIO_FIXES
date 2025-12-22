#include "CameraWidget.h"

#include <QPainter>
#include <QPaintEvent>
#include <QResizeEvent>

CameraWidget::CameraWidget(QWidget *parent)
    : QWidget(parent)
{
    // Set dark background
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, Qt::black);
    setPalette(pal);
    
    // Enable smooth scaling hint
    setAttribute(Qt::WA_OpaquePaintEvent);
}

QSize CameraWidget::sizeHint() const
{
    return QSize(1280, 720);
}

QSize CameraWidget::minimumSizeHint() const
{
    return QSize(320, 240);
}

void CameraWidget::updateFrame(const QImage &frame)
{
    QMutexLocker locker(&frameMutex_);
    currentFrame_ = frame;
    needsScaling_ = true;
    
    // Schedule repaint
    update();
}

void CameraWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    
    QPainter painter(this);
    // Use fast rendering - SmoothPixmapTransform is too slow for real-time video
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
    
    // Fill background
    painter.fillRect(rect(), Qt::black);
    
    QMutexLocker locker(&frameMutex_);
    
    if (currentFrame_.isNull()) {
        // Draw placeholder text when no frame available
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter, "Waiting for camera...");
        return;
    }
    
    // Update scaled frame if needed
    if (needsScaling_) {
        updateScaledFrame();
    }
    
    // Draw the frame centered
    if (!scaledFrame_.isNull()) {
        painter.drawImage(drawRect_, scaledFrame_);
    }
}

void CameraWidget::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event);
    
    QMutexLocker locker(&frameMutex_);
    needsScaling_ = true;
}

void CameraWidget::updateScaledFrame()
{
    if (currentFrame_.isNull()) {
        return;
    }
    
    // Calculate scaled size maintaining aspect ratio
    QSize frameSize = currentFrame_.size();
    QSize widgetSize = size();
    
    QSize scaledSize = frameSize.scaled(widgetSize, Qt::KeepAspectRatio);
    
    // Calculate centered position
    int x = (widgetSize.width() - scaledSize.width()) / 2;
    int y = (widgetSize.height() - scaledSize.height()) / 2;
    drawRect_ = QRect(x, y, scaledSize.width(), scaledSize.height());
    
    // Scale the frame using fast transformation for real-time video
    // Qt::FastTransformation uses nearest-neighbor which is much faster than smooth
    scaledFrame_ = currentFrame_.scaled(scaledSize, Qt::KeepAspectRatio,
                                        Qt::FastTransformation);
    
    needsScaling_ = false;
}
