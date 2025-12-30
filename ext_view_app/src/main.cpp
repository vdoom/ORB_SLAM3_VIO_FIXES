#include "MainWindow.h"

#include <QApplication>
#include <QSurfaceFormat>
#include <QDebug>

int main(int argc, char *argv[])
{
    // Set OpenGL format for future 3D IMU visualization
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setSamples(4);  // Anti-aliasing
    QSurfaceFormat::setDefaultFormat(format);
    
    QApplication app(argc, argv);
    app.setApplicationName("RPi Camera Viewer");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("DroneProject");
    
    qDebug() << "Starting RPi Camera Viewer";
    qDebug() << "Qt version:" << qVersion();
    
    MainWindow mainWindow;
    mainWindow.show();
    
    return app.exec();
}
