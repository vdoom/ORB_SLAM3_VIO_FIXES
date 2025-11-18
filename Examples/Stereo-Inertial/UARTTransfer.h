#ifndef UART_TRANSFER_H
#define UART_TRANSFER_H

#include <string>
#include <vector>
#include <cstdint>
#include <termios.h>  // For speed_t type

class UARTTransfer {
public:
    /**
     * Constructor
     * @param port UART device port (e.g., "/dev/ttyTHS0")
     * @param baudrate Communication speed (default: 115200)
     */
    UARTTransfer(const std::string& port = "/dev/ttyTHS0", int baudrate = 115200);
    
    /**
     * Destructor - automatically closes connection
     */
    ~UARTTransfer();
    
    /**
     * Establish UART connection
     * @return true if successful, false otherwise
     */
    bool connect();
    
    /**
     * Close UART connection
     */
    void disconnect();
    
    /**
     * Check if connection is open
     * @return true if connected, false otherwise
     */
    bool isConnected() const;
    
    /**
     * Send text data via UART
     * @param data String to send
     * @return Number of bytes sent, -1 on error
     */
    int sendData(const std::string& data);
    
    /**
     * Send binary data via UART
     * @param data Vector of bytes to send
     * @return Number of bytes sent, -1 on error
     */
    int sendData(const std::vector<uint8_t>& data);
    
    /**
     * Receive data via UART
     * @param buffer Buffer to store received data
     * @param maxSize Maximum number of bytes to read
     * @param timeoutMs Timeout in milliseconds (0 = non-blocking)
     * @return Number of bytes received, -1 on error
     */
    int receiveData(std::vector<uint8_t>& buffer, size_t maxSize = 1024, int timeoutMs = 1000);
    
    /**
     * Receive text data via UART
     * @param data String to store received data
     * @param maxSize Maximum number of bytes to read
     * @param timeoutMs Timeout in milliseconds
     * @return Number of bytes received, -1 on error
     */
    int receiveData(std::string& data, size_t maxSize = 1024, int timeoutMs = 1000);
    
    /**
     * Send a file via UART
     * @param filepath Path to file to send
     * @param callback Progress callback function (optional)
     * @return true if successful, false otherwise
     */
    bool sendFile(const std::string& filepath, 
                  void (*callback)(size_t current, size_t total) = nullptr);
    
    /**
     * Receive a file via UART
     * @param saveDir Directory to save received file
     * @param callback Progress callback function (optional)
     * @return true if successful, false otherwise
     */
    bool receiveFile(const std::string& saveDir = ".", 
                     void (*callback)(size_t current, size_t total) = nullptr);
    
    /**
     * Set baud rate
     * @param baudrate New baud rate
     * @return true if successful, false otherwise
     */
    bool setBaudrate(int baudrate);
    
    /**
     * Get current baud rate
     * @return Current baud rate
     */
    int getBaudrate() const;
    
    /**
     * Flush input and output buffers
     */
    void flush();
    
    /**
     * Get number of bytes available to read
     * @return Number of bytes available, -1 on error
     */
    int available() const;

private:
    std::string port_;
    int baudrate_;
    int fd_;  // File descriptor
    bool connected_;
    
    // Internal helper methods
    bool configurePort();
    speed_t getBaudrateConstant(int baudrate) const;
    std::string getFilenameFromPath(const std::string& filepath) const;
    size_t getFileSize(const std::string& filepath) const;
};

#endif // UART_TRANSFER_H



