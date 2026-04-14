#!/bin/bash

# Generate timestamp with format: year_month_day_hour_minute_second
generate_timestamp() {
    # Use date command with format specifiers:
    # %Y - Year (4 digits)
    # %m - Month (01-12)
    # %d - Day (01-31)
    # %H - Hour (00-23)
    # %M - Minute (00-59)
    # %S - Second (00-59)
    timestamp=$(date '+%Y_%m_%d_%H_%M_%S')
    echo "$timestamp"
}

# Function to check and create directories and file
create_file_and_dirs() {
    local file_path="$1"
    
    # Extract the directory path from the file path
    dir_path=$(dirname "$file_path")
    
    # Check if directories exist, create them if they don't
    if [ ! -d "$dir_path" ]; then
        echo "Creating directories: $dir_path"
        # -p flag creates parent directories as needed
        mkdir -p "$dir_path"
        
        # Check if directory creation was successful
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create directories."
            exit 1
        fi
        echo "Directories created successfully."
    else
        echo "Directories already exist: $dir_path"
    fi
    
    # Check if file exists, create it if it doesn't
    if [ ! -f "$file_path" ]; then
        echo "Creating file: $file_path"
        touch "$file_path"
        
        # Check if file creation was successful
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create file."
            exit 1
        fi
        echo "File created successfully."
    else
        echo "File already exists: $file_path"
    fi
}

#FILELOCK=/tmp/ramdisk/lock2.lock

#echo "orca shell stated\n" >> /home/nvidia/usr/orca_launcher/log2.txt
timestamp=$(generate_timestamp)
#echo "$timestamp"
FULL_PATH=~/vin_logs/$timestamp.txt


#if [ -f "$FILELOCK" ]; then
    create_file_and_dirs "$FULL_PATH"
    #touch ~/vin_logs/log.lock
    #echo $FULL_PATH > ~/vin_logs/log.lock
    date >> $FULL_PATH 
    echo "Launch VIO\n" >> $FULL_PATH 
    echo "Launch VIO"
    #~/ORB_SLAM3/Examples/Stereo-Inertial
    #./mono_inertial_pear ../../Vocabulary/ORBvoc.txt custom_cam_imu_72deg.yaml 2 >> $FULL_PATH 2>&1
    ./../Stereo-Inertial/stereo_inertial_pear_direct ../../Vocabulary/ORBvoc.txt ../Stereo-Inertial/orbslam3_stereo_inertial.yaml /dev/ttyACM0 /dev/ttyAMA0 1500000 2 0 --brightness-ctrl  >> $FULL_PATH 2>&1
    #./mono_inertial_pear ../../Vocabulary/ORBvoc.txt custom_cam_imu_140deg_arducam.yaml /dev/ttyACM0 /dev/ttyTHS1 1500000 2 0 --autogain-on-lost >> $FULL_PATH 2>&1
    #python3.8 /tmp/ramdisk/orca/drn-orca/ALGO_MAIN.py >> $FULL_PATH 2>&1
#else
    #echo "Main Script is not ready"
    #date >> $FULL_PATH 
    #echo "Main Script is not ready" >> $FULL_PATH 

#fi

