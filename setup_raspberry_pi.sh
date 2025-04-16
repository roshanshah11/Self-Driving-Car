#!/bin/bash
# setup_raspberry_pi.sh
# Setup script for Autonomous Racer on Raspberry Pi

echo "Setting up Autonomous Racer on Raspberry Pi..."
echo "=============================================="

# Exit on error
set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then
  echo "Please do not run as root or with sudo. The script will use sudo when needed."
  exit 1
fi

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Raspberry Pi-specific dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libhdf5-dev \
    libqt5gui5 \
    libqtgui4 \
    libqt5test5 \
    libqt5core5a \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Install Python dependencies from requirements file
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install lighter version of OpenCV for Raspberry Pi
echo "Installing optimized OpenCV for Raspberry Pi..."
pip uninstall -y opencv-python  # Remove standard version
pip install opencv-python-headless  # Install headless version better for Pi

# Install production web server
echo "Installing Gunicorn for production web server..."
pip install gunicorn

# Setup systemd service for auto-start
echo "Setting up systemd service for auto-start..."
SERVICE_FILE="autonomous-racer.service"
cat > $SERVICE_FILE << EOF
[Unit]
Description=Autonomous Racer Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python $(pwd)/autonomous_racer.py --headless
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

# Install the service
sudo mv $SERVICE_FILE /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable autonomous-racer.service

# Configure Raspberry Pi settings for better performance
echo "Optimizing Raspberry Pi settings..."

# Disable swap for better microSD card life
echo "Disabling swap to extend SD card life (optional)..."
read -p "Disable swap? (y/n): " disable_swap
if [[ "$disable_swap" == "y" ]]; then
    sudo systemctl disable dphys-swapfile
    echo "Swap disabled. To re-enable: sudo systemctl enable dphys-swapfile"
fi

# Set CPU governor to performance for better real-time processing
echo "Setting CPU governor to performance mode..."
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo "To monitor CPU frequency: watch -n 1 vcgencmd measure_clock arm"

# Create start script with optimized parameters
echo "Creating startup script..."
cat > start.sh << EOF
#!/bin/bash
# Start the autonomous racer with optimized settings
source venv/bin/activate
# Run with headless mode for Raspberry Pi
python autonomous_racer.py --headless
EOF
chmod +x start.sh

# Print instructions
echo ""
echo "========== Installation Complete =========="
echo "To start the autonomous racer manually:"
echo "  ./start.sh"
echo ""
echo "To start as a service:"
echo "  sudo systemctl start autonomous-racer"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u autonomous-racer -f"
echo ""
echo "Access the web interface at:"
echo "  http://$(hostname).local:5000"
echo "  or"
echo "  http://$(hostname -I | awk '{print $1}'):5000"
echo "========================================"