# WiFi_Sensing_2.0
Detect malpractices conducted during the online examination using wifi sensing. 

# Hardware setup

## Windows

### Requirements

- ESP32 module
- Micro USB cable
- Arduino IDE
- Python 2.7 (preferred)

### Steps to test your ESP32 module's wifi connectivity
1. Connect your ESP32-C210x module with your system with microUSB cable.
2. Download [ESP32 drivers](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers) if not installed automatically.
3. Install [PYTHON2.7](https://www.python.org/download/releases/2.7/)
4. Install arduino IDE from microsoft store or [ARDUINO IDE](https://www.arduino.cc/en/software)
5. Now we have to install the package for ESP32. We use board manager for doing that. Go to **Arduino IDE > Preferences > Addtional Board manager URL** and paste this *https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json* . We can do it in different ways, go to [https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html#](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html#) for more information.
6. Go to **Tools > Manage Libraries** and install esp32 package by **expressi**. 
7. Change your board to port to proper COM (you can find the port details in device manager). Change your board to **ESP32 Arduino**.
8. Try uploading [blinking led](examples/led_light.h) or [connect wifi and fetch time](examples/wifiBasic.h).


