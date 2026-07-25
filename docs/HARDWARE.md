# Hardware

## Final platform

| Component | Final selection | Notes |
|---|---|---|
| Onboard computer | Raspberry Pi 4 | Runs Python control, perception, and mapping |
| RGB-D camera | Intel RealSense D435 | Color image and depth feedback |
| LiDAR | Slamtec RPLIDAR C1 | 360-degree obstacle sensing and map input |
| Drive motors | 4 x JGB37-520, 12 V, 178 RPM | Hall encoder feedback |
| Wheels | 4 x 80 mm mecanum wheels | Operated primarily as skid-steer/differential motion |
| Motor drivers | 2 x DROK/L298 dual H-bridge boards | PWM speed and direction control |
| Battery | 12 V, 5200 mAh | Motor and system power source |
| Regulation | Adjustable buck converter | Regulated logic/sensor supply |
| Chassis | Aluminum, about 8 in x 12 in | Approximate total rover weight: 6.7 lb |

## GPIO assignments

The authoritative pin mapping is defined near the top of the main Python file. Review it before connecting or changing hardware.

## Power architecture

The Raspberry Pi and motor electronics must receive stable power. Motor current transients previously caused undervoltage and SSH disconnections. The preferred arrangement is:

- A stable regulated supply for the Raspberry Pi and USB sensors
- A motor supply capable of handling stall-current transients
- A shared electrical ground between the Pi-side logic and motor drivers
- Short, secure power connections
- Separation between high-current motor wiring and USB/sensor cables

Check power status with:

```bash
vcgencmd get_throttled
```

`0x50005` indicates current and historical undervoltage/throttling and is unsafe for autonomous testing.
