# Troubleshooting

## `No module named 'ultralytics'`

Run the program with the virtual environment Python:

```bash
/home/pi/aacvision-env/bin/python -u src/aacvision_stepwise_basketball_rover_v8_3_smooth_step_approach.py
```

## SSH disconnects when motors start

Check:

```bash
vcgencmd get_throttled
```

A nonzero result, especially `0x50005`, indicates a power problem. Correct the supply and wiring before continuing.

## Real basketball is not detected

1. Run the camera-only test.
2. Inspect `latest_basketball_view.jpg`.
3. Verify that YOLO recognizes the `sports ball` class.
4. Improve lighting and keep most of the ball inside the image.
5. Begin around 1.5-2 m from the camera.

## False positives

The main V8.3 version accepts a target using combined YOLO, orange-color, roundness, and depth evidence. Raising the confirmation score or orange fraction can reduce false positives, but excessive filtering may reject the real basketball.

## Rover turns but position does not update

Inspect wheel encoder inputs and polarity. Stepwise turning depends on odometry updates. Incorrect encoder counts, wheel diameter, track width, or polarity can cause turn timeouts or map distortion.

## Only one corner of the room appears on the map

Possible causes include limited travel coverage, map-size clipping, wheel slip, incomplete odometry, and the mission ending when the basketball is reached. The main V8.3 map is 8 m x 8 m by default.
