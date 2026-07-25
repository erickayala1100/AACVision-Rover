# Test Procedure

## Preflight

1. Raise all four wheels.
2. Confirm secure power and USB connections.
3. Confirm the basketball is stationary.
4. Run `vcgencmd get_throttled` and require no current undervoltage.
5. Run `scripts/check_environment.sh`.
6. Keep immediate access to motor power and `Ctrl+C`.

## Camera-only test

```bash
/home/pi/aacvision-env/bin/python -u experiments/detection/camera_basketball_test.py
```

Confirm that the annotated image is written and that the basketball receives a valid depth measurement.

## Main wheels-up test

```bash
./scripts/run_v8_3.sh
```

Verify:

- Target on the camera left produces a left alignment command.
- Target on the camera right produces a right alignment command.
- Centered target produces a forward pulse.
- A target at or inside the stop distance results in a stop.
- `Ctrl+C` stops all motors.

## First floor test

Use a large, flat, uncluttered room. Start with the rover pointed away from a nearby wall. Add obstacles one at a time only after the basic mission works.

## Results to inspect

```bash
ls -lht /home/pi/aacvision_basketball_images/ | head
ls -lht /home/pi/aacvision_maps/ | head
```
