# Version and Trial History

The repository preserves representative experiments that explain why V8.3 became the main working release.

| Version | Main idea | Observed result |
|---|---|---|
| Camera test | RealSense/YOLO and depth only | Useful for isolating perception from navigation |
| V8 fast approach | Longer/faster approach pulses | Faster, but increased target-loss and overshoot risk |
| V8 continuous adjust | Continuous target steering | Did not spot or hold the target as reliably |
| V8.1 robust detection | More permissive orange/YOLO fusion | Improved basketball acquisition |
| V8.2 reacquire tracking | Lock and local reacquisition | Reacquisition was not smooth enough in trials |
| **V8.3 smooth step approach** | Small verified approach steps | Best overall balance; selected as main release |
| V8.4 false-positive guard | Strict temporal/shape/depth filters | Reduced false positives but could reject a real basketball |
| V9 continuous approach | Continuous motion toward target | Lost the target more easily than stepwise approach |
| V10 frontier exploration | Frontier selection and A* planning | More complete exploration concept, but greater complexity and sensitivity to odometry |
| V8.10 map-aware corners | Use SLAM unknown space for corner choices | Improved some corner choices but was not selected over V8.3 |
| Yellow-target references | Reactive wall-escape navigation | Inspired obstacle-navigation trials, but used a different target detector |

Experimental files are retained for engineering traceability. They are not all recommended for floor operation.
