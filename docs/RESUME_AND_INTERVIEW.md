# Resume and interview language

## Resume project

**DexRecover: Tactile-Aware Dexterous Hand Learning** | Python, MuJoCo, PyTorch, ROS2

- Built a seven-actuator, four-contact-finger MuJoCo manipulation system with keyboard demonstration capture, randomized cube/target placement, fixed-rate control, and guards for stale, malformed, non-finite, and out-of-limit commands.
- Collected 17,600 state-action transitions across 80 successful expert demonstrations and trained a normalized PyTorch behavior-cloning policy using an episode-level train/validation split.
- Achieved 50/50 successes on unseen randomized simulated trials using a 10 mm placement threshold, with 2.21 mm mean final error, 9.00 mm 95th-percentile error, zero safety events, and 0.203 ms mean CPU inference latency; added ROS2 simulator, policy, reset, command, observation, and diagnostics interfaces.
- Improved success from 51% with behavior cloning to 82% with two DAgger rounds on 100 harder trials with 2.5x pose randomization, mass/friction variation, observation noise and action delay; reduced mean final error from 14.43 mm to 6.35 mm.
- Integrated the official 16-DOF MuJoCo Menagerie Allegro Hand V3 and completed 100/100 randomized power-grasp closure trials with 0.0108 rad mean joint RMSE and two maintained hand-object contacts.
- Developed contact-force-based slip detection and bounded grasp recovery for randomized 0.20-0.45 N disturbances, improving recovery from 49/100 fixed-grasp trials to 82/100 while reducing average slip duration from 38.38 to 18.54 control steps.

## 30-second project explanation

I built DexRecover around the official 16-DOF Allegro Hand. It estimates contact count, normal force and object velocity from MuJoCo contact dynamics, detects slip during randomized lateral disturbances, and applies bounded additional closure. On 100 paired trials it improved recovery from 49 to 82 percent and cut slip duration by more than half. The same repository also compares behavior cloning with DAgger under pose, dynamics, sensor-noise and action-delay variation. DAgger improved transport success from 51 to 82 percent, which gave me a concrete example of why low offline loss is not enough for robot policies.

## Honest evidence boundary

The learned transport measurements use a simplified gripper. Allegro results cover power-grasp closure and contact-based disturbance recovery, not learned Allegro object transport. The tactile estimate comes from MuJoCo contacts rather than physical tactile sensors. ROS2 wrapper code is syntax-checked but was not executed on this machine because ROS2 is not installed. The project has not been tested on Tangent hardware and does not claim vision-based control or sim-to-real transfer.

## Strong technical discussion points

- Why episode-level splitting prevents adjacent states from the same trajectory leaking into validation.
- Why low supervised MSE did not guarantee closed-loop success and how rollout distribution shift appeared.
- Why final placement error, success threshold, inference latency, and safety-event count are more useful than training loss alone.
- How `/hand/observation`, `/hand/command`, `/hand/reset`, and `/hand/diagnostics` divide responsibilities between simulation and inference.
- What would come next: gamepad plus video synchronization, tactile/contact observations, a tendon-driven hand, domain randomization, and hardware calibration.
