# Limitations

- The learned transport policy uses a compact multi-contact gripper. The official 16-DOF Allegro model is integrated and benchmarked for power-grasp closure, but the learned policy has not yet been transferred to it.
- Reference training demonstrations come from a scripted expert. A keyboard recorder is included, but human demonstrations have not been used for the reported checkpoint.
- The policy receives normalized state and task phase; it is not a vision-only or foundation policy.
- Results are simulation evidence and must not be presented as physical robot performance.
- The controllers use position actuators. Torque control, tendon compliance and tactile sensing remain future work.
- Sim-to-real validation, calibration, communications timing and hardware emergency stops are outside the current evidence boundary.
