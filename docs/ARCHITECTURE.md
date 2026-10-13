# Architecture

```mermaid
flowchart LR
    A[Scripted expert / teleoperator] --> B[Command safety monitor]
    B --> C[MuJoCo four-finger hand]
    C --> D[State and contact observations]
    D --> E[Demonstration recorder]
    E --> F[PyTorch behavior cloning]
    F --> I[DAgger rollout and relabeling]
    I --> F
    F --> G[Closed-loop robust evaluation]
    G --> H[Metrics and failure report]
    J[Official 16-DOF Allegro Hand] --> K[Randomized power-grasp benchmark]
    J --> L[Contact force and object motion estimate]
    L --> M[Slip detector]
    M --> N[Bounded grasp recovery]
    N --> J
```

The executable reference uses a compact MuJoCo hand with Cartesian palm motion and four independently actuated fingers. The small model keeps the entire demonstration-to-policy loop reproducible on a CPU. Optional ROS2 simulator and inference nodes wrap the same observation, command, reset and diagnostics interfaces.
