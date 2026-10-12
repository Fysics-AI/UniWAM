# Policy WebSocket interface

The evaluator sends MessagePack-encoded requests over WebSocket:

```python
{
    "batch": [
        {
            "image": uint8_array_H_W_3,
            "task": str,
            "navigation_state": float32_array_2,
            "diffusion_seed": int,
            "return_debug": True,
        },
        ...,
    ]
}
```

The server must return a list of equal length:

```python
{
    "batch": [
        {
            "relative_se2_actions": float32_array_32_3,
            "navigation_state_after": float32_array_2,
            # Optional diagnostic fields:
            "raw_velocity_actions": float32_array_32_2,
            "actions": float32_array_32_9,
            "policy_timing": {},
        },
        ...,
    ]
}
```

Each relative action is `[x_body_m, y_body_m, heading_rad]`. A model stop is
detected from consecutive near-zero action chunks according to the evaluator CLI
thresholds. The bundled `openpi_client` module defines the MessagePack transport;
an external model may implement the corresponding server in any framework.
