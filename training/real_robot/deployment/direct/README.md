# Direct Franka clients

Configure your own cameras with `--camera-serials`, in `--camera-names` order:

| Client | Default camera order |
| --- | --- |
| `openpi_robot_client.py` | `left_wrist external` |
| `uniwam_franka_direct_client.py` | `external left_wrist right_wrist` |

Camera serials must be supplied explicitly. The OpenPI client's `--image-files`
option accepts two RGB files instead, without requiring RealSense cameras.

For the OpenPI client, install the upstream `openpi_client` package in the active
Python environment. For a source checkout, pass its directory explicitly, for
example `--openpi-client-src ./openpi/packages/openpi-client/src` from the
repository root. Omitting this option uses the installed package.

Run either client with `--help` for policy-server, robot-interface, camera, and
execution options. Supply the host addresses for your own robot setup.
