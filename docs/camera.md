# Camera

## Canon EOS R7 USB Capture

The camera attached to the end-effector is a Canon EOS R7 with a Canon RF100mm F2.8 L Macro lens.

Canon's support pages list the EOS R7 as supporting the official `EDSDK` over USB, and the EOS R7 support page exposes `Linux ARM` as a supported OS family. This repo includes a still-capture CLI for that path:

```bash
pip install -e ".[hardware]"
openderm-canon-capture --output-dir captures --count 1
openderm-canon-capture --output-dir captures --count 5 --interval-s 1.5
openderm-canon-capture --edsdk-lib /opt/canon/edsdk/libEDSDK.so
```

Notes:

- Download the Canon Linux ARM EDSDK separately from Canon and install libEDSDK.so on the Pi.
- The code sets the camera save target to the host and downloads the image when Canon raises the transfer event.
- A capture timeout usually means the camera is not in a remote-control-ready state over USB or the EDSDK library is missing.
