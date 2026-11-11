"""All drivers return BGR frames with actual stream metadata and metric depth.

`sdk_depth_unit_m` converts raw SDK depth to meters. A calibrated multiplicative
correction is kept separately and never silently applied by a driver.
"""
from typing import Protocol


class Camera(Protocol):
    metadata: dict

    def start(self) -> dict: ...
    def read(self) -> dict: ...
    def stop(self) -> None: ...
