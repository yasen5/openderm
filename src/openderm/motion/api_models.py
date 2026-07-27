"""Strict request models shared by the motion HTTP APIs."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator


FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0),
]
PositiveFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, gt=0),
]
CommanderId = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=128,
    ),
]


class StrictRequest(BaseModel):
    """Base model that rejects fields not declared by the endpoint."""

    model_config = ConfigDict(extra="forbid")


class GantryMoveRequest(StrictRequest):
    x: FiniteFloat
    feed_mm_s: Optional[PositiveFiniteFloat] = None
    tolerance_mm: PositiveFiniteFloat = 0.05
    commander_id: Optional[CommanderId] = None
    blocking: StrictBool = False


class GantryHomeRequest(StrictRequest):
    axes: tuple[Literal["x"], ...] = Field(default=("x",), min_length=1, max_length=1)
    blocking: StrictBool = False


class GantryStopRequest(StrictRequest):
    mode: Literal["soft", "emergency"] = "emergency"


class GantryStreamStartRequest(StrictRequest):
    feed_mm_s: Optional[PositiveFiniteFloat] = None
    feed_mm_min: Optional[PositiveFiniteFloat] = None
    tick_s: PositiveFiniteFloat = 0.05
    min_step_mm: NonNegativeFiniteFloat = 0.01

    @model_validator(mode="after")
    def require_one_feed_unit(self) -> "GantryStreamStartRequest":
        if self.feed_mm_s is not None and self.feed_mm_min is not None:
            raise ValueError("Specify only one of feed_mm_s or feed_mm_min.")
        return self

    @property
    def resolved_feed_mm_s(self) -> float | None:
        if self.feed_mm_s is not None:
            return self.feed_mm_s
        if self.feed_mm_min is not None:
            return self.feed_mm_min / 60.0
        return None


class GantryStreamTargetRequest(StrictRequest):
    x: FiniteFloat


class RxMoveToRequest(StrictRequest):
    position_rad: FiniteFloat
    speed_rad_s: Optional[PositiveFiniteFloat] = None
    accel_rad_s2: Optional[PositiveFiniteFloat] = None


class RxVelocityRequest(StrictRequest):
    velocity_rad_s: FiniteFloat
