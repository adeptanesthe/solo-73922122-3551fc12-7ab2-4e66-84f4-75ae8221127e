"""Request/response schemas for the review-ranking API."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_PANELS = 50_000
MAX_SCORE = 255


class PanelRegistration(BaseModel):
    """A panel to register.

    ``inspection_result`` is the optional actual inspection outcome (0/1).
    Boards without it are pending and can later be filled in by panel id.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    panel_id: int
    score: int = Field(ge=0, le=MAX_SCORE)
    loss: int = Field(gt=0)
    inspection_result: Optional[int] = Field(default=None, ge=0, le=1)


class PanelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    panels: List[PanelRegistration] = Field(min_length=1, max_length=MAX_PANELS)


class InspectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    inspection_result: int = Field(ge=0, le=1)


class RegistrationSummary(BaseModel):
    registered: int
    total_panels: int
    inspected_panels: int


class InspectionSummary(BaseModel):
    panel_id: int
    inspection_result: int
    calibration_version: int
    inspected_panels: int
    pending_panels: int


class QueueItem(BaseModel):
    panel_id: int
    score: int
    defect_rate: float
    expected_loss: float


class QueueResponse(BaseModel):
    k: int
    calibration_version: int
    inspected_panels: int
    pending_panels: int
    items: List[QueueItem]


# ------------------------------------------------------------------ audits
class AuditOrderCreate(BaseModel):
    """Batch-wide defect interval reported downstream."""

    model_config = ConfigDict(extra="forbid", strict=True)

    min_defects: int = Field(ge=0)
    max_defects: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_interval(self) -> "AuditOrderCreate":
        if self.min_defects > self.max_defects:
            raise ValueError("min_defects must not exceed max_defects")
        return self


class AuditOrderSummary(BaseModel):
    order_id: str
    min_defects: int
    max_defects: int
    created_at: str
    panels_at_creation: int
    inspected_at_creation: int
    pending_at_creation: int


class PendingAsDefectRow(BaseModel):
    score: int
    pending: int
    as_defect: int


class AuditExtremum(BaseModel):
    auc: float
    total_defects: int
    pending_as_defect: List[PendingAsDefectRow]


class AuditScoreBucket(BaseModel):
    score: int
    total: int
    inspected_defects: int
    inspected_good: int
    pending: int


class AuditResult(BaseModel):
    order: AuditOrderSummary
    total_panels: int
    inspected_panels: int
    pending_panels: int
    known_defects: int
    feasible_defect_range: List[int]
    minimum: AuditExtremum
    maximum: AuditExtremum
    score_buckets: List[AuditScoreBucket]
