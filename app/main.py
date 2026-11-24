"""FastAPI application exposing the review-ranking service."""

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from .audit import AuditConflict, AuditError, AuditEvaluation, AuditIncomplete
from .calibration import CalibrationMissing
from .models import (
    AuditExtremum,
    AuditOrderCreate,
    AuditOrderSummary,
    AuditResult,
    AuditScoreBucket,
    InspectionSummary,
    InspectionUpdate,
    PanelBatch,
    PendingAsDefectRow,
    QueueItem,
    QueueResponse,
    RegistrationSummary,
)
from .store import MAX_QUEUE, NotFound, PanelStore, ValidationError

description = """
Rank glass panels for re-inspection by expected missed-defect loss.

* Defect rates are fitted on inspected panels with a non-decreasing
  (isotonic) curve; panels sharing a defect score share one rate.
* Scores with no inspections borrow the nearest inspected score on the left,
  or the lowest inspected score when there is none.
* The queue is the top K pending panels by `defect_rate * loss`, ties broken by
  the smaller panel id.  Both inspection updates and queue selection avoid
  scanning the pending population.
* Audit orders (`/audits`) bound the batch defect count; querying one returns
  the minimum and maximum ROC AUC over every label assignment compatible with
  the recorded inspections and the interval, plus the per-score pending
  allocations that attain each extreme.
"""

app = FastAPI(title="Panel Re-inspection Ranking", version="1.0.0", description=description)
store = PanelStore()


@app.exception_handler(CalibrationMissing)
async def _missing_calibration(_request: Request, exc: CalibrationMissing) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": "calibration missing: no inspected panels yet; submit at "
            "least one inspection result before requesting a queue",
            "code": "calibration_missing",
        },
    )


@app.exception_handler(NotFound)
async def _not_found(_request: Request, exc: NotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc), "code": "not_found"})


@app.exception_handler(ValidationError)
async def _invalid(_request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"detail": str(exc), "code": "invalid_request"}
    )


@app.exception_handler(AuditConflict)
async def _audit_conflict(_request: Request, exc: AuditConflict) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc), "code": "audit_conflict"},
    )


@app.exception_handler(AuditIncomplete)
async def _audit_incomplete(_request: Request, exc: AuditIncomplete) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc), "code": "audit_incomplete"},
    )


@app.exception_handler(AuditError)
async def _audit_error(_request: Request, exc: AuditError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc), "code": "audit_unavailable"},
    )


@app.get("/health")
def health() -> dict:
    snapshot = store.counts_snapshot()
    order = store.audit_order
    snapshot["audit_order_id"] = order.order_id if order else None
    return {"status": "ok", **snapshot}


@app.post("/panels", response_model=RegistrationSummary, status_code=201)
def register_panels(batch: PanelBatch) -> RegistrationSummary:
    """Register a batch of panels (optionally with an inspection result)."""
    entries = [
        (p.panel_id, p.score, p.loss, p.inspection_result) for p in batch.panels
    ]
    count = store.register(entries)
    return RegistrationSummary(
        registered=count,
        total_panels=store.total_panels,
        inspected_panels=store.inspected_panels,
    )


@app.put("/panels/{panel_id}/inspection", response_model=InspectionSummary)
def set_inspection(panel_id: int, update: InspectionUpdate) -> InspectionSummary:
    """Fill in or correct the inspection result of one panel.

    Only inspected panels feed the calibration, so this re-fits it.  An
    invalid request is rejected before any state changes, leaving the previous
    calibration and queue intact.
    """
    store.set_inspection(panel_id, update.inspection_result)
    return InspectionSummary(
        panel_id=panel_id,
        inspection_result=update.inspection_result,
        calibration_version=store.calibration_version,
        inspected_panels=store.inspected_panels,
        pending_panels=store.pending_panels,
    )


@app.get("/queue", response_model=QueueResponse)
def queue(k: int = Query(default=10, ge=1, le=MAX_QUEUE)) -> QueueResponse:
    """Top K pending panels by expected missed-defect loss."""
    snapshot = store.queue_snapshot(k)
    return QueueResponse(
        k=k,
        calibration_version=snapshot["calibration_version"],
        inspected_panels=snapshot["inspected_panels"],
        pending_panels=snapshot["pending_panels"],
        items=[
            QueueItem(
                panel_id=pid,
                score=score,
                defect_rate=float(rate),
                expected_loss=float(expected),
            )
            for pid, score, rate, expected in snapshot["items"]
        ],
    )


@app.get("/calibration")
def calibration() -> dict:
    """Fitted calibration, for auditing the queue."""
    fitted = store.require_calibration()
    return {
        "calibration_version": fitted.version,
        "inspected_panels": fitted.inspected_panels,
        "lowest_inspected_score": fitted.lowest_inspected_score,
        "rates": [
            {
                "score": score,
                "defect_rate": float(rate),
                "source": fitted.effective_sources[score],
            }
            for score, rate in sorted(fitted.fitted_rates.items())
        ],
    }


@app.get("/panels/{panel_id}")
def panel(panel_id: int) -> dict:
    item = store.get_panel(panel_id)
    return {
        "panel_id": item.panel_id,
        "score": item.score,
        "loss": item.loss,
        "inspection_result": item.inspected if item.is_inspected else None,
        "inspected": item.is_inspected,
    }


# ------------------------------------------------------------------ audits
def _render_audit(evaluation: "AuditEvaluation") -> AuditResult:
    return AuditResult(
        order=AuditOrderSummary(
            order_id=evaluation.order.order_id,
            min_defects=evaluation.order.min_defects,
            max_defects=evaluation.order.max_defects,
            created_at=evaluation.order.created_at,
            panels_at_creation=evaluation.order.panels_at_creation,
            inspected_at_creation=evaluation.order.inspected_at_creation,
            pending_at_creation=evaluation.order.pending_at_creation,
        ),
        total_panels=evaluation.total_panels,
        inspected_panels=evaluation.inspected_panels,
        pending_panels=evaluation.pending_panels,
        known_defects=evaluation.known_defects,
        feasible_defect_range=list(evaluation.feasible_defect_range),
        minimum=AuditExtremum(
            auc=float(evaluation.minimum.auc),
            total_defects=evaluation.minimum.total_defects,
            pending_as_defect=[
                PendingAsDefectRow(
                    score=row.score, pending=row.pending, as_defect=row.as_defect
                )
                for row in evaluation.minimum.pending_as_defect
            ],
        ),
        maximum=AuditExtremum(
            auc=float(evaluation.maximum.auc),
            total_defects=evaluation.maximum.total_defects,
            pending_as_defect=[
                PendingAsDefectRow(
                    score=row.score, pending=row.pending, as_defect=row.as_defect
                )
                for row in evaluation.maximum.pending_as_defect
            ],
        ),
        score_buckets=[
            AuditScoreBucket(
                score=bucket.score,
                total=bucket.total,
                inspected_defects=bucket.inspected_defects,
                inspected_good=bucket.inspected_good,
                pending=bucket.pending,
            )
            for bucket in evaluation.score_buckets
        ],
    )


@app.post("/audits", response_model=AuditOrderSummary, status_code=201)
def create_audit(order_create: AuditOrderCreate) -> AuditOrderSummary:
    """Create an audit order from the registered panels and defect interval.

    Replaces the current order.  The interval is validated against the
    registered population; known-inspection conflicts surface when the order
    is queried, so corrections can restore a previously failing order.
    """
    if store.total_panels == 0:
        raise ValidationError("register panels before creating an audit order")
    if order_create.min_defects > store.total_panels:
        raise ValidationError(
            f"min_defects ({order_create.min_defects}) exceeds the number of "
            f"registered panels ({store.total_panels})"
        )
    if order_create.max_defects > store.total_panels:
        raise ValidationError(
            f"max_defects ({order_create.max_defects}) exceeds the number of "
            f"registered panels ({store.total_panels})"
        )
    order = store.create_audit_order(
        order_create.min_defects, order_create.max_defects
    )
    return AuditOrderSummary(
        order_id=order.order_id,
        min_defects=order.min_defects,
        max_defects=order.max_defects,
        created_at=order.created_at,
        panels_at_creation=order.panels_at_creation,
        inspected_at_creation=order.inspected_at_creation,
        pending_at_creation=order.pending_at_creation,
    )


@app.get("/audits/current", response_model=AuditResult)
def current_audit() -> AuditResult:
    """Evaluate the current audit order against the latest inspections.

    Returns 409 ``audit_conflict`` when recorded defects contradict the
    interval or 409 ``audit_incomplete`` when no feasible assignment has both
    classes; the order itself is retained and becomes queryable again after a
    correction.
    """
    return _render_audit(store.audit_snapshot())


@app.get("/audits/{order_id}", response_model=AuditResult)
def get_audit(order_id: str) -> AuditResult:
    """Query a specific order; only the current order is retained."""
    return _render_audit(store.audit_snapshot(order_id))
