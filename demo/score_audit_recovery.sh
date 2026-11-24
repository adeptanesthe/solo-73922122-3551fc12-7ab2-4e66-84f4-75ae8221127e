#!/usr/bin/env bash
# Demonstration: score-audit orders report the min/max ROC AUC over every label
# assignment compatible with the downstream defect-count interval, and an
# inspection correction that first breaks (then restores) the audit.
#
# Start the service first:
#   docker compose up --build -d
# Then run against a *fresh* instance (the panel ids below must be new):
#   ./demo/score_audit_recovery.sh
#
# Requires curl and jq.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

say() { printf '\n=== %s ===\n' "$1"; }
post() { curl -sS -X POST "$BASE_URL$1" -H 'Content-Type: application/json' -d "$2"; }
put()  { curl -sS -X PUT  "$BASE_URL$1" -H 'Content-Type: application/json' -d "$2"; }
get()  { curl -sS "$BASE_URL$1"; }

say "1. Register panels: score 10 (one known good + four pending), score 90 (one known defect + four pending)"
post /panels '{"panels":[
  {"panel_id": 1,  "score": 10, "loss": 1, "inspection_result": 0},
  {"panel_id": 2,  "score": 90, "loss": 1, "inspection_result": 1},
  {"panel_id": 100, "score": 10, "loss": 1},
  {"panel_id": 101, "score": 10, "loss": 1},
  {"panel_id": 102, "score": 10, "loss": 1},
  {"panel_id": 103, "score": 10, "loss": 1},
  {"panel_id": 200, "score": 90, "loss": 1},
  {"panel_id": 201, "score": 90, "loss": 1},
  {"panel_id": 202, "score": 90, "loss": 1},
  {"panel_id": 203, "score": 90, "loss": 1}
]}' | jq -c .

say "2. Create audit order with downstream defect interval [1, 3] (replaces any current order)"
ORDER=$(post /audits '{"min_defects": 1, "max_defects": 3}')
echo "$ORDER" | jq -c .
ORDER_ID=$(echo "$ORDER" | jq -r '.order_id')

say "3. Query the audit: min and max AUC, plus per-score pending-as-defect allocations"
get "/audits/$ORDER_ID" | jq -c '{
  known_defects,
  feasible_defect_range,
  minimum:  {auc: .minimum.auc, total_defects: .minimum.total_defects,
             pending_as_defect: .minimum.pending_as_defect},
  maximum:  {auc: .maximum.auc, total_defects: .maximum.total_defects,
             pending_as_defect: .maximum.pending_as_defect}
}'

say "4. A second order replaces the current one; the old id is no longer queryable"
post /audits '{"min_defects": 1, "max_defects": 2}' | jq -c '{order_id, min_defects, max_defects}'
HTTP=$(curl -sS -o /tmp/demo_old_order.json -w '%{http_code}' "$BASE_URL/audits/$ORDER_ID")
echo "old order now: HTTP $HTTP $(jq -c . /tmp/demo_old_order.json)"

say "5. Recreate an order on [1, 2], then confirm two more defects contradict it"
CURRENT=$(post /audits '{"min_defects": 1, "max_defects": 2}')
CURRENT_ID=$(echo "$CURRENT" | jq -r '.order_id')
put /panels/100/inspection '{"inspection_result": 1}' >/dev/null
put /panels/200/inspection '{"inspection_result": 1}' >/dev/null
HTTP=$(curl -sS -o /tmp/demo_conflict.json -w '%{http_code}' "$BASE_URL/audits/current")
echo "HTTP $HTTP: $(jq -c . /tmp/demo_conflict.json)"
[ "$HTTP" = "409" ] && [ "$(jq -r '.code' /tmp/demo_conflict.json)" = "audit_conflict" ] \
  && echo "conflict reported, order retained: OK"

say "6. Correct one inspection back to good (known defects 3 -> 2); the same order is queryable again"
put /panels/200/inspection '{"inspection_result": 0}' | jq -c '{panel_id: .panel_id, inspected_panels}'
RECOVERED=$(get "/audits/$CURRENT_ID")
echo "$RECOVERED" | jq -c '{
  known_defects,
  feasible_defect_range,
  minimum: {auc: .minimum.auc, total_defects: .minimum.total_defects},
  maximum: {auc: .maximum.auc, total_defects: .maximum.total_defects}
}'
[ "$(echo "$RECOVERED" | jq -r '.order.order_id')" = "$CURRENT_ID" ] \
  && echo "same order id recovered: OK"

say "7. Health shows the current order id"
get /health | jq -c .
