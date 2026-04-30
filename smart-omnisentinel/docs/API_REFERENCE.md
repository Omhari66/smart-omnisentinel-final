# SmartOmniSentinel — API Reference

Base URL: `http://localhost:8000/api/v1`  
Interactive docs: `http://localhost:8000/docs` (non-production only)  
Authentication: `Authorization: Bearer <access_token>`

---

## Authentication

### POST /auth/login
Get JWT access and refresh tokens.

**Request:**
```json
{ "email": "admin@sentinel.local", "password": "ChangeMe123!" }
```
**Response 200:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 900,
  "user": { "id": "uuid", "email": "...", "full_name": "...", "role": "ADMIN" }
}
```

### POST /auth/refresh
Exchange a refresh token for a new access token.

**Request:** `{ "refresh_token": "eyJ..." }`  
**Response 200:** Same shape as login response.

### GET /auth/me
Get current user profile. Requires valid access token.

### POST /auth/users
Create a new user. **Admin only.**

**Request:**
```json
{ "email": "guard@site.com", "password": "Secure123!", "full_name": "John Guard", "role": "GUARD" }
```

---

## Cameras

### GET /cameras
List all cameras. Optional filter: `?status=ACTIVE`

**Response 200:**
```json
{
  "items": [
    {
      "id": "uuid", "name": "North Gate", "stream_url": "rtsp://...",
      "location": "Building A", "status": "ACTIVE", "fps_target": 4,
      "last_seen_at": "2024-03-15T02:30:00Z",
      "zone": { "id": "uuid", "name": "entrance", "sensitivity": 0.75 },
      "threshold_profile": { "id": "uuid", "name": "default" }
    }
  ],
  "total": 3, "page": 1, "page_size": 25, "pages": 1
}
```

### POST /cameras
Register a new camera. **Admin only.**

**Request:**
```json
{
  "name": "South Entrance", "stream_url": "rtsp://192.168.1.21:554/stream",
  "location": "Building B South Door", "zone_id": null,
  "threshold_profile_id": null, "fps_target": 4
}
```

### GET /cameras/{camera_id}
Single camera detail.

### PUT /cameras/{camera_id}
Update camera config. **Admin only.** Send only fields to change.

### DELETE /cameras/{camera_id}
Deregister camera. Camera must be stopped first. **Admin only.**

### POST /cameras/{camera_id}/start
Start stream ingestion. **Supervisor+**

### POST /cameras/{camera_id}/stop
Stop stream ingestion. **Supervisor+**

### GET /cameras/{camera_id}/health
Live health metrics: status, last_seen_at, avg_fps.

---

## Incidents

### GET /incidents
Paginated incident list. All query params are optional.

**Query params:**
```
camera_id=uuid          Filter by camera
severity=HIGH,MEDIUM    Comma-separated severity tiers
status=EMERGENCY,ACTIVE_REVIEW
event_type=VIOLENT_INTERACTION,FALL_COLLAPSE
from=2024-03-01T00:00:00Z
to=2024-03-31T23:59:59Z
page=1
page_size=25
```

**Response 200:**
```json
{
  "items": [{
    "id": "uuid", "camera_id": "uuid",
    "event_type": "VIOLENT_INTERACTION",
    "severity": "HIGH", "status": "EMERGENCY",
    "risk_score": 78, "confidence_at_alert": 0.87,
    "people_count": 2, "event_start": "2024-03-15T02:33:45Z",
    "detected_at": "2024-03-15T02:34:11Z",
    "is_locked": true, "retention_days": 365
  }],
  "total": 42, "page": 1, "page_size": 25, "pages": 2
}
```

### GET /incidents/{incident_id}
Full detail including evidence clips and review history.

### GET /incidents/analytics/summary
Aggregated stats for dashboard. **Supervisor+**

**Query params:** `camera_id`, `from`, `to`

**Response 200:**
```json
{
  "total": 156,
  "by_severity": { "LOW": 89, "MEDIUM": 45, "HIGH": 22 },
  "by_event_type": { "VIOLENT_INTERACTION": 34, "FALL_COLLAPSE": 12, ... },
  "by_status": { "RESOLVED": 130, "PENDING_REVIEW": 20, ... },
  "false_positive_rate": 0.127,
  "period_from": "2024-03-01T00:00:00Z",
  "period_to": "2024-03-31T23:59:59Z"
}
```

---

## Review

### GET /review/queue
Ordered review queue. **Guard+**

**Response 200:** Array of `ReviewQueueItem`, priority-sorted.
```json
[{
  "incident_id": "uuid", "camera_name": "North Gate",
  "location": "Building A", "event_type": "VIOLENT_INTERACTION",
  "severity": "HIGH", "risk_score": 78,
  "detected_at": "2024-03-15T02:34:11Z",
  "escalation_deadline": "2024-03-15T02:39:11Z",
  "queue_position": 1
}]
```

### POST /review/{incident_id}/action
Submit a review decision. **Guard+**

**Request:**
```json
{
  "action": "CONFIRM",
  "notes": "Clearly two people fighting near stairwell",
  "confidence_override": 0.95
}
```

**Valid actions:** `CONFIRM` | `DISMISS` | `ESCALATE` | `FALSE_POSITIVE`

**Action effects:**

| Action | New Status | Locks Evidence |
|--------|-----------|---------------|
| CONFIRM | EMERGENCY | Yes |
| DISMISS | RESOLVED | No |
| ESCALATE | EMERGENCY | Yes |
| FALSE_POSITIVE | FALSE_POSITIVE | No |

### GET /review/history
All completed review actions. **Supervisor+**

---

## Evidence

### GET /evidence/{evidence_id}
Evidence clip metadata (not the video file itself).

**Response 200:**
```json
{
  "id": "uuid", "incident_id": "uuid",
  "file_size_bytes": 24576000, "duration_seconds": 58.4,
  "sha256_hash": "a3f8e2...", "is_locked": true,
  "is_anonymized": false, "retention_tag": "LONG_365D",
  "expires_at": null
}
```

### GET /evidence/{evidence_id}/url
Get a 15-minute signed URL for clip playback. **Supervisor+**  
Access is logged to the audit trail.

**Response 200:**
```json
{
  "evidence_id": "uuid",
  "signed_url": "http://host/api/v1/evidence/clips/uuid?token=abc&expires=1710462000",
  "expires_at": "2024-03-15T03:00:00Z",
  "sha256_hash": "a3f8e2..."
}
```

### GET /evidence/{evidence_id}/manifest
Full JSON provenance manifest. **Supervisor+**

### GET /evidence/clips/{evidence_id}?token=...&expires=...
Serve the actual video file. Only reachable via valid signed URL.  
Use the URL from `/evidence/{id}/url` — do not call this endpoint directly.

---

## Alerts

### GET /alerts
Alert history. Optional filters: `severity`, `incident_id`, `from`, `to`

### GET /alerts/{alert_id}
Alert detail.

### GET /alerts/escalations/history
All auto-escalation events (MEDIUM → HIGH timeout). **Supervisor+**

### POST /alerts/{alert_id}/escalate
Manual escalation with reason. **Supervisor+**

**Request:**
```json
{
  "reason": "Confirmed violent assault, requesting police",
  "notify_authorities": true
}
```

---

## Zones and Config

### GET /zones
List all zone configs.

### POST /zones
Create zone. **Admin only.**
```json
{
  "name": "parking_north", "sensitivity": 0.80,
  "suppress_event_types": [],
  "peak_hours_schedule": [{"start": "08:00", "end": "09:30", "days": [1,2,3,4,5]}]
}
```

### PUT /zones/{zone_id}
Update zone. **Admin only.** Send only fields to change.

### DELETE /zones/{zone_id}
Delete zone (cameras using it will have zone set to null). **Admin only.**

### GET /config/thresholds
List threshold profiles.

### POST /config/thresholds
Create threshold profile. **Admin only.**

### PUT /config/thresholds/{profile_id}
Update threshold profile. **Admin only.**

---

## Health

### GET /health
System health summary (disk, inference latency, camera counts). Requires auth.

### GET /health/ping
Unauthenticated liveness probe. Returns `{"status":"ok","ts":"..."}`.  
Used by load balancers and Docker healthcheck.

---

## WebSocket — Live Events

**Connection:** `ws://host/api/v1/ws/events?token=<access_token>`

JWT is passed as a query parameter (standard for WebSocket auth).  
On connect: server sends `CONNECTED` event with current state snapshot.  
Client must respond to `PING` events with `PONG` to maintain connection.

### Event Envelope
All events share this structure:
```json
{
  "event_id": "uuid",
  "event_type": "INCIDENT_CREATED",
  "timestamp": "2024-03-15T02:34:12Z",
  "payload": { ... }
}
```

### Event Types

**CONNECTED** — sent on connect:
```json
{
  "active_incidents": 2, "cameras_online": 3,
  "cameras_offline": 0, "pending_reviews": 5,
  "server_time": "2024-03-15T02:34:00Z"
}
```

**INCIDENT_CREATED** — new incident detected:
```json
{
  "incident_id": "uuid", "camera_id": "uuid", "camera_name": "North Gate",
  "location": "Building A", "event_type": "VIOLENT_INTERACTION",
  "severity": "HIGH", "risk_score": 78, "confidence": 0.87,
  "detected_at": "2024-03-15T02:34:11Z"
}
```

**ALERT_SEVERITY_CHANGED** — escalation or de-escalation:
```json
{
  "incident_id": "uuid", "previous_severity": "MEDIUM", "new_severity": "HIGH",
  "reason": "AUTO_ESCALATED_TIMEOUT", "changed_at": "2024-03-15T02:39:11Z"
}
```

**CAMERA_STATUS_CHANGED** — camera goes offline or recovers:
```json
{
  "camera_id": "uuid", "camera_name": "North Gate",
  "previous_status": "ACTIVE", "new_status": "OFFLINE",
  "reason": "STREAM_TIMEOUT", "changed_at": "2024-03-15T02:40:00Z"
}
```

**EVIDENCE_READY** — clip finished processing:
```json
{ "incident_id": "uuid", "evidence_id": "uuid", "duration_seconds": 58.4 }
```

**REVIEW_REQUESTED** — incident needs human review:
```json
{
  "incident_id": "uuid", "severity": "MEDIUM",
  "queue_position": 3, "escalation_deadline": "2024-03-15T02:39:11Z"
}
```

**INCIDENT_RESOLVED** — reviewer closed an incident:
```json
{
  "incident_id": "uuid", "resolution": "CONFIRM",
  "resolved_by": "guard@site.com", "resolved_at": "2024-03-15T02:41:00Z"
}
```

**SYSTEM_HEALTH_DEGRADED** — storage or service issue:
```json
{
  "component": "STORAGE", "metric": "disk_usage_pct",
  "value": 91.3, "threshold": 85, "severity": "WARNING"
}
```

**PING** — server keepalive (respond with PONG):
```json
{ "ts": "2024-03-15T02:34:30Z" }
```

---

## Error Responses

All errors follow this format:
```json
{ "detail": "Human-readable error message." }
```

| Status | When |
|--------|------|
| 400 | Malformed request body |
| 401 | Missing or invalid JWT |
| 403 | Insufficient role |
| 404 | Resource not found |
| 409 | Conflict (e.g. camera already active, incident already resolved) |
| 422 | Request validation failed (missing required fields) |
| 429 | Rate limit exceeded (Nginx) |
| 500 | Unexpected server error |
