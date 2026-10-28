# FastAPI Service Guide

The API is the trusted boundary between the packet collector and the dashboard.
It validates telemetry, applies the versioned NSL-KDD pipeline, persists an
auditable decision, and creates severity-labelled alerts for threats.

## Start

```powershell
uvicorn backend_api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`
when `CTD_ENABLE_DOCS=true`.

## Endpoints

| Method | Route | Notes |
| --- | --- | --- |
| `GET` | `/health` | Service readiness and `model_loaded` state |
| `GET` | `/model/status` | Model version, class labels, saved hold-out metrics |
| `POST` | `/analyze` | Validate, classify, persist, and possibly create an alert |
| `GET` | `/logs?limit=100&offset=0&threats_only=false` | Persistent packet decisions |
| `GET` | `/alerts?severity=high&status=open` | Filterable security alerts |
| `GET` | `/statistics` | Counts, severity mix, protocol trend, suspicious IPs |

All routes have versioned counterparts under `/api/v1/`; the collector uses
`/api/v1/packets/analyze` for its POST request.

## Packet/flow contract

Required analysis fields are `source_ip`, `destination_ip`, `protocol`, and
`packet_size`. Optional fields include ports, TTL, TCP flags, flow counts and
rates, private-address flags, capture interface, and a list of transparent
anomaly indicators. IP addresses, ports, packet length, timestamps, and the
indicator list are validated by Pydantic before any model call or database
write.

Example JSON:

```json
{
  "source_ip": "192.0.2.10",
  "destination_ip": "198.51.100.20",
  "protocol": "TCP",
  "source_port": 49152,
  "destination_port": 443,
  "packet_size": 512,
  "tcp_flags": "S",
  "flow_packet_count": 1,
  "flow_byte_count": 512,
  "flow_duration_seconds": 0.0,
  "packets_per_second": 1.0,
  "bytes_per_second": 512.0,
  "anomaly_indicators": ["tcp_syn_without_ack"]
}
```

## Model availability is explicit

The service expects `ai_model/artifacts/intrusion_detection_model.joblib` by
default. Configure a different artifact with `CTD_MODEL_PATH`. A bare legacy
pickle is rejected because it cannot prove its feature ordering or
preprocessing. If no valid artifact has been trained, `/health` reports
`model_loaded: false` and `/analyze` records the event with a
`model_unavailable` result; it never invents a size-based threat heuristic.

## Persistence and retention

`CTD_DATABASE_PATH` defaults to `backend_api/data/traffic.db`. SQLite is
appropriate for a local demo and gives the dashboard historical data across
restarts. `CTD_MAX_LOG_RECORDS` bounds retained packet and linked-alert rows to
prevent unbounded local growth. The database is ignored by Git. For a shared
or high-volume deployment, use authentication, TLS, a queue, and a production
database.

## API configuration

| Variable | Default | Description |
| --- | --- | --- |
| `CTD_MODEL_PATH` | `ai_model/artifacts/intrusion_detection_model.joblib` | Versioned inference bundle |
| `CTD_DATABASE_PATH` | `backend_api/data/traffic.db` | SQLite file path |
| `CTD_MAX_LOG_RECORDS` | `1000` | Maximum persistent history |
| `CTD_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed UI origins |
| `CTD_LOG_LEVEL` | `INFO` | Service logging level |
| `CTD_ENABLE_DOCS` | `true` | Expose or hide `/docs` and `/redoc` |

The `.env` file is loaded automatically when present at the repository root.
