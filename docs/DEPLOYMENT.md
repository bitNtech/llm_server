# Deployment

The server is one stateless container image (`python -m app`) that needs:

- a Google Cloud project with Vertex AI,
- the Firestore database `llm_server`,
- three secrets.

Recommended target: **Cloud Run in asia-south1**, next to the Vertex endpoint. A VM with systemd works the same way.

## 1. Google Cloud setup (once per project)

```bash
PROJECT=aica-voice-pilot
REGION=asia-south1
gcloud config set project $PROJECT

# APIs
gcloud services enable aiplatform.googleapis.com firestore.googleapis.com run.googleapis.com \
    secretmanager.googleapis.com artifactregistry.googleapis.com

# Firestore: a named database in native mode, in the same region
gcloud firestore databases create --database=llm_server --location=$REGION --type=firestore-native

# Optional: expire idempotency records automatically (they carry expires_at; 24 h)
gcloud firestore fields ttls update expires_at --collection-group=idempotency --enable-ttl --database=llm_server

# Service account used by the server
gcloud iam service-accounts create aica-llm-server
SA=aica-llm-server@$PROJECT.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$SA --role=roles/aiplatform.user
gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$SA --role=roles/datastore.user
```

No composite Firestore indexes are required.

**Model access.** The default `google/gemini-3.5-flash` (asia-south1 and global) and the fallback `google/gemini-3.1-flash-lite` (global) were verified in `aica-voice-pilot`. In another project, check access first:

```bash
TOKEN=$(gcloud auth print-access-token)
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  "https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$REGION/endpoints/openapi/chat/completions" \
  -d '{"model":"google/gemini-3.5-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

## 2. Secrets

```bash
python3 -c 'import secrets; print("aica_admin_" + secrets.token_urlsafe(32))' | gcloud secrets create aica-admin-api-keys --data-file=-
python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' | gcloud secrets create aica-secrets-encryption-key --data-file=-
python3 -c 'import secrets; print(secrets.token_urlsafe(48))' | gcloud secrets create aica-realtime-token-secret --data-file=-
for s in aica-admin-api-keys aica-secrets-encryption-key aica-realtime-token-secret; do
  gcloud secrets add-iam-policy-binding $s --member=serviceAccount:$SA --role=roles/secretmanager.secretAccessor
done
```

| Secret | Purpose | Rotation |
|---|---|---|
| `ADMIN_API_KEYS` | Platform operators. Comma-separated, at least 24 characters each. | Add the new key, roll out, remove the old one. |
| `SECRETS_ENCRYPTION_KEY` | Encrypts tenant secrets at rest. **Losing it makes stored tool and MCP credentials unreadable.** | Prepend a new key (`new,old`) and roll out. Old data still decrypts, and updates re-encrypt with the newest key. |
| `REALTIME_TOKEN_SECRET` | Signs realtime tokens. | Rotating it invalidates outstanding tokens; they are short-lived. |

## 3. Build and deploy to Cloud Run

```bash
gcloud artifacts repositories create aica --repository-format=docker --location=$REGION
IMAGE=$REGION-docker.pkg.dev/$PROJECT/aica/aica-llm-server:$(git rev-parse --short HEAD)
gcloud builds submit --tag $IMAGE .

# edit deploy/cloudrun/service.yaml: PROJECT_ID, image
gcloud run services replace deploy/cloudrun/service.yaml --region=$REGION
```

Important Cloud Run settings, already in `deploy/cloudrun/service.yaml`:

| Setting | Why |
|---|---|
| `minScale: 1` | The first caller never meets a cold start. |
| `cpu-throttling: false` | Background work (token refresh, keep-warm, leases, usage flush, executions) needs CPU between requests. |
| `sessionAffinity: true` | REST calls of one session reach the instance that holds it. WebSockets are sticky anyway. |
| `timeoutSeconds: 3600` | A WebSocket call can last up to an hour; reconnect with `?session_id=` to continue. |
| `SHUTDOWN_GRACE_S=8` | Cloud Run allows 10 s after SIGTERM. In-flight answers finish, WebSockets receive `session.closing`, writes are flushed and leases released. |
| startup probe on `/health/ready` | Traffic arrives only after the Vertex probe and the Firestore check pass. |

## 4. First tenant

```bash
ADMIN=$(gcloud secrets versions access latest --secret=aica-admin-api-keys)
curl -s -X POST https://<service-url>/v1/tenants -H "Authorization: Bearer $ADMIN" \
     -H 'Content-Type: application/json' -d '{"name": "AICA Production"}'
# → returns the tenant, its first owner API key (aica_...) and the webhook secret (shown once)
```

Then, with the owner key: create `runtime` keys for voice workers, register tools, MCP servers and agents, and mint realtime tokens for browser clients.

## 5. VM deployment (alternative)

```bash
sudo useradd --system --home /opt/aica-llm-server aica
sudo git clone <repo> /opt/aica-llm-server && cd /opt/aica-llm-server
sudo -u aica ./run.sh install
sudo install -d -m 750 -o aica /etc/aica-llm-server
sudo install -m 600 -o aica .env.example /etc/aica-llm-server/server.env    # fill in values and secrets
sudo install -m 600 -o aica <key.json> /etc/aica-llm-server/service-account.json
sudo cp deploy/systemd/aica-llm-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now aica-llm-server
```

Put a TLS-terminating proxy or load balancer in front. It must allow WebSocket upgrades, use an idle timeout of at least 15 minutes, and route each session to one instance. Set `FORWARDED_ALLOW_IPS` to the proxy's address.

## 6. Local development

```bash
gcloud auth application-default login      # Vertex access with your own account
cp .env.example .env                        # set GCP_PROJECT_ID and ADMIN_API_KEYS
./run.sh dev                                # Firestore emulator + server on :8000 (real Vertex)
./run.sh test                               # unit + integration tests (emulator + fake Vertex, no cloud calls)
./run.sh test-live                          # live tests against real Vertex (a few cents)
```

To use the real Firestore database from a laptop, unset `FIRESTORE_EMULATOR_HOST`. Credentials whose project has the Firestore API enabled are required.

## 7. Operations

**Health.** Watch these:

- `/health/ready` for routing,
- `/health` for detail: route health per Vertex location, circuit state, Firestore, MCP connections, write backlog,
- `/metrics` for Prometheus.

**Alerts worth having:**

| Condition | Metric |
|---|---|
| Vertex TTFT degraded | `llm_upstream_ttft_seconds` p90 > 2 s |
| Failover is frequent | `rate(vertex_failovers_total[5m]) > 0` for a sustained period |
| A location's circuit is open | `vertex_circuit_open == 1` |
| Capacity is too low | `llm_admission_rejections_total` increasing (raise `MAX_CONCURRENT_GENERATIONS`, quota or instances) |
| Firestore is slow or down | `firestore_write_backlog` > 1000 or `firestore_errors_total` increasing |
| Webhook receivers failing | `webhook_deliveries_total{outcome="failed"}` |

**Capacity.**

- One instance handles hundreds of concurrent calls: the work is I/O and the server adds about 0.2 ms per turn. The limits are Vertex quota (tokens per minute per region) and `MAX_CONCURRENT_GENERATIONS`.
- 429 `provider_overloaded` means Vertex quota is exhausted on every route. Request more quota for the model in asia-south1.

**Upgrades.** Roll out new instances. Draining instances finish in-flight answers and tell WebSocket clients to reconnect (`session.closing`). Sessions resume on another instance once the old one releases them (immediately on graceful shutdown, or after `SESSION_LEASE_S` after a crash).

**Troubleshooting.**

| Symptom | Cause / fix |
|---|---|
| `/health/ready` 503 `vertex: …403…` | The service account lacks `roles/aiplatform.user`, or the model is not enabled for the project or region. |
| `/health/ready` 503 `firestore: …` | Firestore API disabled, database `llm_server` missing, or `roles/datastore.user` missing. |
| `model_not_found` | The model is not available at any configured location. Check `GET /v1/providers` → `unavailable_routes`. |
| `session_busy_elsewhere` | Another instance holds the session. Enable session affinity, or retry after `SESSION_LEASE_S`. |
| `secret_decrypt_failed` | `SECRETS_ENCRYPTION_KEY` changed without keeping the old key in the list. |
| `provider_empty_response` | Vertex returned empty candidates twice in a row. It is rare after the follow-up thinking fix; the next turn usually succeeds. |
