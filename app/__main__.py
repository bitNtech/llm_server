"""`python -m app` - run the gateway with uvicorn (single worker: sessions live in-process)."""
from __future__ import annotations

import sys

import uvicorn
from pydantic import ValidationError

from app.config import Settings


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None

    from app.main import create_app

    app = create_app(settings)

    class Server(uvicorn.Server):
        def handle_exit(self, sig: int, frame: object) -> None:
            services = getattr(app.state, "services", None)
            if services is not None:
                services.begin_drain()  # readiness -> 503 immediately; WebSockets wind down
            super().handle_exit(sig, frame)

    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,  # our structured logging is installed by create_app
        access_log=False,  # RequestContextMiddleware writes structured access logs
        proxy_headers=True,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        ws_ping_interval=settings.ws_ping_interval_s,
        ws_ping_timeout=settings.ws_ping_interval_s,
        ws_max_size=settings.ws_max_message_bytes,
        timeout_graceful_shutdown=int(settings.shutdown_grace_s) + 5,
        timeout_keep_alive=75,  # > AWS ALB idle timeout default (60 s)
        server_header=False,
    )
    Server(config).run()


if __name__ == "__main__":
    main()
