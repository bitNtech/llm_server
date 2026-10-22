"""Outbound URL policy for tenant-supplied targets (HTTP tools, MCP servers, webhooks).

Tenants configure URLs the server will call, so without a policy a tenant could reach the
metadata server (169.254.169.254), localhost admin ports or the private network (SSRF). Every
target's hostname is resolved and rejected if any address is private, loopback, link-local,
multicast, reserved or unspecified - unless ALLOW_PRIVATE_NETWORK_TARGETS=true or the host is in
ALLOWED_TARGET_HOSTS. Production additionally requires https for non-allowlisted hosts.
The connection is resolved again by the HTTP client, so callers also check the address they
actually connected to (check_peer) before using the response - this defeats DNS rebinding.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

from app.core.errors import BadRequest

_METADATA_HOSTS = {"metadata.google.internal", "metadata", "169.254.169.254"}


class NetGuard:
    def __init__(self, allow_private: bool, allowed_hosts: list[str], require_https: bool) -> None:
        self.allow_private = allow_private
        self.allowed_hosts = set(allowed_hosts)
        self.require_https = require_https

    def check_syntax(self, url: str, what: str = "URL") -> str:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise BadRequest(f"{what} must be an absolute http(s) URL", code="invalid_url")
        if parts.username or parts.password:
            raise BadRequest(f"{what} must not embed credentials; use secret headers", code="invalid_url")
        host = parts.hostname.lower()
        if host in _METADATA_HOSTS:
            raise BadRequest(f"{what} targets the cloud metadata service", code="forbidden_target")
        if self.require_https and parts.scheme != "https" and host not in self.allowed_hosts:
            raise BadRequest(f"{what} must use https", code="insecure_url")
        return host

    async def check(self, url: str, what: str = "URL") -> None:
        host = self.check_syntax(url, what)
        if self.allow_private or host in self.allowed_hosts:
            return
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            loop = asyncio.get_running_loop()
            try:
                infos = await asyncio.wait_for(loop.getaddrinfo(host, None, type=socket.SOCK_STREAM), 5)
            except (TimeoutError, OSError) as exc:
                raise BadRequest(f"{what} host {host!r} does not resolve", code="unresolvable_host") from exc
            addresses = [ipaddress.ip_address(info[4][0].split("%", 1)[0]) for info in infos]
        for addr in addresses:
            if _blocked(addr):
                raise BadRequest(f"{what} resolves to a non-public address ({addr})", code="forbidden_target")

    def check_peer(self, url: str, response: Any, what: str = "URL") -> None:
        """Reject a response whose connection went to a non-public address (an httpx response)."""
        host = self.check_syntax(url, what)
        if self.allow_private or host in self.allowed_hosts:
            return
        stream = response.extensions.get("network_stream")
        peer = stream.get_extra_info("server_addr") if stream is not None else None
        if not peer:
            return
        addr = ipaddress.ip_address(str(peer[0]).split("%", 1)[0])
        if _blocked(addr):
            raise BadRequest(f"{what} connected to a non-public address ({addr})", code="forbidden_target")


    def response_hook(self, what: str) -> Any:
        """httpx event hook that applies check_peer to every response (incl. streaming ones)."""
        async def hook(response: Any) -> None:
            self.check_peer(str(response.request.url), response, what)
        return hook


def _blocked(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
            or addr.is_reserved or addr.is_unspecified)
