from __future__ import annotations

import threading
import time
import requests


class ManyfoldError(RuntimeError):
    pass


def _items(data) -> list:
    """Extract the item list from a paginated JSON/JSON-LD response."""
    if isinstance(data, list):
        return data
    for key in ("member", "members", "items", "models", "collections", "libraries", "@graph"):
        if isinstance(data.get(key), list):
            return data[key]
    return []


def _next_link(data) -> str | None:
    """Find the next-page link in a JSON/JSON-LD paginated response."""
    if not isinstance(data, dict):
        return None
    view = data.get("view")
    if isinstance(view, dict) and view.get("next"):
        return view["next"]
    nxt = data.get("next")
    if isinstance(nxt, str):
        return nxt
    if isinstance(nxt, dict):
        return nxt.get("@id")
    return None


def model_tags(model: dict) -> list[str]:
    """Read a model's tags regardless of response shape (strings or objects)."""
    for key in ("tags", "tag_list", "keywords"):
        val = model.get(key)
        if isinstance(val, list):
            out = []
            for t in val:
                if isinstance(t, str):
                    out.append(t)
                elif isinstance(t, dict):
                    name = t.get("name") or t.get("title")
                    if name:
                        out.append(name)
            return out
    return []


class ManyfoldClient:
    """Minimal client for the Manyfold JSON API.

    Auth is either a personal API token (sent as a Bearer token) or OAuth2
    client credentials exchanged at /oauth/token. Endpoint paths follow the
    documented v0 API; run `upload --check` against your instance to dump its
    OpenAPI spec and confirm capabilities (the docs live at /api on every
    instance).
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        min_interval: float = 0.25,
    ):
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._client_id = client_id
        self._client_secret = client_secret
        self._rate_lock = threading.Lock()
        self._last_request = 0.0
        self._min_interval = min_interval

    # --- plumbing ---------------------------------------------------------

    def _ensure_token(self) -> str:
        if self._token:
            return self._token
        if not (self._client_id and self._client_secret):
            raise ManyfoldError(
                "Set MANYFOLD_API_TOKEN, or MANYFOLD_CLIENT_ID and "
                "MANYFOLD_CLIENT_SECRET for OAuth client credentials."
            )
        resp = requests.post(
            f"{self.base_url}/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "read write",
            },
            timeout=30,
        )
        if not resp.ok:
            raise ManyfoldError(f"OAuth token request failed: HTTP {resp.status_code}")
        self._token = resp.json()["access_token"]
        return self._token

    def _request(self, method: str, path: str, retries: int = 4, **kwargs) -> requests.Response:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.setdefault("Accept", "application/json")
        headers["Authorization"] = f"Bearer {self._ensure_token()}"
        last_status = None
        for attempt in range(retries):
            with self._rate_lock:
                fire_at = max(time.time(), self._last_request + self._min_interval)
                self._last_request = fire_at
            gap = fire_at - time.time()
            if gap > 0:
                time.sleep(gap)
            try:
                resp = requests.request(method, url, headers=headers, timeout=60, **kwargs)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_status = str(e)
                time.sleep(min(30, 2 * 2 ** attempt))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_status = resp.status_code
                time.sleep(min(60, 2 * 2 ** attempt))
                continue
            return resp
        raise ManyfoldError(f"{method} {url} failed after {retries} attempts (last: {last_status})")

    def _paginate(self, path: str) -> list[dict]:
        items: list[dict] = []
        url: str | None = path
        seen: set[str] = set()
        while url and url not in seen:
            seen.add(url)
            resp = self._request("GET", url)
            if not resp.ok:
                raise ManyfoldError(f"GET {url} -> HTTP {resp.status_code}")
            data = resp.json()
            items.extend(_items(data))
            url = _next_link(data)
        return items

    @staticmethod
    def _resource_path(resource: dict, kind: str) -> str:
        """Build the request path for a fetched resource ('@id' URL or id)."""
        rid = resource.get("@id") or resource.get("id")
        if rid is None:
            raise ManyfoldError(f"Resource has no id: {resource}")
        if isinstance(rid, str) and rid.startswith("http"):
            return rid
        if isinstance(rid, str) and rid.startswith("/"):
            return rid
        return f"/{kind}/{rid}"

    # --- API surface ------------------------------------------------------

    def list_models(self) -> list[dict]:
        return self._paginate("/models")

    def list_collections(self) -> list[dict]:
        return self._paginate("/collections")

    def list_libraries(self) -> list[dict]:
        return self._paginate("/libraries")

    def update_model(self, model: dict, attributes: dict) -> None:
        path = self._resource_path(model, "models")
        resp = self._request("PATCH", path, json={"model": attributes})
        if resp.status_code in (400, 415, 422):
            # Some API versions accept flat JSON rather than Rails-wrapped
            resp = self._request("PATCH", path, json=attributes)
        if not resp.ok:
            raise ManyfoldError(f"PATCH {path} -> HTTP {resp.status_code}: {resp.text[:200]}")

    def set_model_tags(self, model: dict, tags: list[str]) -> None:
        self.update_model(model, {"tag_list": tags})

    def create_collection(self, name: str) -> dict:
        resp = self._request("POST", "/collections", json={"collection": {"name": name}})
        if resp.status_code in (400, 415, 422):
            resp = self._request("POST", "/collections", json={"name": name})
        if not resp.ok:
            raise ManyfoldError(f"POST /collections -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def trigger_scan(self) -> bool:
        """Ask the instance to rescan its libraries. Returns False if no scan
        endpoint is available (older API) — scan manually in the UI then."""
        try:
            libraries = self.list_libraries()
        except ManyfoldError:
            libraries = []
        triggered = False
        for lib in libraries:
            path = self._resource_path(lib, "libraries")
            resp = self._request("POST", f"{path}/scan")
            if resp.ok:
                triggered = True
        if not triggered:
            resp = self._request("POST", "/scan")
            triggered = resp.ok
        return triggered

    def openapi_spec(self) -> dict | None:
        """Fetch the instance's OpenAPI spec (served under /api)."""
        for path in ("/api/openapi.json", "/api-docs/openapi.json", "/api/spec.json", "/api.json"):
            resp = self._request("GET", path)
            if resp.ok:
                try:
                    return resp.json()
                except ValueError:
                    continue
        return None

    def capabilities(self) -> dict:
        """Summarize what this instance's API supports, from its OpenAPI spec."""
        spec = self.openapi_spec()
        caps = {
            "spec_found": spec is not None,
            "model_update": None,
            "model_upload": None,
            "collections_write": None,
            "scan": None,
        }
        if spec:
            paths = spec.get("paths", {})
            caps["model_update"] = any(
                "patch" in (ops or {}) or "put" in (ops or {})
                for p, ops in paths.items() if "/models/" in p
            )
            caps["model_upload"] = any(
                "post" in (ops or {}) for p, ops in paths.items()
                if p.rstrip("/").endswith("/models")
            )
            caps["collections_write"] = any(
                "post" in (ops or {}) for p, ops in paths.items()
                if p.rstrip("/").endswith("/collections")
            )
            caps["scan"] = any("scan" in p for p in paths)
        return caps
