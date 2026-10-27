import json
import re
import time

import requests
from openhost_test_harness import OpenhostStack

# A ~1 MB model that llama.cpp's own server tests use, so a completion is cheap on CI.
TINY_MODEL_REPO = "ggml-org/tiny-llamas"
TINY_MODEL_FILE = "stories260K.gguf"


def _logs(stack: OpenhostStack) -> str:
    return f"\n\napp logs:\n{stack.app_logs()[-5000:]}"


def _owner_page(stack: OpenhostStack) -> str:
    deadline = time.monotonic() + 60
    while True:
        response = stack.owner_session.get(f"{stack.url}/", headers={"Accept": "text/html"}, timeout=30)
        if response.status_code == 200 or time.monotonic() > deadline:
            break
        time.sleep(2)
    assert response.status_code == 200, f"GET / returned {response.status_code}: {response.text[:500]}{_logs(stack)}"
    return response.text


def _api_token(page: str) -> str:
    match = re.search(r'id="api-token" value="([^"]+)"', page)
    assert match, f"no API token on the owner's page: {page[:1000]}"
    return match.group(1)


def test_owner_gets_token_and_setup_urls(stack: OpenhostStack) -> None:
    page = _owner_page(stack)
    token = _api_token(page)
    # The setup snippets point editors at the app's public URL, and the token persists in app_data.
    assert f'endpoint_fim = "{stack.url}/infill"' in page, f"setup instructions don't use {stack.url}: {page[-3000:]}"
    state = json.loads((stack.data_dir / "state.json").read_text())
    assert state.get("api_token") == token, f"token not persisted in app_data state.json: {state}"


def test_inference_endpoints_need_the_app_token(stack: OpenhostStack) -> None:
    token = _api_token(_owner_page(stack))
    for path in ("/completions", "/v1/completions", "/infill"):
        anonymous = requests.post(f"{stack.url}{path}", json={"prompt": "a"}, timeout=30)
        assert anonymous.status_code == 401 and "invalid_api_key" in anonymous.text, (
            f"POST {path} without a token: {anonymous.status_code}: {anonymous.text[:300]}"
        )
        wrong = requests.post(
            f"{stack.url}{path}", json={"prompt": "a"}, headers={"Authorization": "Bearer nope"}, timeout=30
        )
        assert wrong.status_code == 401, f"POST {path} with a wrong token: {wrong.status_code}: {wrong.text[:300]}"
        # Past the router (a public path) and past the app's token check; 503 until a model is loaded.
        authorized = requests.post(
            f"{stack.url}{path}", json={"prompt": "a"}, headers={"Authorization": f"Bearer {token}"}, timeout=60
        )
        assert authorized.status_code != 401, (
            f"POST {path} with the app token was refused: {authorized.status_code}: {authorized.text[:300]}"
        )
    # The management UI and its API stay owner-only.
    for path in ("/", "/api/models"):
        response = requests.get(f"{stack.url}{path}", allow_redirects=False, timeout=30)
        assert response.status_code in (302, 303, 307, 401, 404), (
            f"anonymous GET {path}: {response.status_code}: {response.text[:300]}"
        )
        assert "api-token" not in response.text, f"anonymous GET {path} leaked the token page"


def test_download_activate_and_complete(stack: OpenhostStack) -> None:
    token = _api_token(_owner_page(stack))
    download = stack.owner_session.post(
        f"{stack.url}/models/download", data={"repo": TINY_MODEL_REPO, "filename": TINY_MODEL_FILE}, timeout=30
    )
    assert download.status_code == 200 and "err=" not in download.url, f"download: {download.url}{_logs(stack)}"

    deadline = time.monotonic() + 300
    status = None
    while time.monotonic() < deadline:
        models = stack.owner_session.get(f"{stack.url}/api/models", timeout=30).json()
        status = models["downloads"].get(TINY_MODEL_FILE, {})
        if status.get("status") in ("complete", "failed"):
            break
        time.sleep(2)
    assert status and status.get("status") == "complete", f"download didn't complete: {status}{_logs(stack)}"
    # Weights go to the backup-excluded temp data dir, not app_data.
    assert (stack.temp_data_dir / "models" / TINY_MODEL_FILE).is_file(), f"model not in {stack.temp_data_dir}/models"

    activate = stack.owner_session.post(
        f"{stack.url}/models/activate", data={"filename": TINY_MODEL_FILE}, timeout=180
    )
    assert activate.status_code == 200 and "err=" not in activate.url, f"activate: {activate.url}{_logs(stack)}"

    response = requests.post(
        f"{stack.url}/v1/completions",
        json={"prompt": "Once upon a time", "max_tokens": 8},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    assert response.status_code == 200, f"completion: {response.status_code}: {response.text[:500]}{_logs(stack)}"
    choices = response.json().get("choices")
    assert choices and choices[0].get("text"), f"no completion text: {response.text[:500]}"
