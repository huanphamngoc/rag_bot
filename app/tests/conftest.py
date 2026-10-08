"""Suite-wide guards."""
from __future__ import annotations

import os

import pytest

# `import mlflow` starts usage telemetry unless this is set before the import, and test modules
# import it at collection time - before pytest's own markers would make MLflow skip it.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"


@pytest.fixture(autouse=True)
def tracing_never_reaches_a_real_server(monkeypatch):
    """No test may send a trace to a real MLflow server.

    The compose service loads .env, so a developer who set MLFLOW_TRACKING_URI would otherwise
    record fake test conversations on their own server (the crawler once leaked test traces). The
    tracing module decides once per process from that variable, so the variables are removed and
    the decision reset around every test; tests of tracing point it at a fake server on localhost.
    """
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_EXPERIMENT_ID", "MLFLOW_UI_URL",
                "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    from crawlerrag.rag import tracing
    tracing.reset()
    yield
    tracing.reset()


@pytest.fixture(scope="session")
def rules_dir():
    """The YAML rules shipped with the app (app/rules, /app/rules in the image)."""
    from pathlib import Path
    path = Path(__file__).parents[1] / "rules"
    if not path.is_dir():
        pytest.fail(f"rules directory not found: {path}")
    return path


@pytest.fixture(scope="session")
def ruleset(rules_dir):
    from crawlerrag.rules import load_rules
    return load_rules(rules_dir)


def permissive_ruleset():
    """A rule set whose input gate enforces only the length limits.

    The tests of the conversation, tracing and web layers are about those layers, not about which
    questions ``rules/qualify.yaml`` lets through - ``test_qualify.py`` and ``test_chat_graph.py`` cover
    that. Without this they would all have to phrase every question so the real gate passes it.
    """
    from types import SimpleNamespace

    from crawlerrag.rules.models import QualifyRules
    return SimpleNamespace(qualify=QualifyRules())
