"""Avvolge l'applicazione Event Manager senza modificarne il codice.

Il progetto di Programmazione Web resta intatto e consegnabile: qui viene
importato come dipendenza e arricchito dall'esterno con

* un middleware che conta le chiamate REST per run (header ``X-Run-Id``),
  che realizza la metrica proxy "numero di interazioni con il sistema";
* un endpoint di reset che ripristina uno stato iniziale deterministico
  fra una run e l'altra;
* la disattivazione dell'echo SQL di SQLAlchemy, che altrimenti
  scriverebbe su stdout a ogni query introducendo rumore nella latenza.

Avvio::

    EVENT_MANAGER_ROOT=/percorso/al/progetto uvicorn server.wrapper:app
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path.home() / "Desktop" / "ProgettoProgrammazioneWeb2026"
PROJECT_ROOT = Path(os.environ.get("EVENT_MANAGER_ROOT", _DEFAULT_ROOT)).resolve()

if not (PROJECT_ROOT / "app" / "main.py").exists():
    raise RuntimeError(
        f"Progetto Event Manager non trovato in {PROJECT_ROOT}. "
        "Imposta la variabile d'ambiente EVENT_MANAGER_ROOT."
    )

sys.path.insert(0, str(PROJECT_ROOT))

# La configurazione va fissata prima di importare app.main: il modulo
# app.data.db calcola il percorso del database al momento dell'import, a
# partire da config.root_dir. Senza percorso assoluto, il database
# verrebbe cercato relativamente alla directory di lavoro del benchmark.
from app.config import config  # noqa: E402

config.root_dir = PROJECT_ROOT / "app"

from fastapi import FastAPI, Request  # noqa: E402
from sqlmodel import Session, SQLModel  # noqa: E402

from app.data.db import engine  # noqa: E402
from app.main import app  # noqa: E402

from server.fixture import seed_fixture  # noqa: E402

# L'engine del progetto e' creato con echo=True: ogni query verrebbe
# stampata su stdout. Durante una campagna sono decine di migliaia di
# righe di I/O sincrono, cioe' rumore diretto sulla latenza misurata.
engine.echo = False

#: Prefissi di percorso esclusi dal conteggio: servono il frontend HTML,
#: gli asset statici, la documentazione interattiva e gli endpoint del
#: benchmark stesso. Nessuno di questi e' una chiamata dell'agente.
EXCLUDED_PREFIXES = (
    "/static",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/__bench__",
    "/events_list",
    "/event_detail",
    "/users_list",
)

#: Conteggio delle chiamate REST per run: run_id -> "METHOD /path" -> n
rest_calls: dict[str, Counter[str]] = {}


def _is_counted(path: str) -> bool:
    if path == "/":
        return False
    return not path.startswith(EXCLUDED_PREFIXES)


#: Conteggio incondizionato, indipendente dall'intestazione X-Run-Id.
#:
#: Serve al braccio MCP. Le sue chiamate REST partono dal processo del
#: **server MCP**, non da quello dell'host: la variabile di contesto che
#: porta l'identificativo di esecuzione vive nel processo dell'host e non
#: lo raggiunge, quindi l'intestazione non viene apposta e il conteggio
#: per esecuzione resterebbe a zero. Verificato: due invocazioni via MCP
#: producevano un conteggio di zero.
#:
#: Attribuire per identificativo richiederebbe di far viaggiare la run
#: dentro il protocollo, e l'unica via sarebbe aggiungere un argomento
#: agli strumenti — cioe' rompere la parita' degli schemi, che e' il
#: controllo su cui poggia l'intero confronto. Poiche' le esecuzioni
#: della campagna sono seriali, azzerare questo contatore prima di
#: ciascuna e leggerlo dopo attribuisce comunque in modo esatto.
global_calls: Counter[str] = Counter()


@app.middleware("http")
async def count_rest_calls(request: Request, call_next):
    """Conta ogni chiamata REST, per esecuzione e in modo incondizionato."""
    response = await call_next(request)
    if _is_counted(request.url.path):
        route = request.scope.get("route")
        key = f"{request.method} {route.path if route else request.url.path}"
        global_calls[key] += 1
        run_id = request.headers.get("X-Run-Id")
        if run_id:
            rest_calls.setdefault(run_id, Counter())[key] += 1
    return response


bench = FastAPI()


@bench.post("/reset")
def reset_state() -> dict[str, Any]:
    """Riporta il database allo stato iniziale deterministico.

    Piu' rapido di riavviare il processo fra una run e l'altra, e
    garantisce che ogni run parta dallo stesso mondo: senza questo, la
    run *n* opererebbe su uno stato modificato dalla run *n-1* e le
    traiettorie divergerebbero per ragioni estranee al protocollo.
    """
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        counts = seed_fixture(session)
    return {"ok": True, **counts}


@bench.post("/counters/{run_id}/reset")
def reset_counter(run_id: str) -> dict[str, Any]:
    """Azzera il conteggio delle chiamate REST per una run."""
    rest_calls.pop(run_id, None)
    return {"ok": True, "run_id": run_id}


@bench.post("/global/reset")
def reset_global() -> dict[str, Any]:
    """Azzera il contatore incondizionato, prima di una esecuzione."""
    global_calls.clear()
    return {"ok": True}


@bench.get("/global")
def get_global() -> dict[str, Any]:
    """Legge il contatore incondizionato, dopo una esecuzione."""
    return {
        "total": sum(global_calls.values()),
        "by_endpoint": dict(global_calls),
    }


@bench.get("/counters/{run_id}")
def get_counter(run_id: str) -> dict[str, Any]:
    """Restituisce il conteggio delle chiamate REST di una run."""
    counter = rest_calls.get(run_id, Counter())
    return {
        "run_id": run_id,
        "total": sum(counter.values()),
        "by_endpoint": dict(counter),
    }


app.mount("/__bench__", bench)
