"""Registrazione integrale di una esecuzione agentica.

Ogni esecuzione — un compito svolto da un braccio con un modello — produce
un file JSON che contiene **tutto** ciò che è stato scambiato: i payload
inviati al modello, le risposte ricevute per intero, gli argomenti e i
risultati di ogni strumento invocato, i tempi e i conteggi.

La ragione è pratica. Interrogare il modello costa minuti e l'endpoint è
instabile: se una metrica viene definita dopo l'esecuzione — il numero di
token, la lunghezza dei testi scambiati, quante volte il modello ha
ripetuto uno strumento — ricalcolarla deve essere possibile leggendo i
file, senza rieseguire nulla. Per questo si registra anche ciò che al
momento non serve.

Il nome del file include braccio, modello, compito e istante: due
esecuzioni non si sovrascrivono mai, nemmeno se lanciate nello stesso
secondo con la stessa configurazione.

Le intestazioni HTTP non vengono mai registrate, perché conterrebbero la
chiave API.

Uso tipico dentro un ciclo agentico::

    trace = RunTrace(arm="mcp", model=MODEL, task_id="t1", prompt=...)
    ...
    trace.llm_call(payload, response_json, elapsed_s)
    trace.tool_call("get_event", {"event_id": 1}, result, elapsed_ms)
    ...
    trace.finish(final_answer="...", status="ok", rest_counts=counters)
    path = trace.save()
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACES_DIR = Path(__file__).resolve().parent.parent / "results" / "traces"

#: Chiavi del payload che non vanno mai registrate.
_REDACT = {"api_key", "authorization", "headers"}


def _clean(payload: dict[str, Any]) -> dict[str, Any]:
    """Copia il payload escludendo qualunque campo possa contenere segreti."""
    return {k: v for k, v in payload.items() if k.lower() not in _REDACT}


def _slug(text: str) -> str:
    """Riduce una stringa a una forma sicura per un nome di file."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")[:60]


@dataclass
class RunTrace:
    """Accumula gli eventi di una singola esecuzione e li salva su file."""

    arm: str
    model: str
    task_id: str
    prompt: str
    #: Configurazione che potrebbe servire a interpretare i risultati:
    #: temperatura, limite di iterazioni, trasporto, politica di sessione.
    config: dict[str, Any] = field(default_factory=dict)

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    final_answer: str | None = None
    status: str = "incompleto"
    rest_counts: dict[str, Any] = field(default_factory=dict)
    finished_utc: str | None = None

    # --- registrazione ----------------------------------------------------

    def llm_call(
        self,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        elapsed_s: float,
        *,
        attempt: int = 1,
        error: str | None = None,
    ) -> None:
        """Registra una interrogazione al modello, richiesta e risposta intere.

        ``request`` viene salvato completo di ``messages`` e ``tools``: è
        ciò che consente di ricontare i token o di verificare a posteriori
        che i due bracci abbiano davvero inviato lo stesso input.
        """
        usage = (response or {}).get("usage") or {}
        self.llm_calls.append(
            {
                "index": len(self.llm_calls) + 1,
                "attempt": attempt,
                "elapsed_s": elapsed_s,
                "request": _clean(request),
                "response": response,
                "error": error,
                # Estratti per comodità: restano comunque ricavabili da
                # ``response``, che è salvato per intero.
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": (
                    (response or {}).get("choices", [{}])[0].get("finish_reason")
                    if response
                    else None
                ),
            }
        )

    def tool_call(
        self,
        name: str,
        arguments: Any,
        result: Any,
        elapsed_ms: float,
        *,
        via: str | None = None,
        is_error: bool | None = None,
    ) -> None:
        """Registra l'invocazione di uno strumento e il suo esito."""
        self.tool_calls.append(
            {
                "index": len(self.tool_calls) + 1,
                "name": name,
                "arguments": arguments,
                "result": result,
                "elapsed_ms": elapsed_ms,
                "via": via or self.arm,
                "is_error": is_error,
            }
        )

    def event(self, kind: str, **payload: Any) -> None:
        """Registra un fatto che non è né una chiamata al modello né a uno
        strumento: un ritentativo, un limite raggiunto, un errore di rete."""
        self.events.append(
            {
                "kind": kind,
                "at_utc": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )

    def finish(
        self,
        *,
        final_answer: str | None = None,
        status: str = "ok",
        rest_counts: dict[str, Any] | None = None,
    ) -> None:
        self.final_answer = final_answer
        self.status = status
        self.rest_counts = rest_counts or {}
        self.finished_utc = datetime.now(timezone.utc).isoformat()

    # --- metriche ---------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        """Calcola le metriche della tesi a partire dagli eventi registrati.

        Sono ricalcolabili in qualunque momento rileggendo il file: questa
        funzione è una comodità, non l'unica via per ottenerle.

        I token vengono presi da ``usage`` se l'endpoint lo fornisce. In
        caso contrario il campo resta ``None`` e si riporta il numero di
        caratteri, che consente una stima a posteriori senza rieseguire
        nulla.
        """
        prompt_tok = [c["prompt_tokens"] for c in self.llm_calls if c["prompt_tokens"]]
        compl_tok = [
            c["completion_tokens"] for c in self.llm_calls if c["completion_tokens"]
        ]
        chars_in = sum(
            len(json.dumps(c["request"].get("messages", []), ensure_ascii=False))
            + len(json.dumps(c["request"].get("tools", []), ensure_ascii=False))
            for c in self.llm_calls
        )
        chars_out = sum(
            len(json.dumps(c["response"], ensure_ascii=False)) if c["response"] else 0
            for c in self.llm_calls
        )
        return {
            "n_llm_calls": len(self.llm_calls),
            "n_tool_calls": len(self.tool_calls),
            "n_rest_calls": self.rest_counts.get("total"),
            "latency_llm_s": sum(c["elapsed_s"] for c in self.llm_calls),
            "latency_tools_ms": sum(c["elapsed_ms"] for c in self.tool_calls),
            "prompt_tokens": sum(prompt_tok) if prompt_tok else None,
            "completion_tokens": sum(compl_tok) if compl_tok else None,
            "tokens_reported_by_endpoint": bool(prompt_tok),
            "chars_sent": chars_in,
            "chars_received": chars_out,
            "n_retries": sum(1 for e in self.events if e["kind"] == "retry"),
            "status": self.status,
        }

    # --- persistenza ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "model": self.model,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "config": self.config,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "status": self.status,
            "final_answer": self.final_answer,
            "rest_counts": self.rest_counts,
            "metrics": self.metrics(),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "events": self.events,
        }

    def save(self, directory: Path | None = None) -> Path:
        """Scrive la traccia su un file dal nome irripetibile.

        Il nome contiene braccio, modello, compito, istante e
        identificativo dell'esecuzione: nessuna esecuzione può
        sovrascriverne un'altra, nemmeno a parità di configurazione.
        """
        directory = directory or TRACES_DIR
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = (
            f"{_slug(self.arm)}__{_slug(self.model)}__{_slug(self.task_id)}"
            f"__{stamp}__{self.run_id}.json"
        )
        path = directory / name
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
        )
        return path


# --- lettura ---------------------------------------------------------------


def load_traces(directory: Path | None = None) -> list[dict[str, Any]]:
    """Rilegge tutte le tracce salvate, per l'analisi a posteriori."""
    directory = directory or TRACES_DIR
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f"  ! traccia illeggibile, saltata: {p.name}")
    return out


def metrics_table(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estrae una riga di metriche per traccia, pronta per l'analisi."""
    return [
        {
            "run_id": t["run_id"],
            "arm": t["arm"],
            "model": t["model"],
            "task_id": t["task_id"],
            **t.get("metrics", {}),
        }
        for t in traces
    ]
