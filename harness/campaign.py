"""Fase 6 — esecuzione della campagna di misura.

Esegue i compiti con entrambi i bracci, su uno o piu' modelli, ripetendo
ciascuna combinazione, e salva una traccia per esecuzione.

Tre discipline governano l'ordine, e nessuna e' cosmetica.

**I bracci si alternano esecuzione per esecuzione**, non a blocchi. Due
esecuzioni identiche della fase 0, a poche ore di distanza, hanno dato
latenze fra 18 e 164 secondi con un ritentativo e fra 9 e 26 senza: la
dispersione fra sessioni supera quella interna a una sessione. Un
endpoint carico cambia anche la frequenza dei ritentativi e il
troncamento delle risposte, quindi puo' toccare il numero di iterazioni,
che e' una delle metriche rivendicate. Eseguire un braccio di mattina e
l'altro di sera non sarebbe rumore ma un fattore confondente.

**L'ordine ruota** a ogni cella, cosi' che nessuno dei due paghi
sistematicamente il costo di essere sempre il primo o l'ultimo.

**Lo stato viene ripristinato prima di ogni esecuzione**, non solo prima
dei compiti che scrivono. Costa poco e rimuove in blocco ogni dipendenza
dall'ordine.

L'esecuzione e' seriale e riprendibile: se viene interrotta, rilanciarla
con la stessa cartella di destinazione salta le esecuzioni gia' presenti.

Uso::

    uv run python -m harness.campaign --repetitions 3
    uv run python -m harness.campaign --models openai/gpt-oss-120b,nvidia/nemotron-3-super-120b-a12b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from harness import summary
from harness.trace import RunTrace
from shared.env import ENV_PATH  # carica .env, se presente  # noqa: F401
from shared.tasks import TASKS, TASKS_BY_ID, Task

BENCH_URL = "http://127.0.0.1:8000/__bench__"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: I due bracci, ciascuno nella forma idiomatica del proprio ecosistema.
ARMS = ("mcp", "langchain")


def _reset_state() -> None:
    """Ripristina i dati di prova e azzera il contatore incondizionato."""
    httpx.post(f"{BENCH_URL}/reset", timeout=30.0).raise_for_status()
    httpx.post(f"{BENCH_URL}/global/reset", timeout=30.0).raise_for_status()


def _read_counter() -> dict[str, Any]:
    """Legge quante chiamate REST ha prodotto l'esecuzione appena conclusa.

    Si usa il contatore **incondizionato** e non quello per identificativo:
    nel braccio MCP le chiamate partono dal processo del server, che non
    riceve l'identificativo dell'esecuzione. Poiche' le esecuzioni sono
    seriali e lo stato viene azzerato prima di ciascuna, il totale letto
    qui appartiene per intero all'ultima eseguita.
    """
    return httpx.get(f"{BENCH_URL}/global", timeout=30.0).json()


async def run_one(arm: str, task: Task, model: str, repetition: int) -> RunTrace:
    """Esegue una singola combinazione e restituisce la traccia."""
    _reset_state()

    if arm == "mcp":
        from arm_mcp.host import run_task as run_mcp

        trace = await run_mcp(task, model=model)
    else:
        from arm_langchain.agent import run_task as run_lc

        trace = await run_lc(task, model=model)

    counts = _read_counter()
    trace.rest_counts = counts
    trace.config["repetition"] = repetition

    # La verifica sullo stato e' piu' solida di quella sul testo, perche'
    # non dipende da come il modello ha formulato la risposta.
    if task.mutates_state:
        from shared import operations
        from shared.tasks import verify_state

        ok = verify_state(task, operations.list_registrations() or [])
        trace.config["state_verified"] = ok
        if trace.status == "ok" and ok is False:
            trace.status = "stato_non_modificato"

    return trace


def _already_done(directory: Path) -> set[tuple[str, str, str, int]]:
    """Chiavi delle esecuzioni gia' presenti, per poter riprendere."""
    done: set[tuple[str, str, str, int]] = set()
    for path in directory.glob("*.json"):
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rep = (d.get("config") or {}).get("repetition")
        if rep is not None and d.get("status") not in {"errore_llm", "errore_agente"}:
            done.add((d["arm"], d["model"], d["task_id"], rep))
    return done


def build_plan(
    models: list[str], tasks: list[Task], repetitions: int
) -> list[tuple[str, Task, str, int]]:
    """Costruisce l'ordine di esecuzione.

    Entro ogni cella i bracci sono consecutivi, cosi' che incontrino lo
    stesso stato dell'endpoint. L'ordine **ruota** a ogni cella, cosi' che
    ciascun braccio occupi ogni posizione lo stesso numero di volte e
    nessuno paghi sistematicamente il costo di essere il primo o l'ultimo.
    Con due bracci la rotazione coincide con l'alternanza; resta scritta
    in forma generale perche' regge anche un terzo braccio.
    """
    plan: list[tuple[str, Task, str, int]] = []
    cella = 0
    for rep in range(1, repetitions + 1):
        for task in tasks:
            for model in models:
                k = cella % len(ARMS)
                for arm in ARMS[k:] + ARMS[:k]:
                    plan.append((arm, task, model, rep))
                cella += 1
    return plan


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--models", default="openai/gpt-oss-120b")
    parser.add_argument("--tasks", default=None, help="id separati da virgola")
    parser.add_argument("--out", type=Path, default=None)
    parsed = parser.parse_args()

    models = [m.strip() for m in parsed.models.split(",") if m.strip()]
    tasks = (
        [TASKS_BY_ID[t.strip()] for t in parsed.tasks.split(",")]
        if parsed.tasks
        else list(TASKS)
    )

    directory = parsed.out or RESULTS_DIR / "campagna"
    directory.mkdir(parents=True, exist_ok=True)

    try:
        _reset_state()
    except httpx.HTTPError as exc:
        print(f"Server di prova non raggiungibile su {BENCH_URL}: {exc}")
        print("Avviare prima:  uv run uvicorn server.wrapper:app --port 8000")
        return 1

    plan = build_plan(models, tasks, parsed.repetitions)
    done = _already_done(directory)
    todo = [p for p in plan if (p[0], p[2], p[1].task_id, p[3]) not in done]

    print(
        f"Campagna: {len(models)} modelli x {len(tasks)} compiti x "
        f"{parsed.repetitions} ripetizioni x {len(ARMS)} bracci = {len(plan)} esecuzioni"
    )
    if done:
        print(f"Gia' presenti in {directory.name}: {len(plan) - len(todo)}, saltate.")
    print()

    started = time.perf_counter()
    for i, (arm, task, model, rep) in enumerate(todo, 1):
        label = f"[{i}/{len(todo)}] {arm:<10} {task.task_id:<22} rep{rep}"
        try:
            trace = await run_one(arm, task, model, rep)
        except Exception as exc:  # noqa: BLE001 — una esecuzione non ferma la campagna
            print(f"{label}  ECCEZIONE {type(exc).__name__}: {exc}")
            continue

        path = trace.save(directory)
        m = trace.metrics()
        print(
            f"{label}  {trace.status:<20} "
            f"llm={m['n_llm_calls']} tool={m['n_tool_calls']} "
            f"rest={m['n_rest_calls']} tok={m['prompt_tokens']}/{m['completion_tokens']} "
            f"{m['latency_llm_s']:.1f}s  -> {path.name[:40]}"
        )

    elapsed = time.perf_counter() - started
    print(f"\nCompletata in {elapsed / 60:.1f} minuti. Tracce in {directory}")

    # Il riepilogo si genera qui, ma e' derivato: rileggendo le tracce si
    # ricostruisce identico in qualunque momento, anche su una campagna
    # interrotta a meta'. Sostituisce il vecchio indice, che riportava il
    # piano di esecuzione e nessun risultato.
    percorso = summary.scrivi(directory, {"planned": len(plan)})
    if percorso is not None:
        print(f"Riepilogo: {percorso}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
