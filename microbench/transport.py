"""Esperimento A — microbenchmark di trasporto, senza LLM.

Misura il costo di invocare la stessa identica operazione attraverso
percorsi diversi, senza coinvolgere alcun modello linguistico. Isolare la
misura dall'LLM e' necessario: la varianza dell'endpoint condiviso e' di
ordini di grandezza superiore all'overhead qui misurato, e lo renderebbe
inosservabile.

Condizioni
----------

``diretto``
    chiamata in-process a ``operations.call``, senza alcuna mediazione.
    Non e' un braccio sperimentale: e' il riferimento inferiore, il costo
    dell'operazione quando nessun meccanismo di esposizione e' presente.
``langchain_tool``
    invocazione di un vero ``StructuredTool`` LangChain. E' il braccio
    LangChain dell'esperimento A.
``mcp_stdio_persistent``
    una sola sessione MCP su stdio, riusata per tutte le invocazioni.
``mcp_stdio_new_process``
    una nuova sessione MCP su stdio per ogni invocazione. Su stdio questo
    implica **avviare un nuovo processo Python**: il nome lo dice
    esplicitamente perche' la misura e' dominata dal tempo di import
    dell'interprete, non dal protocollo.
``mcp_http_persistent``
    una sola sessione MCP su Streamable HTTP, riusata.
``mcp_http_new_session``
    una nuova sessione MCP su HTTP per ogni invocazione, contro un server
    gia' in esecuzione: si paga il solo handshake di inizializzazione.

Il confronto fra ``mcp_stdio_new_process`` e ``mcp_http_new_session`` e'
il motivo per cui il trasporto HTTP e' stato aggiunto: separa il costo di
avvio del processo da quello, molto minore, del protocollo. Senza questa
separazione i ~470 ms della condizione stdio verrebbero attribuiti a MCP,
mentre sono tempo di import di Python.

Operazioni
----------

``echo``
    non effettua alcuna chiamata REST: la differenza fra le condizioni e'
    overhead di protocollo puro. **E' su questa operazione che va letto
    l'overhead**, perche' elimina del tutto la componente di rete.
``list_events``
    effettua una vera chiamata REST: colloca quell'overhead nel contesto
    di un'operazione realistica. La sottrazione fra condizioni e' qui meno
    pulita, perche' il percorso stdio effettua la chiamata REST da un
    processo con un proprio pool di connessioni.

Accorgimenti metodologici, dovuti al fatto che le misure girano su un
portatile senza ventola:

* le condizioni sono alternate a rotazione anziche' eseguite in blocco,
  cosi' che l'eventuale deriva termica colpisca tutte le condizioni allo
  stesso modo invece di penalizzare l'ultima;
* le prime ripetizioni sono scartate come riscaldamento, perche' import,
  cache e apertura delle connessioni inquinano le prime misure.

Uso::

    # opzionale, per le condizioni HTTP:
    BENCH_EXPOSE_ECHO=1 uv run uvicorn arm_mcp.http_server:app --port 8100

    uv run python -m microbench.transport --repetitions 100 --warmup 20
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from shared import operations

BENCH_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCH_ROOT / "results"

#: La barra finale non e' facoltativa: ``Mount("/mcp")`` di Starlette
#: risponde a ``/mcp`` con un redirect 307, che il client HTTP segue in
#: modo trasparente. La misura conterrebbe allora **due** round trip
#: invece di uno, gonfiando silenziosamente tutte le condizioni HTTP.
#: Stesso inganno del percorso ``/users/`` in ``operations.py``.
MCP_HTTP_URL = os.environ.get("BENCH_MCP_HTTP_URL", "http://127.0.0.1:8100/mcp/")

#: Operazioni misurate: nome logico -> (nome strumento, argomenti).
OPERATIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "echo": ("_bench_echo", {"payload": "ping"}),
    "list_events": ("list_events", {}),
}

Caller = Callable[[str, dict[str, Any]], Awaitable[Any]]

#: Ruolo di ciascuna condizione nel confronto. Serve a rendere leggibile
#: l'esito: dei sei percorsi misurati solo due sono i bracci sperimentali,
#: uno e' il riferimento inferiore e tre sono varianti di configurazione
#: di MCP che esistono per scomporne il costo. Senza questa colonna la
#: tabella non lascia capire quali righe siano il confronto vero.
ROLES: dict[str, str] = {
    "diretto": "riferimento",
    "langchain_tool": "BRACCIO LangChain",
    "mcp_stdio_persistent": "BRACCIO MCP",
    "mcp_http_persistent": "variante MCP",
    "mcp_http_new_session": "variante MCP",
    "mcp_stdio_new_process": "variante MCP",
}


def _server_params() -> StdioServerParameters:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BENCH_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Solo il microbenchmark chiede al server di pubblicare lo strumento
    # fittizio: negli esperimenti con il modello resta invisibile.
    env["BENCH_EXPOSE_ECHO"] = "1"
    return StdioServerParameters(
        command=sys.executable, args=["-m", "arm_mcp.server"], env=env
    )


# --- Costruzione dei percorsi di invocazione ------------------------------


async def _call_diretto(tool: str, args: dict[str, Any]) -> Any:
    """Invocazione diretta, senza alcuna mediazione."""
    return operations.call(tool, args)


def _build_langchain_caller() -> Caller:
    """Costruisce gli strumenti LangChain e restituisce l'invocatore.

    Si usa ``ainvoke`` e non ``invoke`` perche' e' cio' che fara' il
    braccio LangChain vero e proprio (fase 3): ``ToolNode`` di LangGraph
    invoca gli strumenti in modo asincrono quando il grafo gira async.
    Il costo eventuale di quel percorso fa parte del framework, non e' un
    artefatto della misura, e va quindi misurato nella forma in cui
    verra' effettivamente pagato.
    """
    from langchain_core.tools import tool as _tool

    from arm_langchain.tools import TOOLS_BY_NAME, _ripristinabile

    tools = dict(TOOLS_BY_NAME)
    # Lo strumento fittizio si costruisce nello stesso modo di quelli
    # reali — decoratore ``@tool`` sulla funzione, schema dedotto — perche'
    # la condizione misurata deve essere la stessa che paga il braccio.
    echo = _tool(_ripristinabile(operations._bench_echo))
    echo.handle_tool_error = True
    tools["_bench_echo"] = echo

    async def _call(tool: str, args: dict[str, Any]) -> Any:
        return await tools[tool].ainvoke(args)

    return _call


def _session_caller(session: ClientSession) -> Caller:
    async def _call(tool: str, args: dict[str, Any]) -> Any:
        return await session.call_tool(tool, args)

    return _call


async def _call_stdio_new_process(tool: str, args: dict[str, Any]) -> Any:
    """Avvia un processo server, apre una sessione, chiama, chiude.

    Include l'avvio dell'interprete Python e l'import dell'SDK, che
    dominano la misura: e' proprio per distinguerli dal costo di
    protocollo che esiste la condizione HTTP corrispondente.
    """
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


async def _call_http_new_session(tool: str, args: dict[str, Any]) -> Any:
    """Apre una nuova sessione MCP su un server HTTP gia' in esecuzione.

    A differenza della condizione stdio non si avvia alcun processo: il
    costo misurato e' l'handshake di inizializzazione piu' il round trip.
    """
    async with streamable_http_client(MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


async def _http_available() -> bool:
    """Verifica che il server MCP su HTTP risponda, senza farlo fallire."""
    try:
        await asyncio.wait_for(_call_http_new_session("_bench_echo", {}), timeout=10)
    except Exception as exc:  # noqa: BLE001 - qualunque errore = non disponibile
        print(f"  ! server MCP su HTTP non raggiungibile ({type(exc).__name__});"
              f" le condizioni HTTP verranno saltate", file=sys.stderr)
        return False
    return True


# --- Esecuzione ------------------------------------------------------------


async def run(
    repetitions: int,
    warmup: int,
    wanted: list[str] | None = None,
    seed: int = 20260803,
) -> tuple[dict[str, list[float]], list[str]]:
    """Esegue il microbenchmark e restituisce i campioni grezzi.

    L'ordine delle condizioni entro ciascuna ripetizione e' permutato a
    caso. Con un ordine fisso, la condizione misurata per prima verrebbe
    eseguita **sempre** subito dopo quelle piu' pesanti — che avviano
    processi e chiudono connessioni — e ne pagherebbe sistematicamente gli
    strascichi. Sarebbe una distorsione costante, non rumore, e cadrebbe
    ogni volta sulla stessa cella. La permutazione la trasforma in
    varianza distribuita; il seme e' fisso perche' l'esecuzione resti
    riproducibile.
    """
    async with contextlib.AsyncExitStack() as stack:
        wanted_set = set(wanted) if wanted else None

        def _requested(name: str) -> bool:
            return wanted_set is None or name in wanted_set

        callers: dict[str, Caller] = {}
        if _requested("diretto"):
            callers["diretto"] = _call_diretto
        if _requested("langchain_tool"):
            callers["langchain_tool"] = _build_langchain_caller()

        if _requested("mcp_stdio_persistent") or _requested("mcp_stdio_new_process"):
            read, write = await stack.enter_async_context(stdio_client(_server_params()))
            stdio_session = await stack.enter_async_context(ClientSession(read, write))
            await stdio_session.initialize()
            if _requested("mcp_stdio_persistent"):
                callers["mcp_stdio_persistent"] = _session_caller(stdio_session)
            if _requested("mcp_stdio_new_process"):
                callers["mcp_stdio_new_process"] = _call_stdio_new_process

        if (
            _requested("mcp_http_persistent") or _requested("mcp_http_new_session")
        ) and await _http_available():
            streams = await stack.enter_async_context(streamable_http_client(MCP_HTTP_URL))
            http_session = await stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )
            await http_session.initialize()
            if _requested("mcp_http_persistent"):
                callers["mcp_http_persistent"] = _session_caller(http_session)
            if _requested("mcp_http_new_session"):
                callers["mcp_http_new_session"] = _call_http_new_session

        conditions = list(callers)
        samples: dict[str, list[float]] = {
            f"{cond}::{op}": [] for cond in conditions for op in OPERATIONS
        }

        rng = random.Random(seed)
        order = list(conditions)
        total = warmup + repetitions
        for i in range(total):
            measuring = i >= warmup
            # Rotazione: a ogni giro si attraversano tutte le celle, cosi'
            # la deriva termica e' distribuita uniformemente.
            for op_label, (tool, args) in OPERATIONS.items():
                rng.shuffle(order)
                for cond in order:
                    start = time.perf_counter_ns()
                    await callers[cond](tool, args)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    if measuring:
                        samples[f"{cond}::{op_label}"].append(elapsed_ms)
            if measuring and (i - warmup + 1) % 10 == 0:
                print(f"  {i - warmup + 1}/{repetitions} ripetizioni", file=sys.stderr)

        return samples, conditions

    raise AssertionError("unreachable")


# --- Analisi ---------------------------------------------------------------


def summarize(values: list[float]) -> dict[str, float]:
    """Statistiche robuste: mediana e IQR, non media e deviazione standard.

    La distribuzione delle latenze e' asimmetrica e con code lunghe; la
    media sarebbe dominata dagli outlier.
    """
    ordered = sorted(values)
    q1, _, q3 = statistics.quantiles(ordered, n=4)
    return {
        "n": len(ordered),
        "median_ms": statistics.median(ordered),
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def report(samples: dict[str, list[float]], conditions: list[str]) -> dict[str, Any]:
    """Stampa il riepilogo e calcola gli overhead rilevanti."""
    summary = {key: summarize(vals) for key, vals in samples.items()}

    print(
        f"\n{'condizione':<24} {'ruolo':<19} {'operazione':<13} "
        f"{'mediana':>10} {'IQR':>10} {'p95':>10}"
    )
    print("-" * 92)
    for op_label in OPERATIONS:
        for cond in conditions:
            s = summary[f"{cond}::{op_label}"]
            print(
                f"{cond:<24} {ROLES.get(cond, ''):<19} {op_label:<13} "
                f"{s['median_ms']:>9.3f}ms {s['iqr_ms']:>9.3f}ms {s['p95_ms']:>9.3f}ms"
            )

    overhead: dict[str, dict[str, float]] = {}
    print(f"\n{'overhead sul riferimento diretto (mediana)':<50}")
    print("-" * 72)
    for op_label in OPERATIONS:
        base = summary[f"diretto::{op_label}"]["median_ms"]
        for cond in conditions:
            if cond == "diretto":
                continue
            median = summary[f"{cond}::{op_label}"]["median_ms"]
            overhead[f"{cond}::{op_label}"] = {
                "delta_ms": median - base,
                "factor": median / base if base > 0 else float("nan"),
            }
            print(f"{cond:<24} {op_label:<14} {median - base:>+9.3f}ms")

    # Il confronto che interessa davvero la tesi: MCP contro LangChain, non
    # MCP contro una chiamata di funzione nuda.
    contrasts: dict[str, float] = {}
    if "langchain_tool" in conditions:
        print(f"\n{'MCP rispetto al braccio LangChain (mediana, operazione echo)':<50}")
        print("-" * 72)
        lc = summary["langchain_tool::echo"]["median_ms"]
        for cond in conditions:
            if not cond.startswith("mcp_"):
                continue
            delta = summary[f"{cond}::echo"]["median_ms"] - lc
            contrasts[cond] = delta
            print(f"{cond:<24} {'echo':<14} {delta:>+9.3f}ms")

    if "mcp_stdio_new_process" in conditions and "mcp_http_new_session" in conditions:
        spawn = (
            summary["mcp_stdio_new_process::echo"]["median_ms"]
            - summary["mcp_http_new_session::echo"]["median_ms"]
        )
        print(
            f"\nCosto attribuibile all'avvio del processo Python "
            f"(stdio - http, nuova sessione): {spawn:+.3f}ms"
        )
        contrasts["process_spawn_cost_ms"] = spawn

    return {"summary": summary, "overhead": overhead, "contrasts": contrasts}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Sottoinsieme di condizioni separate da virgola. Serve a "
            "isolare un confronto dagli strascichi delle condizioni "
            "pesanti, p.es. --conditions diretto,langchain_tool"
        ),
    )
    parsed = parser.parse_args()

    wanted = parsed.conditions.split(",") if parsed.conditions else None

    print(
        f"Microbenchmark: {parsed.repetitions} ripetizioni misurate "
        f"(+{parsed.warmup} di riscaldamento) su {len(OPERATIONS)} operazioni",
        file=sys.stderr,
    )

    samples, conditions = await run(
        parsed.repetitions, parsed.warmup, wanted, parsed.seed
    )
    analysis = report(samples, conditions)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = parsed.out or RESULTS_DIR / f"microbench_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "A_transport",
                "timestamp_utc": timestamp,
                "repetitions": parsed.repetitions,
                "roles": ROLES,
                "warmup": parsed.warmup,
                "conditions": conditions,
                "python": sys.version,
                "platform": sys.platform,
                "samples_ms": samples,
                **analysis,
            },
            indent=2,
        )
    )
    print(f"\nRisultati grezzi salvati in {out_path}", file=sys.stderr)

    operations.close_client()


if __name__ == "__main__":
    asyncio.run(main())
