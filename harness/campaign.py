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
import hashlib
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

#: Esiti che non vanno riutilizzati riprendendo: l'esecuzione non ha
#: prodotto una misura e va rifatta. ``limite_iterazioni`` **non** e' fra
#: questi: e' un fallimento del modello, cioe' un risultato.
NON_RIUTILIZZABILI = {"errore_llm", "errore_agente"}

#: File il cui contenuto determina *cosa* viene misurato. Un'esecuzione
#: prodotta da una versione diversa di questi file non e' confrontabile
#: con una prodotta da quella corrente, e riprendendo non va riutilizzata.
_SORGENTI_MISURA = (
    "shared/tasks.py",
    "shared/operations.py",
    "shared/nim.py",
    "arm_mcp/host.py",
    "arm_mcp/server.py",
    "arm_langchain/agent.py",
    "arm_langchain/tools.py",
    "arm_langchain/wire.py",
    "harness/trace.py",
    "harness/campaign.py",
)


def impronta_codice() -> str:
    """Riassume in dodici caratteri la versione del banco di prova.

    Serve al meccanismo di ripresa. Senza, riprendere una campagna
    riutilizza qualunque file gia' presente nella cartella, **anche se
    prodotto da una versione precedente del codice**: e' accaduto
    davvero, e ha reso inutilizzabile la prima ripetizione di una
    campagna a cinque, perche' tredici esecuzioni del giorno prima sono
    state saltate in silenzio e finite nella stessa tabella di quelle
    nuove.

    Non e' un controllo crittografico: e' un modo per accorgersi che il
    codice e' cambiato. Che sia cambiato in modo rilevante non lo si puo'
    stabilire meccanicamente, quindi qualunque differenza vale come tale.
    """
    radice = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for relativo in _SORGENTI_MISURA:
        percorso = radice / relativo
        digest.update(relativo.encode())
        digest.update(percorso.read_bytes() if percorso.exists() else b"<assente>")
    return digest.hexdigest()[:12]


def impronta_schemi(trace: RunTrace) -> tuple[str | None, int | None]:
    """Riassume gli schemi che il braccio ha davvero pubblicato al modello.

    L'impronta del codice non basta, e la ragione e' strutturale: il
    braccio MCP non deriva gli schemi dal sorgente che sta qui, li chiede
    a un **processo separato e di lunga durata**. Se quel server e' stato
    avviato prima di una modifica a ``shared/operations.py``, continua a
    servire i vecchi, e l'impronta del codice — calcolata sui file —
    resta perfettamente coerente mentre la misura non lo e' piu'.

    E' accaduto per una campagna intera: il braccio MCP pubblicava per
    ``create_event`` una descrizione di 21 caratteri ("Crea un nuovo
    evento.") mentre LangChain ne pubblicava 270, comprensive del formato
    ISO 8601 della data. I due bracci hanno quindi ricevuto descrizioni
    diverse dello stesso strumento, con l'asimmetria a favore di
    LangChain proprio su ``t7_creazione``, l'unico compito che richieda
    di sintetizzare una data nel formato dichiarato. Nessun controllo
    esistente se ne e' accorto.

    Registrando l'impronta in ogni traccia la cosa diventa **verificabile
    a posteriori dai soli file**: uno scarto fra i bracci, o una deriva a
    meta' campagna, si vedono rileggendo le tracce, senza dover ricordare
    quando un server era stato avviato.
    """
    if not trace.llm_calls:
        return None, None
    tools = (trace.llm_calls[0].get("request") or {}).get("tools") or []
    if not tools:
        return None, None
    testo = json.dumps(tools, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(testo.encode()).hexdigest()[:12], len(testo)


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
    trace.config["schema_hash"], trace.config["schema_chars"] = impronta_schemi(trace)
    trace.config["code_fingerprint"] = impronta_codice()

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


Chiave = tuple[str, str, str, int]


def _esistenti(directory: Path) -> dict[Chiave, list[tuple[Path, dict[str, Any]]]]:
    """Indicizza per chiave le tracce gia' presenti nella cartella."""
    per_chiave: dict[Chiave, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rep = (d.get("config") or {}).get("repetition")
        if rep is None:
            continue
        chiave = (d["arm"], d["model"], d["task_id"], rep)
        per_chiave.setdefault(chiave, []).append((path, d))
    return per_chiave


def _riutilizzabili(
    esistenti: dict[Chiave, list[tuple[Path, dict[str, Any]]]], impronta: str
) -> tuple[set[Chiave], set[Chiave]]:
    """Divide le esecuzioni presenti fra riutilizzabili e da rifare.

    Restituisce ``(da_saltare, obsolete)``: le seconde sono esecuzioni
    riuscite ma prodotte da un'altra versione del codice, che vanno
    rifatte e vanno **segnalate**, perche' il modo in cui questo problema
    si e' manifestato la prima volta e' stato il silenzio.
    """
    saltare: set[Chiave] = set()
    obsolete: set[Chiave] = set()
    for chiave, voci in esistenti.items():
        for _, d in voci:
            if d.get("status") in NON_RIUTILIZZABILI:
                continue
            if (d.get("config") or {}).get("code_fingerprint") != impronta:
                obsolete.add(chiave)
                continue
            saltare.add(chiave)
            break
    return saltare, obsolete - saltare


def _rimuovi_superate(
    esistenti: dict[Chiave, list[tuple[Path, dict[str, Any]]]], chiave: Chiave
) -> int:
    """Cancella le tracce che l'esecuzione in corso sostituisce.

    Senza, riprendendo si accumulano piu' file per la stessa cella e il
    riepilogo li conta tutti. Osservato: una campagna a settanta
    esecuzioni ne dichiarava settantuno presenti, e un compito compariva
    con sei valori invece di cinque, perche' il tentativo fallito era
    rimasto accanto alla sua ripetizione.
    """
    voci = esistenti.pop(chiave, [])
    for path, _ in voci:
        path.unlink(missing_ok=True)
    return len(voci)


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

    impronta = impronta_codice()
    plan = build_plan(models, tasks, parsed.repetitions)
    esistenti = _esistenti(directory)
    saltare, obsolete = _riutilizzabili(esistenti, impronta)
    todo = [p for p in plan if (p[0], p[2], p[1].task_id, p[3]) not in saltare]

    print(
        f"Campagna: {len(models)} modelli x {len(tasks)} compiti x "
        f"{parsed.repetitions} ripetizioni x {len(ARMS)} bracci = {len(plan)} esecuzioni"
    )
    print(f"Versione del banco di prova: {impronta}")
    if saltare:
        print(f"Gia' presenti in {directory.name}: {len(plan) - len(todo)}, saltate.")
    if obsolete:
        print(
            f"  ! {len(obsolete)} esecuzioni presenti sono state prodotte da "
            "un'altra versione del codice:\n"
            "    verranno rifatte, e le vecchie tracce rimosse."
        )
    print()

    started = time.perf_counter()
    for i, (arm, task, model, rep) in enumerate(todo, 1):
        label = f"[{i}/{len(todo)}] {arm:<10} {task.task_id:<22} rep{rep}"
        try:
            trace = await run_one(arm, task, model, rep)
        except Exception as exc:  # noqa: BLE001 — una esecuzione non ferma la campagna
            print(f"{label}  ECCEZIONE {type(exc).__name__}: {exc}")
            continue

        rimosse = _rimuovi_superate(esistenti, (arm, model, task.task_id, rep))
        path = trace.save(directory)
        if rimosse:
            print(f"{' ' * len(label)}  ({rimosse} traccia superata rimossa)")
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
