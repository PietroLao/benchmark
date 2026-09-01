"""Rende leggibile una traccia, e affianca i due bracci turno per turno.

Serve a rispondere a mano alla domanda che nessuna metrica puo' chiudere:
**il banco di prova sta misurando davvero cio' che il modello fa?**

Le tabelle del riepilogo sono numeri derivati. Se un difetto del nostro
codice alterasse cio' che il modello riceve, i numeri resterebbero
plausibili e nessun controllo automatico se ne accorgerebbe: e' accaduto
davvero, con l'host che consegnava un ottavo dei dati e il modello che
rispondeva coerentemente sbagliato. L'unico rimedio e' leggere la
conversazione.

Due modi d'uso.

``--task t5_cancellazione_multipla`` mostra la conversazione di un braccio
per esteso: cosa e' stato chiesto, cosa il modello ha risposto, quali
strumenti ha invocato con quali argomenti, cosa gli e' tornato indietro.

``--task t5_cancellazione_multipla --confronta`` affianca i due bracci
turno per turno, evidenziando dove divergono. E' la vista che rende
verificabile a mano l'uguaglianza riportata dal riepilogo.

Uso::

    uv run python -m harness.leggi --task t1_conteggio
    uv run python -m harness.leggi --task t6_iscrizione_condizionale --confronta
    uv run python -m harness.leggi --task t4_conflitto --confronta --intero
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: Quanto mostrare di ogni testo prima di troncare. Con ``--intero`` il
#: troncamento non avviene: serve quando si sospetta che il difetto stia
#: proprio in cio' che viene tagliato.
LARGHEZZA = 150


def _taglia(testo: str, intero: bool) -> list[str]:
    """Riduce un testo a una o piu' righe pronte per la stampa.

    Gli spazi vengono compattati: i risultati degli strumenti arrivano come
    JSON indentato dal braccio MCP e compatto da LangChain, e mostrare
    quell'indentazione renderebbe illeggibile proprio cio' che si vuole
    confrontare. La differenza di formattazione resta visibile nelle
    parentesi e nella lunghezza dichiarata a fine riga.
    """
    piatto = re.sub(r"\s+", " ", (testo or "")).strip()
    if not intero:
        if len(piatto) > LARGHEZZA:
            piatto = f"{piatto[:LARGHEZZA]}…  [{len(testo)} car.]"
        return [piatto]
    return textwrap.wrap(piatto, 96) or [""]


def carica(directory: Path, arm: str, task: str, rep: int) -> dict[str, Any] | None:
    for path in sorted(directory.glob(f"{arm}__*{task}*.json")):
        traccia = json.loads(path.read_text())
        if (traccia.get("config") or {}).get("repetition") in (rep, None):
            return traccia
    return None


def turni(traccia: dict[str, Any]) -> list[tuple[str, str]]:
    """Riduce una traccia alla sequenza di eventi osservabili.

    Si legge dall'**ultima** richiesta, che contiene l'intera
    conversazione accumulata, e vi si aggiunge la risposta finale. E' cio'
    che il modello ha effettivamente visto, non una ricostruzione a
    posteriori dagli eventi.
    """
    if not traccia.get("llm_calls"):
        return []
    messaggi = traccia["llm_calls"][-1]["request"].get("messages", [])
    ultima = (traccia["llm_calls"][-1].get("response") or {}).get("choices", [{}])[0]

    fuori: list[tuple[str, str]] = []
    for m in messaggi:
        ruolo = m.get("role")
        if ruolo == "system":
            fuori.append(("sistema", m.get("content") or ""))
        elif ruolo == "user":
            fuori.append(("utente", m.get("content") or ""))
        elif ruolo == "assistant":
            for chiamata in m.get("tool_calls") or []:
                fn = chiamata.get("function", {})
                fuori.append(("chiama", f"{fn.get('name')}({fn.get('arguments')})"))
            if m.get("content"):
                fuori.append(("modello", m["content"]))
        elif ruolo == "tool":
            fuori.append(("risultato", m.get("content") or ""))

    finale = (ultima.get("message") or {}).get("content")
    if finale:
        fuori.append(("modello", finale))
    return fuori


def mostra(traccia: dict[str, Any], intero: bool) -> None:
    print(f"  esito: {traccia['status']}   modello: {traccia['model']}")
    stato = (traccia.get("config") or {}).get("state_verified")
    if stato is not None:
        print(f"  stato del servizio verificato: {'si' if stato else 'NO'}")
    print()
    for etichetta, testo in turni(traccia):
        righe = _taglia(testo, intero)
        print(f"  {etichetta:>10} │ {righe[0]}")
        for r in righe[1:]:
            print(f"  {'':>10} │ {r}")


def confronta(a: dict[str, Any], b: dict[str, Any], intero: bool) -> int:
    """Affianca i due bracci turno per turno."""
    ta, tb = turni(a), turni(b)
    divergenze = 0

    for i in range(max(len(ta), len(tb))):
        ea, xa = ta[i] if i < len(ta) else ("—", "")
        eb, xb = tb[i] if i < len(tb) else ("—", "")
        uguali = ea == eb and xa == xb
        if not uguali:
            divergenze += 1
        segno = " " if uguali else "≠"

        if uguali:
            for j, r in enumerate(_taglia(xa, intero)):
                print(f"{segno if j == 0 else ' '}  {ea if j == 0 else '':>10} │ {r}")
        else:
            for etichetta, braccio, testo in ((ea, "mcp", xa), (eb, "langchain", xb)):
                righe = _taglia(testo, intero)
                print(f"{segno}  {etichetta:>10} │ {braccio:<10}│ {righe[0]}")
                for r in righe[1:]:
                    print(f"   {'':>10} │ {'':<10}│ {r}")

    print()
    if divergenze == 0:
        print(f"Nessuna divergenza: {len(ta)} turni identici nei due bracci.")
    else:
        print(f"{divergenze} turni divergenti su {max(len(ta), len(tb))}.")
        print(
            "Attenzione: una divergenza nel **testo** non implica un difetto.\n"
            "I due ecosistemi formattano diversamente i risultati degli\n"
            "strumenti e i messaggi d'errore, e il modello genera testo\n"
            "diverso a ogni esecuzione. Cio' che deve coincidere e' la\n"
            "sequenza delle azioni: quali strumenti, con quali argomenti,\n"
            "in quale ordine."
        )
    return divergenze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dir", type=Path, default=RESULTS_DIR / "campagna")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--arm", default="mcp", choices=("mcp", "langchain"))
    parser.add_argument("--confronta", action="store_true")
    parser.add_argument("--intero", action="store_true", help="non troncare i testi")
    parsed = parser.parse_args()

    if parsed.confronta:
        a = carica(parsed.dir, "mcp", parsed.task, parsed.rep)
        b = carica(parsed.dir, "langchain", parsed.task, parsed.rep)
        if not a or not b:
            print(f"Servono due tracce per {parsed.task} in {parsed.dir}.")
            return 1
        print(f"{parsed.task}, ripetizione {parsed.rep}")
        print(f"  mcp       : {a['status']}")
        print(f"  langchain : {b['status']}")
        print()
        return 0 if confronta(a, b, parsed.intero) >= 0 else 1

    t = carica(parsed.dir, parsed.arm, parsed.task, parsed.rep)
    if not t:
        print(f"Traccia non trovata: {parsed.arm} / {parsed.task} in {parsed.dir}.")
        return 1
    print(f"{parsed.task}, braccio {parsed.arm}, ripetizione {parsed.rep}")
    mostra(t, parsed.intero)
    return 0


if __name__ == "__main__":
    sys.exit(main())
