"""Fase 4, seconda meta' — parita' dei ``messages`` fra i due bracci.

``schema_gate`` verifica il campo ``tools``. Questo verifica l'altra
meta' dell'input: la conversazione. Confronta due tracce dello stesso
compito, una per braccio, e stabilisce se il modello ha ricevuto lo
stesso contesto a ogni giro.

Il confronto non puo' essere letterale, per tre ragioni tutte prive di
significato semantico:

* gli **identificativi di chiamata** sono generati casualmente a ogni
  esecuzione, quindi differiscono sempre;
* ``content: null`` e assenza del campo esprimono la stessa cosa;
* la traccia LangChain e' una **ricostruzione** dei messaggi ottenuta dal
  callback, non il payload letterale inviato sul filo: puo' quindi
  contenere campi in piu' — per esempio ``name`` sui messaggi di
  strumento — che il framework non necessariamente trasmette.

La normalizzazione rimuove esattamente queste tre categorie e nulla
altro. Resta come prova indipendente il **conteggio dei token in
ingresso** riportato dall'endpoint: se coincide, il modello ha ricevuto
lo stesso input a prescindere da come le due tracce lo rappresentano. E'
il criterio piu' solido dei due, perche' non dipende dal nostro codice di
registrazione.

Uso::

    uv run python -m harness.messages_gate --task t1_conteggio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TRACES_DIR = Path(__file__).resolve().parent.parent / "results" / "traces"

#: Tolleranza sul conteggio dei token.
#:
#: L'uguaglianza esatta non e' raggiungibile, e non per un difetto del
#: banco di prova. Gli identificativi delle chiamate a strumento sono
#: generati casualmente dall'endpoint e compaiono nella conversazione;
#: pur avendo sempre la stessa lunghezza, sequenze esadecimali diverse si
#: segmentano in un numero diverso di token. Misurato: due esecuzioni
#: dello **stesso** braccio, con messaggi identici carattere per carattere
#: (1785 in entrambe), hanno prodotto 1105 e 1122 token in ingresso.
#:
#: La tolleranza copre quindi una variabilita' che esiste anche fra due
#: esecuzioni identiche, e non maschera differenze di contenuto: quelle
#: le intercetta il confronto dei messaggi normalizzati.
#: Misurata: su contenuto identico carattere per carattere, il conteggio
#: dei token ha oscillato fra 1089 e 1122 — trentatre' token, il 3%. La
#: tolleranza e' calibrata su quella variabilita' osservata e non scelta
#: a priori, cosi' che non possa mascherare differenze reali: quelle le
#: intercetta il confronto esatto sul contenuto, qui sotto.
TOLLERANZA = 0.04
TOLLERANZA_ASSOLUTA = 50

OK = "✓"
KO = "✗"


def normalize(message: dict[str, Any]) -> dict[str, Any]:
    """Riduce un messaggio alla sua sostanza semantica."""
    out: dict[str, Any] = {"role": message.get("role")}

    content = message.get("content")
    if content:  # ``None`` e stringa vuota sono equivalenti
        out["content"] = content

    calls = message.get("tool_calls") or []
    if calls:
        out["tool_calls"] = [
            {
                "name": c.get("function", {}).get("name"),
                # Gli argomenti arrivano come stringa JSON: si confronta
                # la struttura, non la formattazione.
                "arguments": _parse(c.get("function", {}).get("arguments")),
            }
            for c in calls
        ]
    return out


def impronta(messages: list[dict[str, Any]]) -> str:
    """Serializza i messaggi escludendo cio' che e' generato a caso.

    E' il criterio **esatto** del gate. Gli identificativi delle chiamate
    a strumento cambiano a ogni esecuzione e non hanno significato; tolti
    quelli, due bracci che presentano al modello lo stesso contesto devono
    produrre la stessa identica stringa. Il conteggio dei token, per
    contro, oscilla anche a contenuto invariato, perche' dipende da come
    si segmentano quegli identificativi.
    """
    puliti = []
    for m in messages:
        c = {k: v for k, v in m.items() if k not in ("tool_call_id",)}
        if c.get("tool_calls"):
            c["tool_calls"] = [
                {k: v for k, v in call.items() if k != "id"}
                for call in c["tool_calls"]
            ]
        if not c.get("content"):
            c.pop("content", None)
        puliti.append(c)
    return json.dumps(puliti, sort_keys=True, ensure_ascii=False)


def _parse(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return raw
    return raw


def latest(pattern: str) -> Path | None:
    found = sorted(TRACES_DIR.glob(pattern))
    return found[-1] if found else None


def compare(mcp: dict[str, Any], lc: dict[str, Any]) -> bool:
    ok = True
    calls = list(zip(mcp["llm_calls"], lc["llm_calls"]))

    if len(mcp["llm_calls"]) != len(lc["llm_calls"]):
        print(
            f"{KO} numero di interrogazioni diverso: "
            f"mcp={len(mcp['llm_calls'])} lc={len(lc['llm_calls'])}"
        )
        ok = False

    for i, (a, b) in enumerate(calls, 1):
        ra, rb = a["request"], b["request"]
        print(f"\n--- interrogazione {i} ---")

        ta, tb = a["prompt_tokens"], b["prompt_tokens"]
        if ta is None or tb is None:
            print(f"  {KO} token in ingresso non riportati: mcp={ta} lc={tb}")
            ok = False
        elif ta == tb:
            print(f"  {OK} token in ingresso identici: {ta}")
        elif abs(ta - tb) <= max(TOLLERANZA_ASSOLUTA, TOLLERANZA * max(ta, tb)):
            print(
                f"  {OK} token in ingresso entro tolleranza: mcp={ta} lc={tb} "
                f"(scarto {abs(ta - tb)})"
            )
        else:
            print(f"  {KO} token in ingresso: mcp={ta} lc={tb} (scarto {abs(ta-tb)})")
            ok = False

        same_tools = json.dumps(ra.get("tools"), sort_keys=True) == json.dumps(
            rb.get("tools"), sort_keys=True
        )
        print(f"  {OK if same_tools else KO} strumenti identici")
        ok = ok and same_tools

        ia, ib = impronta(ra.get("messages", [])), impronta(rb.get("messages", []))
        if ia == ib:
            print(f"  {OK} contenuto identico ({len(ia)} caratteri, id esclusi)")
        else:
            print(f"  {KO} contenuto diverso: mcp={len(ia)} lc={len(ib)} caratteri")
            ok = False

        ma = [normalize(m) for m in ra.get("messages", [])]
        mb = [normalize(m) for m in rb.get("messages", [])]
        if ma == mb:
            print(f"  {OK} messaggi equivalenti ({len(ma)})")
        else:
            print(f"  {KO} messaggi divergenti")
            ok = False
            for j, (x, y) in enumerate(zip(ma, mb)):
                if x != y:
                    print(f"      msg[{j}] mcp: {json.dumps(x, ensure_ascii=False)[:150]}")
                    print(f"             lc : {json.dumps(y, ensure_ascii=False)[:150]}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="t1_conteggio")
    parsed = parser.parse_args()

    # Si confronta ``mcp-conforme``, non ``mcp``. La parita' del contesto
    # e' cio' che quella condizione realizza per definizione; il braccio
    # idiomatico diverge di proposito, e passarlo qui produrrebbe un
    # fallimento che non segnala nulla.
    a = latest(f"mcp-conforme__*{parsed.task}*.json")
    b = latest(f"langchain__*{parsed.task}*.json")
    if not a or not b:
        print(f"{KO} servono due tracce per il compito {parsed.task}, una per braccio.")
        print("  Eseguire prima:")
        print(f"    uv run python -m arm_mcp.host --task {parsed.task} --conforme")
        print(f"    uv run python -m arm_langchain.agent --task {parsed.task}")
        return 1

    print(f"MCP conf.: {a.name}")
    print(f"LangChain: {b.name}")

    ok = compare(json.loads(a.read_text()), json.loads(b.read_text()))
    print()
    if ok:
        print(f"{OK} I due bracci hanno presentato al modello lo stesso contesto.")
        return 0
    print(f"{KO} Divergenza rilevata: le differenze misurate non sarebbero attribuibili.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
