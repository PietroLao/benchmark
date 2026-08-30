"""Fase 6b — riepilogo leggibile di una campagna.

Una traccia contiene tutto quello che serve a ricostruire un'esecuzione:
i messaggi, gli schemi, le risposte integrali. E' la sua ragione d'essere,
ed e' anche il motivo per cui non si puo' leggere. Trenta file da qualche
decina di migliaia di caratteri non rispondono alla domanda per cui la
campagna e' stata eseguita, che e' sempre la stessa: **i bracci si
comportano diversamente, e dove.**

Questo modulo estrae quella risposta e la scrive in un unico file
Markdown accanto alle tracce.

Due scelte di presentazione, entrambe deliberate.

**Si riportano i valori di ogni ripetizione, non la loro mediana.** Una
mediana avrebbe nascosto il risultato piu' importante ottenuto finora: la
divergenza nei token era visibile perche' le tre ripetizioni stavano una
accanto all'altra e mostravano 100,103,103 contro 38,34,38 — una
regolarita' che nessun indice sintetico avrebbe reso, e che con tre
ripetizioni sarebbe stata indistinguibile dal rumore.

**I bracci stanno su righe adiacenti**, non in tabelle separate. Il
confronto e' l'oggetto della misura, non un esercizio lasciato a chi
legge.

Uso::

    uv run python -m harness.summary
    uv run python -m harness.summary --dir results/campagna
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.trace import tentativi_falliti

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ARMS = ("mcp", "langchain")

#: L'unico confronto, e cio' che isola. Resta in forma di tabella perche'
#: il disegno ha gia' ospitato piu' condizioni e potrebbe ospitarne
#: ancora.
CONFRONTI: tuple[tuple[str, str, str], ...] = (
    (
        "mcp",
        "langchain",
        "i due approcci, ciascuno nella forma idiomatica del proprio ecosistema",
    ),
)

#: Esiti dovuti all'infrastruttura e non al comportamento dell'agente. Un
#: 503 dell'endpoint o un timeout di rete non dicono nulla su MCP o su
#: LangChain: vanno contati a parte e mai aggregati con gli altri, perche'
#: aggregarli significherebbe attribuire a un approccio la salute del
#: fornitore.
INFRASTRUTTURALI = {"errore_llm", "errore_agente"}

VUOTO = "—"


# --- lettura --------------------------------------------------------------


def carica(directory: Path) -> list[dict[str, Any]]:
    """Legge le tracce, ignorando i file di servizio (prefisso ``_``)."""
    tracce = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            tracce.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"  ! traccia illeggibile, saltata: {path.name}", file=sys.stderr)
    return tracce


def _metrica(traccia: dict[str, Any], chiave: str) -> Any:
    return (traccia.get("metrics") or {}).get(chiave)


def _rep(traccia: dict[str, Any]) -> int:
    return (traccia.get("config") or {}).get("repetition") or 0


def _falliti(traccia: dict[str, Any], chiave: str) -> Any:
    """Legge un dato sui tentativi a vuoto **ricalcolandolo dagli eventi**.

    Non si usa il valore in ``metrics``: le tracce scritte prima della
    correzione contengono un ``n_retries`` che ignorava i timeout, e una
    di esse riportava zero ritentativi per un'esecuzione durata dodici
    minuti. Gli eventi, invece, sono sempre stati registrati per intero,
    quindi ricalcolare qui rende corrette anche le tracce vecchie senza
    doverle rieseguire.
    """
    return tentativi_falliti(traccia.get("events") or []).get(chiave)


def _mediana(tracce: list[dict[str, Any]], chiave: str) -> float | None:
    valori = sorted(
        v for v in (_metrica(t, chiave) for t in tracce) if v is not None
    )
    if not valori:
        return None
    meta = len(valori) // 2
    if len(valori) % 2:
        return float(valori[meta])
    return (valori[meta - 1] + valori[meta]) / 2


#: Soglie sotto le quali uno scarto fra conteggi di token non viene
#: riportato. Il conteggio oscilla anche a contenuto invariato, perche'
#: dipende da come si segmentano gli identificativi di chiamata generati a
#: caso: misurato, fino a trentatre' token su contenuto identico carattere
#: per carattere. Riportare scarti di quell'ordine significherebbe
#: annunciare rumore come risultato.
SOGLIA_TOKEN_ASSOLUTA = 40
SOGLIA_TOKEN_RELATIVA = 0.01


def _rilevante(a: float, b: float) -> bool:
    scarto = abs(b - a)
    return scarto >= SOGLIA_TOKEN_ASSOLUTA and scarto >= SOGLIA_TOKEN_RELATIVA * a


def raggruppa(
    tracce: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Indicizza per modello, compito e braccio, ordinando per ripetizione."""
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in tracce:
        gruppi[(t["model"], t["task_id"], t["arm"])].append(t)
    for v in gruppi.values():
        v.sort(key=_rep)
    return gruppi


# --- formattazione --------------------------------------------------------


def serie(valori: list[Any]) -> str:
    """Valori di tutte le ripetizioni, nell'ordine in cui sono state svolte."""
    if not valori or all(v is None for v in valori):
        return VUOTO
    return ",".join(VUOTO if v is None else str(v) for v in valori)


def serie_arrotondata(valori: list[Any], cifre: int = 1) -> str:
    return serie([None if v is None else round(v, cifre) for v in valori])


def esiti(tracce: list[dict[str, Any]]) -> str:
    """Riassume gli stati come conteggio, in ordine di frequenza.

    Non si collassa su riuscito/fallito. ``risposta_errata`` e
    ``stato_non_modificato`` descrivono il comportamento del modello,
    ``limite_iterazioni`` il ciclo, gli errori l'infrastruttura: sono cose
    diverse e ridurle a un booleano cancellerebbe proprio l'informazione
    per cui vale la pena guardare una campagna.
    """
    conteggio = Counter(t.get("status") or "?" for t in tracce)
    return ", ".join(
        f"{n}×{stato}" if n > 1 else stato for stato, n in conteggio.most_common()
    )


def stato_verificato(tracce: list[dict[str, Any]]) -> str:
    """Esito del controllo sullo stato del servizio, per i compiti che scrivono."""
    valori = [(t.get("config") or {}).get("state_verified") for t in tracce]
    if all(v is None for v in valori):
        return VUOTO
    return serie([None if v is None else ("si" if v else "NO") for v in valori])


def token(tracce: list[dict[str, Any]], chiave: str) -> str:
    """Token riportati dall'endpoint, o caratteri quando non li riporta.

    Se l'endpoint non fornisce ``usage``, ``metrics`` lascia i token a
    ``None`` e conserva il numero di caratteri. Ripiegare su quelli e'
    meglio che stampare una casella vuota, purche' sia dichiarato: la cella
    porta allora il suffisso ``c``, e le due unita' non vanno confrontate
    fra loro.
    """
    fuori = {"prompt_tokens": "chars_sent", "completion_tokens": "chars_received"}
    celle = []
    for t in tracce:
        valore = _metrica(t, chiave)
        if valore is None:
            ripiego = _metrica(t, fuori[chiave])
            celle.append(None if ripiego is None else f"{ripiego}c")
        else:
            celle.append(valore)
    return serie(celle)


def tabella(intestazioni: list[str], righe: list[list[str]]) -> str:
    """Compone una tabella Markdown allineando le colonne nel sorgente.

    L'allineamento e' superfluo una volta reso in HTML, ma il file viene
    letto quasi sempre cosi' com'e', in un editor o in un terminale.
    """
    larghezze = [
        max(len(str(r[i])) for r in [intestazioni, *righe])
        for i in range(len(intestazioni))
    ]
    def riga(celle: list[str]) -> str:
        return "| " + " | ".join(
            str(c).ljust(larghezze[i]) for i, c in enumerate(celle)
        ) + " |"

    separatore = "|" + "|".join("-" * (w + 2) for w in larghezze) + "|"
    return "\n".join([riga(intestazioni), separatore, *(riga(r) for r in righe)])


# --- sezioni --------------------------------------------------------------


def sezione_comportamento(
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]], modello: str
) -> str:
    """Le metriche che l'esperimento B rivendica.

    Iterazioni, chiamate a strumento e chiamate REST descrivono *cosa ha
    fatto* l'agente. Sono le uniche tre grandezze su cui la tesi si
    pronuncia, perche' sono le sole indipendenti dal carico dell'endpoint.
    """
    righe = []
    compiti = sorted({c for (m, c, _) in gruppi if m == modello})
    for compito in compiti:
        for braccio in ARMS:
            tracce = gruppi.get((modello, compito, braccio), [])
            if not tracce:
                continue
            righe.append(
                [
                    compito if braccio == ARMS[0] else "",
                    braccio,
                    esiti(tracce),
                    stato_verificato(tracce),
                    serie([_metrica(t, "n_llm_calls") for t in tracce]),
                    serie([_metrica(t, "n_tool_calls") for t in tracce]),
                    serie([_metrica(t, "n_rest_calls") for t in tracce]),
                ]
            )
    return tabella(
        ["compito", "braccio", "esiti", "stato", "iteraz.", "strum.", "REST"], righe
    )


def sezione_costo(
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]], modello: str
) -> str:
    """Il costo dell'esecuzione.

    ``dispatch`` e' il tempo speso a invocare gli strumenti, ed e' l'unica
    colonna che si ricollega all'esperimento A. ``lat. LLM`` misura invece
    l'attesa dell'endpoint remoto, che dipende dal suo carico e non dal
    braccio: e' contesto, non un risultato, e va letta insieme ai
    ritentativi che la producono.

    ``persi s`` e' il tempo speso in tentativi andati a vuoto, che ``lat.
    LLM`` non contiene perche' quella somma i soli tentativi riusciti.
    Senza questa colonna un'esecuzione durata dodici minuti di orologio
    comparirebbe in tabella con sei secondi di latenza, come e' accaduto.

    Le esecuzioni interrotte da guasti dell'infrastruttura sono escluse:
    i loro conteggi sono parziali senza dichiararlo. Compaiono nella
    tabella del comportamento, dove la colonna degli esiti le rende
    visibili, e nell'elenco in fondo.
    """
    righe = []
    compiti = sorted({c for (m, c, _) in gruppi if m == modello})
    for compito in compiti:
        for braccio in ARMS:
            # Si escludono le esecuzioni interrotte da guasti. I loro
            # conteggi sono parziali e non lo dichiarano: una traccia con
            # due interrogazioni di cui la seconda andata in timeout somma
            # i token della sola prima, e il totale sembra completo.
            # Osservato: 579 token accanto ai 1668 degli altri bracci, che
            # avrebbe suggerito un consumo tre volte inferiore laddove
            # l'esecuzione era semplicemente morta a meta'.
            tracce = [
                t
                for t in gruppi.get((modello, compito, braccio), [])
                if t.get("status") not in INFRASTRUTTURALI
            ]
            if not tracce:
                continue
            righe.append(
                [
                    compito if braccio == ARMS[0] else "",
                    braccio,
                    token(tracce, "prompt_tokens"),
                    token(tracce, "completion_tokens"),
                    serie_arrotondata([_metrica(t, "latency_tools_ms") for t in tracce]),
                    serie([_falliti(t, "n_retries") for t in tracce]),
                    serie_arrotondata(
                        [_falliti(t, "latency_retries_s") for t in tracce], 0
                    ),
                    serie_arrotondata([_metrica(t, "latency_llm_s") for t in tracce]),
                ]
            )
    return tabella(
        [
            "compito",
            "braccio",
            "tok in",
            "tok out",
            "dispatch ms",
            "ritent.",
            "persi s",
            "lat. LLM s",
        ],
        righe,
    )


def sezione_divergenze(
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]], modello: str
) -> str:
    """Dove i bracci hanno fatto cose diverse.

    I confronti sono tre e rispondono a domande distinte: si veda
    ``CONFRONTI``. Riportarne uno solo, o fonderli, farebbe perdere
    l'informazione per cui i bracci sono tre.

    Si confrontano solo le tre metriche di comportamento, e solo sulle
    esecuzioni non compromesse dall'infrastruttura. Il confronto e' fra
    insiemi ordinati di valori e non fra medie: con tre ripetizioni una
    media renderebbe indistinguibili 2,2,2 e 1,2,3.
    """
    compiti = sorted({c for (m, c, _) in gruppi if m == modello})
    usabili = {
        (compito, braccio): [
            t
            for t in gruppi.get((modello, compito, braccio), [])
            if t.get("status") not in INFRASTRUTTURALI
        ]
        for compito in compiti
        for braccio in ARMS
    }

    blocchi = []
    for primo, secondo, spiegazione in CONFRONTI:
        voci = []
        for compito in compiti:
            a, b = usabili[(compito, primo)], usabili[(compito, secondo)]
            if not a or not b:
                continue

            # Numeri diversi di esecuzioni utilizzabili rendono il
            # confronto privo di senso: due valori contro uno differiscono
            # per costruzione, e segnalarlo come divergenza attribuirebbe
            # all'approccio quello che e' stato un guasto di rete.
            if len(a) != len(b):
                voci.append(
                    f"- **{compito}**: non confrontabile, {len(a)} esecuzioni "
                    f"utilizzabili per {primo} contro {len(b)} per {secondo}"
                )
                continue

            for etichetta, chiave in (
                ("iterazioni", "n_llm_calls"),
                ("chiamate a strumento", "n_tool_calls"),
                ("chiamate REST", "n_rest_calls"),
            ):
                va = sorted(
                    v for v in (_metrica(t, chiave) for t in a) if v is not None
                )
                vb = sorted(
                    v for v in (_metrica(t, chiave) for t in b) if v is not None
                )
                if va and vb and va != vb:
                    voci.append(
                        f"- **{compito}**, {etichetta}: "
                        f"{primo} {serie(va)} contro {secondo} {serie(vb)}"
                    )

            # I token in ingresso vanno trattati a parte. Le tre metriche
            # sopra sono interi piccoli e si confrontano per uguaglianza;
            # questi oscillano, quindi si confrontano le mediane e si
            # riporta lo scarto solo quando supera il rumore. Senza questa
            # voce, due bracci che si comportano in modo identico ma
            # presentano al modello un contesto di costo diverso —
            # schemi derivati diversamente, ragionamento rimandato da un
            # lato solo — risulterebbero indistinguibili.
            ma, mb = _mediana(a, "prompt_tokens"), _mediana(b, "prompt_tokens")
            if ma is not None and mb is not None and _rilevante(ma, mb):
                delta = mb - ma
                voci.append(
                    f"- **{compito}**, token in ingresso: {primo} {ma:.0f} "
                    f"contro {secondo} {mb:.0f} "
                    f"({delta:+.0f}, {delta / ma:+.1%})"
                )

        corpo = (
            "\n".join(voci)
            if voci
            else "Nessuna: stessi valori in ogni compito."
        )
        blocchi.append(f"**{primo} contro {secondo}** — {spiegazione}\n\n{corpo}\n")

    return "\n".join(blocchi)


def sezione_fallimenti(tracce: list[dict[str, Any]]) -> str:
    """Elenca per esteso le esecuzioni non riuscite."""
    perse = [t for t in tracce if t.get("status") != "ok"]
    if not perse:
        return "Nessuna: tutte le esecuzioni si sono concluse con esito `ok`.\n"

    righe = []
    for t in sorted(perse, key=lambda x: (x["task_id"], x["arm"], _rep(x))):
        dettaglio = ""
        for evento in t.get("events", []):
            if evento.get("kind") in {"errore_llm", "errore_agente", "limite_iterazioni"}:
                dettaglio = str(evento.get("detail") or evento.get("kind"))[:70]
                break
        righe.append(
            [t["task_id"], t["arm"], f"rep{_rep(t)}", t.get("status") or "?", dettaglio or VUOTO]
        )
    return tabella(["compito", "braccio", "rip.", "esito", "dettaglio"], righe)


# --- composizione ---------------------------------------------------------


def componi(
    tracce: list[dict[str, Any]], contesto: dict[str, Any] | None = None
) -> str:
    """Costruisce il documento completo."""
    contesto = contesto or {}
    gruppi = raggruppa(tracce)
    modelli = sorted({t["model"] for t in tracce})
    ripetizioni = sorted({_rep(t) for t in tracce})
    infra = sum(1 for t in tracce if t.get("status") in INFRASTRUTTURALI)
    riusciti = sum(1 for t in tracce if t.get("status") == "ok")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parti = [
        f"# Riepilogo della campagna\n",
        f"Generato il {stamp}.\n",
    ]

    intestazione = [
        ["modelli", ", ".join(modelli) or VUOTO],
        ["compiti", str(len({t["task_id"] for t in tracce}))],
        ["ripetizioni", str(max(ripetizioni) if ripetizioni else 0)],
        ["bracci", ", ".join(ARMS)],
        ["esecuzioni presenti", str(len(tracce))],
        ["di cui riuscite", f"{riusciti} su {len(tracce)}"],
        ["di cui errori d'infrastruttura", str(infra)],
    ]
    if contesto.get("planned"):
        intestazione.insert(
            4, ["esecuzioni previste", str(contesto["planned"])]
        )
    parti.append(tabella(["", ""], intestazione))

    if infra:
        quante = (
            "Un'esecuzione si e' interrotta"
            if infra == 1
            else f"{infra} esecuzioni si sono interrotte"
        )
        parti.append(
            f"\n> {quante} per errori dell'endpoint o della rete.\n"
            "> Non dicono nulla sui due approcci e sono escluse dal confronto.\n"
        )

    for modello in modelli:
        parti.append(f"\n## {modello}\n")
        parti.append("### Comportamento dell'agente\n")
        parti.append(
            "Un valore per ripetizione, nell'ordine di esecuzione.\n"
        )
        parti.append(sezione_comportamento(gruppi, modello))
        parti.append("\n### Costo\n")
        parti.append(
            "`dispatch ms` e' il tempo di invocazione degli strumenti, l'unica\n"
            "colonna confrontabile con l'esperimento A. `lat. LLM s` misura\n"
            "l'attesa dell'endpoint remoto: dipende dal suo carico, non dal\n"
            "braccio, e va letta insieme ai ritentativi.\n"
        )
        parti.append(sezione_costo(gruppi, modello))
        parti.append("\n### Divergenze fra i bracci\n")
        parti.append(sezione_divergenze(gruppi, modello))

    parti.append("\n## Esecuzioni non riuscite\n")
    parti.append(sezione_fallimenti(tracce))

    parti.append(
        "\n---\n\nLe tracce integrali stanno nei file di questa cartella: "
        "questo riepilogo\ne' derivato e si puo' rigenerare in qualunque "
        "momento con `python -m harness.summary`.\n"
    )
    return "\n".join(parti)


def scrivi(
    directory: Path, contesto: dict[str, Any] | None = None
) -> Path | None:
    """Genera il riepilogo accanto alle tracce e ne restituisce il percorso."""
    tracce = carica(directory)
    if not tracce:
        print(f"Nessuna traccia in {directory}.")
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    percorso = directory / f"_riepilogo_{stamp}.md"
    percorso.write_text(componi(tracce, contesto), encoding="utf-8")
    return percorso


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=RESULTS_DIR / "campagna")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="stampa il riepilogo invece di scriverlo su file",
    )
    parsed = parser.parse_args()

    if not parsed.dir.is_dir():
        print(f"Cartella inesistente: {parsed.dir}")
        return 1

    if parsed.stdout:
        tracce = carica(parsed.dir)
        if not tracce:
            print(f"Nessuna traccia in {parsed.dir}.")
            return 1
        print(componi(tracce))
        return 0

    percorso = scrivi(parsed.dir)
    if percorso is None:
        return 1
    print(f"Riepilogo scritto in {percorso}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
