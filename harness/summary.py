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

**Le serie sono incolonnate a passo fisso**, e le esecuzioni escluse
lasciano un vuoto anziche' accorciare la riga. Vale la stessa ragione
delle due scelte precedenti: se la n-esima ripetizione non sta alla
stessa ascissa in ogni colonna e in entrambi i bracci, riportare i
valori uno per ripetizione non serve a nulla, perche' non si possono
mettere in corrispondenza. Con le celle allineate a destra e una
esecuzione tolta, il buco compariva all'inizio della riga e faceva
leggere come mancante la prima ripetizione invece di quella davvero
interrotta.

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

#: Un carattere per esito, cosi' che la colonna resti larga quanto il
#: numero di ripetizioni e si legga **posizione per posizione** accanto
#: alle colonne numeriche. Prima l'esito compariva come conteggio in
#: ordine di frequenza — ``3×ok, stato_non_modificato, risposta_errata``
#: — che oltre a dilatare la tabella era in un ordine *diverso* da quello
#: delle altre colonne: non si poteva sapere quale ripetizione fosse
#: fallita, cioe' proprio la cosa per cui i valori sono riportati uno per
#: ripetizione anziche' come mediana.
SIMBOLI = {
    "ok": "·",
    "risposta_errata": "R",
    "stato_non_modificato": "S",
    "limite_iterazioni": "L",
    "errore_llm": "!",
    "errore_agente": "!",
}

LEGENDA = {
    "·": "compito risolto",
    "R": "risposta errata",
    "S": "stato del servizio non come atteso",
    "L": "limite di iterazioni raggiunto",
    "!": "guasto dell'endpoint o della rete",
    "?": "esito non riconosciuto",
}


class Serie(list):
    """Valori di ogni ripetizione, in ordine di esecuzione.

    E' una lista marcata: ``tabella`` la riconosce e incolonna i valori
    su larghezza fissa, cosi' che la n-esima ripetizione stia alla stessa
    ascissa in ogni colonna e le righe dei due bracci si confrontino a
    vista. Separati da virgola e di larghezza variabile — ``13026,5506``
    accanto a ``2,2`` — quei numeri non erano leggibili.
    """


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


def _arrotonda(valore: Any, cifre: int) -> Any:
    if valore is None:
        return None
    return round(valore, cifre) if cifre else int(round(valore))


def simbolo(traccia: dict[str, Any]) -> str:
    """Un carattere per l'esito di una esecuzione.

    Non si collassa su riuscito/fallito. ``risposta_errata`` e
    ``stato_non_modificato`` descrivono il comportamento del modello,
    ``limite_iterazioni`` il ciclo, gli errori l'infrastruttura: sono cose
    diverse e ridurle a un booleano cancellerebbe proprio l'informazione
    per cui vale la pena guardare una campagna.
    """
    return SIMBOLI.get(traccia.get("status") or "?", "?")


def legenda(tracce: list[dict[str, Any]]) -> str:
    """Spiega i soli simboli che compaiono davvero."""
    presenti = Counter(SIMBOLI.get(t.get("status") or "?", "?") for t in tracce)
    voci = [
        f"`{s}` {LEGENDA[s]} ({presenti[s]})"
        for s in LEGENDA
        if presenti.get(s)
    ]
    return "  ·  ".join(voci)


def stato_verificato(
    tracce: list[dict[str, Any]], ripetizioni: list[int]
) -> Serie | str:
    """Esito del controllo sullo stato del servizio, per i compiti che scrivono."""
    if all((t.get("config") or {}).get("state_verified") is None for t in tracce):
        return VUOTO

    def reso(t: dict[str, Any]) -> Any:
        v = (t.get("config") or {}).get("state_verified")
        return None if v is None else ("si" if v else "NO")

    return per_ripetizione(tracce, reso, ripetizioni)


#: Su cosa ripiegare quando l'endpoint non dichiara ``usage``.
_RIPIEGO_TOKEN = {
    "prompt_tokens": "chars_sent",
    "completion_tokens": "chars_received",
}


def token(traccia: dict[str, Any], chiave: str) -> Any:
    """Token riportati dall'endpoint per una esecuzione, o i caratteri.

    Se l'endpoint non fornisce ``usage``, ``metrics`` lascia i token a
    ``None`` e conserva il numero di caratteri. Ripiegare su quelli e'
    meglio che stampare una casella vuota, purche' sia dichiarato: la cella
    porta allora il suffisso ``c``, e le due unita' non vanno confrontate
    fra loro.
    """
    valore = _metrica(traccia, chiave)
    if valore is not None:
        return valore
    ripiego = _metrica(traccia, _RIPIEGO_TOKEN[chiave])
    return None if ripiego is None else f"{ripiego}c"


def _testo(valore: Any) -> str:
    return VUOTO if valore is None else str(valore)


def _incolonna(intestazioni: list[str], righe: list[list[Any]]) -> None:
    """Rende ogni ``Serie`` una stringa a passo fisso, colonna per colonna.

    La larghezza si calcola sull'intera colonna, non sulla singola cella:
    e' la condizione perche' la n-esima ripetizione stia alla stessa
    ascissa nelle righe dei due bracci, che e' l'unico modo di
    confrontarle a vista.
    """
    for i in range(len(intestazioni)):
        serie_colonna = [r[i] for r in righe if isinstance(r[i], Serie)]
        if not serie_colonna:
            continue
        larghezza = max(
            (len(_testo(v)) for cella in serie_colonna for v in cella), default=1
        )
        for r in righe:
            if isinstance(r[i], Serie):
                r[i] = (
                    " ".join(_testo(v).rjust(larghezza) for v in r[i])
                    if len(r[i])
                    else VUOTO
                )


def tabella(intestazioni: list[str], righe: list[list[Any]]) -> str:
    """Compone una tabella Markdown allineando le colonne nel sorgente.

    L'allineamento e' superfluo una volta reso in HTML, ma il file viene
    letto quasi sempre cosi' com'e', in un editor o in un terminale.

    Le colonne numeriche e quelle di tipo ``Serie`` si allineano a
    destra, le altre a sinistra: incolonnare le cifre e' cio' che rende
    confrontabili a vista due righe adiacenti.
    """
    righe = [list(r) for r in righe]
    seriali = {
        i for i in range(len(intestazioni)) if any(isinstance(r[i], Serie) for r in righe)
    }
    _incolonna(intestazioni, righe)
    larghezze = [
        max(len(str(r[i])) for r in [intestazioni, *righe])
        for i in range(len(intestazioni))
    ]

    def riga(celle: list[Any], intesta: bool = False) -> str:
        rese = []
        for i, c in enumerate(celle):
            testo = str(c)
            allinea = testo.rjust if (i in seriali and not intesta) else testo.ljust
            rese.append(allinea(larghezze[i]))
        return "| " + " | ".join(rese) + " |"

    separatore = "|" + "|".join(
        ("-" * (w + 1) + ":") if i in seriali else "-" * (w + 2)
        for i, w in enumerate(larghezze)
    ) + "|"
    return "\n".join(
        [riga(intestazioni, intesta=True), separatore, *(riga(r) for r in righe)]
    )


def _ripetizioni(
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]], modello: str
) -> list[int]:
    """Le ripetizioni previste, ricavate da quelle osservate."""
    viste = {
        _rep(t) for (m, _, _), v in gruppi.items() if m == modello for t in v
    }
    return sorted(viste)


def per_ripetizione(
    tracce: list[dict[str, Any]], valore: Any, ripetizioni: list[int]
) -> Serie:
    """Colloca ogni valore alla posizione della **sua** ripetizione.

    Costruire la serie scorrendo le tracce presenti sembra equivalente e
    non lo e': quando una esecuzione manca del tutto — la campagna e'
    stata interrotta su quella cella, o e' morta senza salvare — la riga
    si accorcia, e poiche' le celle sono allineate a destra il buco
    compare **all'inizio**. Con la quinta ripetizione mancante la riga
    mostrava quattro valori e faceva leggere come assente la prima.

    E' la stessa correzione gia' applicata alle esecuzioni escluse per
    guasto, estesa a quelle che non esistono affatto: in entrambi i casi
    la n-esima posizione deve restare la n-esima ripetizione.
    """
    per_rep = {_rep(t): valore(t) for t in tracce}
    return Serie(per_rep.get(r) for r in ripetizioni)


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
    reps = _ripetizioni(gruppi, modello)
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
                    per_ripetizione(tracce, simbolo, reps),
                    stato_verificato(tracce, reps),
                    per_ripetizione(tracce, lambda t: _metrica(t, "n_llm_calls"), reps),
                    per_ripetizione(tracce, lambda t: _metrica(t, "n_tool_calls"), reps),
                    per_ripetizione(tracce, lambda t: _metrica(t, "n_rest_calls"), reps),
                ]
            )
    return tabella(
        ["compito", "braccio", "esiti", "stato", "iteraz.", "strum.", "REST"], righe
    )


def _compromessa(traccia: dict[str, Any]) -> bool:
    """Se i conteggi dell'esecuzione siano parziali senza dichiararlo.

    Una traccia con due interrogazioni di cui la seconda andata in
    timeout somma i token della sola prima, e il totale sembra completo.
    Osservato: 579 token accanto ai 1668 degli altri bracci, che avrebbe
    suggerito un consumo tre volte inferiore laddove l'esecuzione era
    semplicemente morta a meta'.
    """
    return traccia.get("status") in INFRASTRUTTURALI


def _righe_costo(
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]],
    modello: str,
    colonne: list[Any],
) -> list[list[Any]]:
    """Costruisce le righe di una tabella di costo, due bracci per compito.

    Ogni ``colonne`` e' una funzione di **una** traccia. Le esecuzioni
    compromesse da un guasto lasciano un vuoto, e cosi' quelle che non
    esistono affatto: in entrambi i casi la posizione nella riga resta
    quella della ripetizione, che e' la sola cosa che rende le due righe
    di un compito confrontabili a vista.
    """
    righe = []
    reps = _ripetizioni(gruppi, modello)
    for compito in sorted({c for (m, c, _) in gruppi if m == modello}):
        for braccio in ARMS:
            tracce = gruppi.get((modello, compito, braccio), [])
            if not tracce or all(_compromessa(t) for t in tracce):
                continue

            def cella(f: Any, tracce: list[dict[str, Any]] = tracce) -> Serie:
                return per_ripetizione(
                    tracce, lambda t: None if _compromessa(t) else f(t), reps
                )

            righe.append(
                [compito if braccio == ARMS[0] else "", braccio]
                + [cella(f) for f in colonne]
            )
    return righe


def sezione_token(
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]], modello: str
) -> str:
    """Quanto contesto i due bracci fanno leggere e scrivere al modello.

    Sta in una tabella propria. Insieme ai tempi erano otto colonne di
    serie numeriche, e nessuna delle due domande — *quanto costa in
    token* e *quanto e' durata* — si riusciva a leggere: le cifre dei
    token sono a quattro o cinque caratteri e schiacciavano tutto il
    resto.
    """
    return tabella(
        ["compito", "braccio", "tok in", "tok out"],
        _righe_costo(
            gruppi,
            modello,
            [
                lambda t: token(t, "prompt_tokens"),
                lambda t: token(t, "completion_tokens"),
            ],
        ),
    )


def sezione_tempo(
    gruppi: dict[tuple[str, str, str], list[dict[str, Any]]], modello: str
) -> str:
    """Il tempo dell'esecuzione, e quanto ne e' andato perso.

    ``dispatch`` e' il tempo speso a invocare gli strumenti, ed e' l'unica
    colonna che si ricollega all'esperimento A. ``lat. LLM`` misura invece
    l'attesa dell'endpoint remoto, che dipende dal suo carico e non dal
    braccio: e' contesto, non un risultato, e va letta insieme ai
    ritentativi che la producono.

    ``persi s`` e' il tempo speso in tentativi andati a vuoto, che ``lat.
    LLM`` non contiene perche' quella somma i soli tentativi riusciti.
    Senza questa colonna un'esecuzione durata dodici minuti di orologio
    comparirebbe in tabella con sei secondi di latenza, come e' accaduto.
    """
    return tabella(
        ["compito", "braccio", "dispatch ms", "ritent.", "persi s", "lat. LLM s"],
        _righe_costo(
            gruppi,
            modello,
            [
                lambda t: _arrotonda(_metrica(t, "latency_tools_ms"), 1),
                lambda t: _falliti(t, "n_retries"),
                lambda t: _arrotonda(_falliti(t, "latency_retries_s"), 0),
                lambda t: _arrotonda(_metrica(t, "latency_llm_s"), 1),
            ],
        ),
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
        # Le due domande vengono separate. *Hanno fatto le stesse cose?*
        # si risponde per uguaglianza fra interi piccoli, e quando la
        # risposta e' no serve vedere i valori di ogni ripetizione.
        # *Costano lo stesso?* si risponde su una grandezza che oscilla,
        # quindi con una mediana e una soglia, ed e' la stessa domanda per
        # tutti i compiti: sta percio' in una tabella sola, con i compiti
        # per riga.
        #
        # Tenute insieme, producevano o un elenco piatto di venti voci
        # tutte della stessa forma, o sette tabelle da una riga ciascuna.
        comportamento: list[str] = []
        concordi: list[str] = []
        righe_token: list[list[Any]] = []
        sotto_soglia: list[str] = []

        for compito in compiti:
            a, b = usabili[(compito, primo)], usabili[(compito, secondo)]
            if not a or not b:
                continue

            # Numeri diversi di esecuzioni utilizzabili rendono il
            # confronto privo di senso: due valori contro uno differiscono
            # per costruzione, e segnalarlo come divergenza attribuirebbe
            # all'approccio quello che e' stato un guasto di rete.
            if len(a) != len(b):
                comportamento.append(
                    f"- **{compito}** — non confrontabile: {len(a)} esecuzioni "
                    f"utilizzabili per {primo}, {len(b)} per {secondo}."
                )
                continue

            righe = []
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
                    righe.append([etichetta, Serie(va), Serie(vb)])
            if righe:
                comportamento.append(
                    f"**{compito}** — valori ordinati, non in ordine di "
                    f"esecuzione\n\n" + tabella(["", primo, secondo], righe)
                )
            else:
                concordi.append(compito)

            ma, mb = _mediana(a, "prompt_tokens"), _mediana(b, "prompt_tokens")
            if ma is None or mb is None:
                continue
            if _rilevante(ma, mb):
                delta = mb - ma
                righe_token.append(
                    [
                        compito,
                        Serie([f"{ma:.0f}"]),
                        Serie([f"{mb:.0f}"]),
                        Serie([f"{delta:+.0f}"]),
                        Serie([f"{delta / ma:+.1%}"]),
                    ]
                )
            else:
                sotto_soglia.append(compito)

        parti = [f"**{primo} contro {secondo}** — {spiegazione}\n"]

        parti.append("#### Comportamento\n")
        parti.append(
            "Iterazioni, chiamate a strumento, chiamate REST: le tre grandezze\n"
            "su cui la tesi si pronuncia, perche' sono le sole indipendenti dal\n"
            "carico dell'endpoint.\n"
        )
        if comportamento:
            parti.extend(comportamento)
            if concordi:
                parti.append(
                    "Stessi valori nei due bracci su "
                    + ", ".join(f"`{c}`" for c in concordi)
                    + "."
                )
        else:
            parti.append("**Nessuna divergenza in alcun compito.**")

        parti.append("\n#### Contesto inviato al modello\n")
        parti.append(
            "Token in ingresso, mediana sulle ripetizioni. Il conteggio oscilla\n"
            f"anche a contenuto invariato, quindi si riporta solo lo scarto oltre\n"
            f"{SOGLIA_TOKEN_ASSOLUTA} token e {SOGLIA_TOKEN_RELATIVA:.0%}.\n"
        )
        if righe_token:
            parti.append(
                tabella(["compito", primo, secondo, "scarto", "rel."], righe_token)
            )
            if sotto_soglia:
                parti.append(
                    "Sotto la soglia su "
                    + ", ".join(f"`{c}`" for c in sotto_soglia)
                    + "."
                )
        else:
            parti.append("Nessuno scarto oltre la soglia.")

        blocchi.append("\n\n".join(p.strip("\n") for p in parti))

    return "\n\n".join(blocchi)


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
            [t["task_id"], t["arm"], f"rep{_rep(t)}", t.get("status") or "?", dettaglio]
        )

    # La colonna del dettaglio esiste solo per i guasti, che sono pochi:
    # quando nessuno ne ha uno resta una colonna di trattini larga quanto
    # l'intestazione, e va tolta.
    if not any(r[4] for r in righe):
        return tabella(
            ["compito", "braccio", "rip.", "esito"], [r[:4] for r in righe]
        )
    for r in righe:
        r[4] = r[4] or VUOTO
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
        del_modello = [t for t in tracce if t["model"] == modello]
        parti.append(f"\n## {modello}\n")

        parti.append("### Comportamento dell'agente\n")
        parti.append(
            "Un valore per ripetizione, nell'ordine di esecuzione: la n-esima\n"
            "posizione e' la stessa esecuzione in ogni colonna, e le due righe\n"
            "di uno stesso compito si leggono affiancate.\n"
        )
        parti.append(sezione_comportamento(gruppi, modello))
        parti.append(f"\n{legenda(del_modello)}\n")

        parti.append("\n### Costo in token\n")
        parti.append(
            "Contati dall'endpoint, non da noi. Le esecuzioni interrotte da un\n"
            "guasto sono escluse: i loro totali sono parziali senza dirlo.\n"
        )
        parti.append(sezione_token(gruppi, modello))

        parti.append("\n### Tempi\n")
        parti.append(
            "`dispatch ms` e' il tempo di invocazione degli strumenti, l'unica\n"
            "colonna confrontabile con l'esperimento A. `lat. LLM s` misura\n"
            "l'attesa dell'endpoint remoto: dipende dal suo carico, non dal\n"
            "braccio, e va letta insieme a `ritent.` e `persi s`, che sono i\n"
            "tentativi andati a vuoto e il tempo che sono costati.\n"
        )
        parti.append(sezione_tempo(gruppi, modello))

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
