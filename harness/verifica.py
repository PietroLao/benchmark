"""Mette ogni metrica accanto alla prova grezza da cui è calcolata.

Il riepilogo riporta numeri derivati. Questo modulo serve a controllarli a
mano: per ogni grandezza mostra il valore riportato e, sotto, gli elementi
registrati che lo compongono, uno per riga. Si contano con gli occhi.

La parte che conta davvero è il **riscontro incrociato**. Alcune grandezze
sono misurate due volte, da fonti indipendenti fra loro:

* le invocazioni di strumento sono registrate dal nostro codice, dentro il
  ciclo dell'agente;
* le chiamate REST sono contate dal **server sotto test**, che non sa nulla
  del banco di prova e conta le richieste HTTP che riceve;
* i token sono riportati dall'**endpoint del modello**, non calcolati da
  noi.

Se due fonti indipendenti concordano, l'errore dovrebbe stare in entrambe
allo stesso modo. Se discordano, la differenza ha sempre una spiegazione
precisa — tipicamente una chiamata respinta prima di toccare la rete — e
va capita, non tollerata.

Uso::

    uv run python -m harness.verifica --task t6_iscrizione_condizionale
    uv run python -m harness.verifica --tutte
    uv run python -m harness.verifica --sensibilita
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from shared.tasks import TASKS_BY_ID

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

OK, KO, DUBBIO = "✓", "✗", "?"


def carica(directory: Path) -> list[dict[str, Any]]:
    fuori = []
    for path in sorted(directory.glob("*__*.json")):
        try:
            fuori.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"  ! illeggibile: {path.name}", file=sys.stderr)
    return fuori


# --- verifica di una singola esecuzione -----------------------------------


RAIL = "─" * 74


def _titolo(testo: str) -> None:
    print(f"\n\033[1m{testo}\033[0m" if sys.stdout.isatty() else f"\n{testo}")


def _compatta(valore: Any, larghezza: int) -> str:
    """Riduce un risultato a una riga leggibile, senza JSON indentato."""
    testo = re.sub(r"\s+", " ", str(valore or "")).strip()
    return testo if len(testo) <= larghezza else testo[: larghezza - 1] + "…"


def verifica_traccia(t: dict[str, Any]) -> int:
    """Stampa metriche e prove, e restituisce il numero di incongruenze."""
    m = t["metrics"]
    problemi = 0
    rip = (t.get("config") or {}).get("repetition")

    print(RAIL)
    print(f" {t['arm']} · {t['task_id']} · ripetizione {rip} · esito {t['status']}")
    print(RAIL)

    # --- iterazioni ---
    chiamate = t["llm_calls"]
    ok_iter = m["n_llm_calls"] == len(chiamate)
    problemi += not ok_iter
    _titolo(f" ITERAZIONI · riportate {m['n_llm_calls']}   {OK if ok_iter else KO}")
    print("   una riga per ogni richiesta a /v1/chat/completions")
    print(f"   {'#':>3}  {'messaggi':>8}  {'chiamate':>8}  finish_reason")
    for i, c in enumerate(chiamate, 1):
        msg = ((c.get("response") or {}).get("choices") or [{}])[0]
        n_str = len((msg.get("message") or {}).get("tool_calls") or [])
        print(
            f"   {i:>3}  {len(c['request'].get('messages', [])):>8}  {n_str:>8}  "
            f"{msg.get('finish_reason')}"
        )

    # --- invocazioni di strumento ---
    ok_tool = m["n_tool_calls"] == len(t["tool_calls"])
    problemi += not ok_tool
    _titolo(f" STRUMENTI · riportati {m['n_tool_calls']}   {OK if ok_tool else KO}")
    print(f"   {'#':>3}  {'strumento':<24} {'argomenti':<34} {'esito':<7} risultato")
    for i, c in enumerate(t["tool_calls"], 1):
        arg = _compatta(json.dumps(c.get("arguments"), ensure_ascii=False), 34)
        stato = "ERRORE" if c.get("is_error") else "ok"
        print(
            f"   {i:>3}  {c['name']:<24} {arg:<34} {stato:<7} "
            f"{len(str(c.get('result') or '')):>5} car."
        )

    # --- riscontro incrociato ---
    # L'invariante non e' un'uguaglianza ma un intervallo. Ogni invocazione
    # riuscita ha raggiunto il servizio, quindi il contatore non puo' essere
    # piu' basso delle riuscite; nessuna invocazione produce piu' di una
    # richiesta, quindi non puo' superare le invocate. In mezzo stanno le
    # chiamate fallite *dopo* aver raggiunto il servizio — una registrazione
    # duplicata riceve un 400 e viene contata — mentre quelle respinte
    # *prima* della rete non compaiono affatto nel contatore.
    conteggi = t.get("rest_counts") or {}
    totale = conteggi.get("total")
    riuscite = sum(1 for c in t["tool_calls"] if not c.get("is_error"))
    invocate = len(t["tool_calls"])
    dentro = totale is not None and riuscite <= totale <= invocate
    problemi += not dentro

    _titolo(f" RISCONTRO INCROCIATO   {OK if dentro else KO}")
    print("   due misure indipendenti: il ciclo dell'agente, e il servizio")
    print("   sotto test che non sa di essere misurato")
    print(f"   {'registrate dal ciclo':<34} {invocate} invocate, di cui {riuscite} riuscite")
    print(f"   {'contate dal servizio':<34} {totale}")
    for percorso, n in sorted((conteggi.get("by_endpoint") or {}).items()):
        print(f"   {'':<34}   {n:>2} x {percorso}")
    if dentro:
        print(f"   {'invariante':<34} {riuscite} ≤ {totale} ≤ {invocate}")
        if totale > riuscite:
            print(f"   {'':<34} {totale - riuscite} fallite dopo aver raggiunto il servizio")
        if totale < invocate:
            print(f"   {'':<34} {invocate - totale} respinte prima di toccare la rete")
    else:
        print(f"   {'invariante VIOLATO':<34} atteso {riuscite} ≤ x ≤ {invocate}, trovato {totale}")

    # --- token ---
    somma_in = sum(c["prompt_tokens"] or 0 for c in chiamate)
    somma_out = sum(c["completion_tokens"] or 0 for c in chiamate)
    ok_tok = somma_in == (m["prompt_tokens"] or 0) and somma_out == (
        m["completion_tokens"] or 0
    )
    problemi += not ok_tok
    _titolo(
        f" TOKEN · riportati {m['prompt_tokens']} in ingresso, "
        f"{m['completion_tokens']} in uscita   {OK if ok_tok else KO}"
    )
    print("   non li calcoliamo noi: li dichiara l'endpoint, chiamata per chiamata")
    print(f"   {'#':>3}  {'in':>7}  {'out':>7}")
    for i, c in enumerate(chiamate, 1):
        print(f"   {i:>3}  {str(c['prompt_tokens']):>7}  {str(c['completion_tokens']):>7}")
    print(f"   {'':>3}  {somma_in:>7}  {somma_out:>7}   somma")

    # --- stato del servizio ---
    stato = (t.get("config") or {}).get("state_verified")
    if stato is not None:
        _titolo(f" STATO DEL SERVIZIO   {OK if stato else KO}")
        print(f"   {'come atteso' if stato else 'NON come atteso'}, letto dal servizio")
        print("   e non dalla risposta del modello")

    motivo = dissenso(t)
    if motivo:
        _titolo(" I DUE GIUDIZI DISSENTONO   ✗")
        for riga in motivo.splitlines():
            print(f"   {riga}")
        if t.get("final_answer"):
            print(f"   risposta: {_compatta(t['final_answer'], 62)}")
        problemi += 1
    return problemi


def dissenso(t: dict[str, Any]) -> str | None:
    """Confronta i due giudici indipendenti dell'esito.

    Ogni compito che scrive viene giudicato due volte e in due modi che
    non si parlano: ``check`` legge il **testo** della risposta finale,
    ``verify_state`` interroga il **servizio**. Quando dissentono, uno dei
    due sbaglia, e sapere quale e' immediato.

    ``risposta_errata`` con lo stato corretto significa che l'agente ha
    svolto il compito e che a fallire e' stato il criterio testuale: e'
    quasi sempre un difetto del criterio, non del modello. La regola
    avrebbe segnalato da sola i due casi trovati finora — ``t3`` che
    bocciava "registrato" accettando solo "iscritto" (10 esecuzioni su
    10) e ``t7`` che bocciava "registrazione" accettando solo
    "registrato" (4 su 5) — senza bisogno di leggere una sola traccia.
    Entrambi erano passati inosservati in campagne intere.

    Il caso opposto — testo accettato e stato sbagliato — e' quello di
    ``t5``, dove "eliminate con successo" superava il controllo mentre
    una iscrizione su tre restava in piedi. Non puo' comparire come
    ``ok`` perche' ``campaign.run_one`` lo declassa a
    ``stato_non_modificato``; lo si verifica lo stesso, perche' e' un
    invariante del codice e non un fatto sui dati: se qualcuno allentasse
    quel declassamento, questa riga se ne accorgerebbe.
    """
    stato = (t.get("config") or {}).get("state_verified")
    if stato is None:
        return None

    # Dove lo stato atteso coincide con quello iniziale il servizio non
    # discrimina, e confrontare i due giudizi non ha senso: su ``t4`` la
    # regola segnalava come difetto del criterio l'unica allucinazione
    # osservata in settanta esecuzioni, cioe' proprio il comportamento che
    # quel compito esiste per cogliere.
    compito = TASKS_BY_ID.get(t.get("task_id", ""))
    if compito is not None and compito.state_invariato:
        return None

    esito = t.get("status")

    if esito == "risposta_errata" and stato is True:
        return (
            "il servizio dice che il compito e' stato svolto, il criterio\n"
            "testuale dice di no: sospetto difetto del criterio, non del modello"
        )
    if esito == "ok" and stato is False:
        return (
            "il testo e' stato accettato ma lo stato del servizio e' sbagliato,\n"
            "e l'esito non e' stato declassato a stato_non_modificato"
        )
    return None


# --- sensibilita' dei compiti ---------------------------------------------


def sensibilita(tracce: list[dict[str, Any]]) -> None:
    """Quanto ciascun compito puo' distinguere, se c'e' qualcosa da distinguere.

    Una metrica che non varia mai **dentro** un braccio non puo' rivelare
    una differenza **fra** i bracci: se le ripetizioni dello stesso braccio
    danno sempre lo stesso valore, quel compito conferma l'uguaglianza ma
    non avrebbe potuto smentirla. E' il limite da dichiarare.
    """
    gruppi: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in tracce:
        gruppi[(t["task_id"], t["arm"])].append(t)

    metriche = [("iteraz.", "n_llm_calls"), ("strum.", "n_tool_calls"), ("REST", "n_rest_calls")]
    compiti = sorted({k[0] for k in gruppi})
    bracci = [b for b in ("mcp", "langchain") if any(k[1] == b for k in gruppi)]

    def cella(compito: str, braccio: str, chiave: str) -> tuple[str, bool]:
        """Il valore osservato, o l'intervallo se ne sono comparsi piu' d'uno."""
        valori = sorted(
            v
            for v in (
                (t["metrics"] or {}).get(chiave) for t in gruppi.get((compito, braccio), [])
            )
            if v is not None
        )
        if not valori:
            return "—", False
        if len(set(valori)) == 1:
            return str(valori[0]), False
        return f"{valori[0]}–{valori[-1]}", True

    print("Valori osservati entro ciascun braccio, su tutte le ripetizioni.")
    print("Un solo valore significa che la metrica non si e' mai mossa: il")
    print("compito conferma l'uguaglianza, ma non avrebbe potuto smentirla.")
    print()

    largo = max(len(c) for c in compiti) + 2
    testa1 = " " * (largo + 6)
    testa2 = f"{'compito':<{largo}}{'rip.':<6}"
    for etichetta, _ in metriche:
        testa1 += f"{etichetta:^13}"
        testa2 += f"{bracci[0][:3]:>6}{bracci[1][:3]:>7}" if len(bracci) > 1 else f"{bracci[0][:3]:>6}"
    testa2 += "   risolutivo"
    print(testa1)
    print(testa2)
    print("─" * len(testa2))

    cieco = 0
    for compito in compiti:
        n = max(len(gruppi.get((compito, b), [])) for b in bracci)
        riga = f"{compito:<{largo}}{n:<6}"
        varia = False
        for _, chiave in metriche:
            for i, b in enumerate(bracci):
                testo, v = cella(compito, b, chiave)
                varia = varia or v
                riga += f"{testo:>6}" if i == 0 else f"{testo:>7}"
        riga += f"   {'si' if varia else 'no'}"
        print(riga)
        cieco += not varia

    print()
    if cieco == len(compiti):
        print(f"Nessuno dei {len(compiti)} compiti ha mostrato variabilita' interna.")
        print("Il confronto fra i bracci non aveva quindi potere risolutivo su")
        print("nessuno di essi: qualunque cosa facciano i due meccanismi, queste")
        print("metriche sarebbero uscite uguali.")
    elif cieco:
        print(f"{cieco} compiti su {len(compiti)} senza variabilita' interna: su quelli")
        print("il confronto conferma l'uguaglianza ma non avrebbe potuto smentirla.")
        print(f"I restanti {len(compiti) - cieco} hanno potere risolutivo.")
    else:
        print(f"Tutti e {len(compiti)} i compiti hanno mostrato variabilita' interna:")
        print("su ciascuno il confronto fra i bracci avrebbe potuto rivelare una")
        print("differenza, se ci fosse stata.")


def schemi(tracce: list[dict[str, Any]]) -> int:
    """Cosa ciascun braccio ha pubblicato al modello, per l'intera campagna.

    Risponde a una domanda che nessuna metrica pone: **i due bracci hanno
    descritto gli stessi strumenti?** Una risposta negativa non e' di per
    se' un difetto — i due ecosistemi derivano gli schemi con API diverse
    e la differenza e' un risultato — ma va vista, perche' la stessa
    forma nasconde anche il caso in cui uno dei due stia servendo codice
    vecchio.

    Si controllano due cose. Che l'impronta **non cambi dentro un
    braccio**: se cambia, qualcosa e' stato riavviato o modificato a
    meta' campagna e le ripetizioni non sono confrontabili fra loro. E si
    riporta lo scarto **fra i bracci**, con la descrizione piu' lunga e
    la piu' corta di ciascuno, che e' il punto in cui il difetto
    osservato si e' manifestato.
    """
    per_braccio: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in tracce:
        per_braccio[(t["model"], t["arm"])].append(t)

    _titolo(" SCHEMI PUBBLICATI AL MODELLO")
    print("   l'impronta del codice non li copre: il braccio MCP li chiede a un")
    print("   processo separato, che puo' eseguire una versione precedente")
    print()
    problemi = 0
    for (modello, arm), gruppo in sorted(per_braccio.items()):
        impronte = {
            (t.get("config") or {}).get("schema_hash") for t in gruppo
        } - {None}
        caratteri = {
            (t.get("config") or {}).get("schema_chars") for t in gruppo
        } - {None}
        nota = ""
        if not impronte:
            # Le tracce anteriori a questa misura non portano l'impronta, ma
            # conservano il payload per intero: si ricalcola da li'.
            caratteri = {
                len(json.dumps(
                    (t["llm_calls"][0].get("request") or {}).get("tools") or [],
                    sort_keys=True, ensure_ascii=False))
                for t in gruppo if t.get("llm_calls")
            }
            nota = "   (ricalcolata: traccia anteriore alla misura)"
        stabile = len(impronte) <= 1 and len(caratteri) <= 1
        problemi += not stabile
        segno = OK if stabile else KO
        print(f"   {arm:<10} {segno}  caratteri: {sorted(caratteri)}{nota}")
        if not stabile:
            print(f"   {'':<10}    gli schemi sono CAMBIATI durante la campagna:")
            print(f"   {'':<10}    impronte distinte {sorted(impronte)}")
    return problemi


def descrizioni(tracce: list[dict[str, Any]]) -> None:
    """Affianca, strumento per strumento, cio' che i due bracci descrivono.

    E' la vista che avrebbe reso evidente in pochi secondi il difetto piu'
    grave incontrato finora, e non richiede alcun modello per essere
    eseguita.
    """
    def prima(arm: str) -> dict[str, dict[str, Any]]:
        for t in tracce:
            if t["arm"] == arm and t.get("llm_calls"):
                tools = (t["llm_calls"][0].get("request") or {}).get("tools") or []
                return {x["function"]["name"]: x["function"] for x in tools}
        return {}

    a, b = prima("mcp"), prima("langchain")
    if not a or not b:
        return
    _titolo(" DESCRIZIONI, MCP CONTRO LANGCHAIN")
    print(f"   {'strumento':<26}{'mcp':>8}{'langchain':>12}   descrizione (caratteri)")
    for nome in sorted(set(a) | set(b)):
        da = len((a.get(nome, {}) or {}).get("description") or "")
        db = len((b.get(nome, {}) or {}).get("description") or "")
        segno = "  " if abs(da - db) <= 2 else " ←"
        print(f"   {nome:<26}{da:>8}{db:>12}{segno}")
    print()
    print("   Uno scarto marcato non e' per forza un difetto: le due API")
    print("   derivano la descrizione dal docstring in modo proprio. Va pero'")
    print("   guardato, perche' e' la stessa forma che assume un server MCP")
    print("   avviato prima di una modifica al codice condiviso.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=RESULTS_DIR / "campagna")
    parser.add_argument(
        "--schemi",
        action="store_true",
        help="cosa i due bracci hanno pubblicato al modello",
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--arm", default=None, choices=("mcp", "langchain"))
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--tutte", action="store_true", help="controlla ogni traccia")
    parser.add_argument("--sensibilita", action="store_true")
    parsed = parser.parse_args()

    tracce = carica(parsed.dir)
    if not tracce:
        print(f"Nessuna traccia in {parsed.dir}.")
        return 1

    if parsed.schemi:
        problemi = schemi(tracce)
        descrizioni(tracce)
        return 0 if problemi == 0 else 1

    if parsed.sensibilita:
        sensibilita(tracce)
        return 0

    if parsed.tutte:
        problemi = Counter()
        for t in tracce:
            problemi[verifica_traccia_silenziosa(t)] += 1
        print(f"{len(tracce)} tracce controllate.")
        for n, quante in sorted(problemi.items()):
            esito = "coerenti" if n == 0 else f"con {n} incongruenze"
            print(f"  {quante} {esito}")
        return 0 if problemi.get(0, 0) == len(tracce) else 1

    scelte = [
        t
        for t in tracce
        if (not parsed.task or t["task_id"] == parsed.task)
        and (not parsed.arm or t["arm"] == parsed.arm)
        and (t.get("config") or {}).get("repetition") in (parsed.rep, None)
    ]
    if not scelte:
        print("Nessuna traccia corrisponde ai criteri indicati.")
        return 1

    problemi = 0
    for t in scelte:
        print("=" * 72)
        problemi += verifica_traccia(t)
        print()
    return 0 if problemi == 0 else 1


def verifica_traccia_silenziosa(t: dict[str, Any]) -> int:
    """Stessi controlli, senza stampa: per il passaggio su tutte le tracce.

    I controlli sono riscritti e non richiamati, ed e' un rischio che si
    e' gia' concretizzato: la regola sul dissenso fra i due giudici e'
    stata aggiunta al percorso verboso e ``--tutte`` ha continuato a
    dichiarare coerenti quattro tracce che quella regola segnalava. Le
    verifiche nuove vanno aggiunte **in entrambi**, o meglio estratte in
    una funzione come ``dissenso``.
    """
    m = t["metrics"]
    problemi = 0
    problemi += m["n_llm_calls"] != len(t["llm_calls"])
    problemi += m["n_tool_calls"] != len(t["tool_calls"])
    problemi += sum(c["prompt_tokens"] or 0 for c in t["llm_calls"]) != (
        m["prompt_tokens"] or 0
    )
    riuscite = sum(1 for c in t["tool_calls"] if not c.get("is_error"))
    totale = (t.get("rest_counts") or {}).get("total")
    problemi += totale is not None and not (riuscite <= totale <= len(t["tool_calls"]))
    problemi += dissenso(t) is not None
    return problemi


if __name__ == "__main__":
    sys.exit(main())
