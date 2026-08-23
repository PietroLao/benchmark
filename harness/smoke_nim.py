"""Fase 0 — verifica che il tool calling funzioni su NIM.

Va eseguito **prima** di costruire i loop agentici. Se il modello non
emette blocchi ``tool_calls`` strutturati, l'intero esperimento B e'
invalido, e il sintomo somiglia a un problema di qualita' del modello:
un harness costruito contro un endpoint che non parsifica i tool call
puo' costare giorni di diagnosi sbagliata.

Verifica quattro cose:

1. l'endpoint risponde e il modello e' raggiungibile;
2. il modello emette ``tool_calls`` invece di descrivere a parole cosa
   farebbe, usando **gli schemi reali del benchmark**, non un esempio
   giocattolo: e' anche una validazione anticipata di ``tools_spec``;
3. sa estrarre correttamente gli argomenti da una richiesta in italiano;
4. chiude il giro, cioe' produce una risposta finale quando gli si
   restituisce il risultato dello strumento.

Riporta inoltre le latenze osservate: e' una prima lettura del rumore
dell'endpoint condiviso, utile per dimensionare la fase 5.

Ogni esecuzione viene salvata in ``results/``. Non e' un dettaglio
accessorio: la dispersione *fra* esecuzioni dello stesso identico test e'
la prova empirica che ``t_llm`` non e' attribuibile al protocollo, e vale
piu' di qualunque singola esecuzione presa da sola. Serve percio' che le
esecuzioni si accumulino su disco anziche' nello scrollback del terminale.

Uso::

    export NVIDIA_API_KEY=nvapi-...
    uv run python -m harness.smoke_nim
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness.trace import RunTrace
from shared.tools_spec import openai_tools_format

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct")
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

#: Traccia integrale dell'esecuzione. La fase 0 non è un esperimento, ma
#: registrarla serve comunque a due cose: verificare che l'endpoint
#: riporti il conteggio dei token in ``usage``, e collaudare il formato
#: delle tracce prima che lo usino i cicli agentici veri.
trace = RunTrace(
    arm="smoke",
    model=MODEL,
    task_id="fase0",
    prompt="(tre verifiche indipendenti sul tool calling)",
)

OK = "✓"
KO = "✗"


#: Temperatura effettivamente accettata dall'endpoint. Si parte da 0 per
#: la riproducibilita'; alcuni endpoint NIM rifiutano lo zero e in quel
#: caso si ripiega sul minimo utilizzabile. Il valore va poi riportato in
#: sede di setup sperimentale, perche' incide sulla varianza fra run.
TEMPERATURE = 0.0
_temperature_negotiated = False


#: Codici su cui ha senso ritentare: sovraccarico o indisponibilita'
#: temporanea dell'endpoint condiviso, non errori della nostra richiesta.
RETRYABLE = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5

#: Ritentativi osservati, riportati nel riepilogo: sono un dato sulla
#: stabilita' dell'endpoint, utile a dimensionare la campagna.
retries_observed: list[tuple[int, float]] = []

#: Scostamenti fra gli argomenti attesi e quelli emessi dal modello.
#: Vanno registrati perche' documentano un comportamento deterministico
#: del modello (i numeri restituiti come stringhe) che motiva
#: ``tools_spec.coerce_arguments``.
anomalies: list[dict[str, Any]] = []


def _post(payload: dict[str, Any]) -> tuple[httpx.Response, float]:
    """Invia la richiesta, ritentando con backoff sugli errori transitori."""
    last: httpx.Response | None = None
    for attempt in range(MAX_ATTEMPTS):
        start = time.perf_counter()
        response = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180.0,
        )
        elapsed = time.perf_counter() - start
        if response.status_code not in RETRYABLE:
            return response, elapsed

        last = response
        delay = min(2**attempt, 30)
        retries_observed.append((response.status_code, delay))
        trace.event("retry", status_code=response.status_code, backoff_s=delay)
        print(
            f"  ~ HTTP {response.status_code}, ritento fra {delay}s "
            f"(tentativo {attempt + 2}/{MAX_ATTEMPTS})"
        )
        time.sleep(delay)

    assert last is not None
    return last, 0.0


def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any], float]:
    """Invia una richiesta e restituisce (messaggio, latenza in secondi)."""
    global TEMPERATURE, _temperature_negotiated

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": TEMPERATURE,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response, elapsed = _post(payload)

    # Alcuni endpoint richiedono temperature > 0: si rinegozia una volta
    # sola e lo si segnala, invece di far fallire lo smoke test con un
    # 400 di difficile lettura.
    if response.status_code == 400 and not _temperature_negotiated:
        _temperature_negotiated = True
        if "temperature" in response.text.lower():
            TEMPERATURE = 0.01
            print(
                f"  ! l'endpoint rifiuta temperature=0; si ripiega su "
                f"{TEMPERATURE} (da riportare in §4.3)"
            )
            payload["temperature"] = TEMPERATURE
            response, elapsed = _post(payload)

    if response.status_code >= 400:
        trace.llm_call(payload, None, elapsed, error=f"HTTP {response.status_code}")
        response.raise_for_status()

    body = response.json()
    trace.llm_call(payload, body, elapsed)
    return body, elapsed


def main() -> int:
    if not API_KEY:
        print(f"{KO} NVIDIA_API_KEY non impostata.")
        print("  Ottieni una chiave gratuita su https://build.nvidia.com")
        print("  poi:  export NVIDIA_API_KEY=nvapi-...")
        return 1

    tools = openai_tools_format()
    print(f"Endpoint : {BASE_URL}")
    print(f"Modello  : {MODEL}")
    print(f"Strumenti: {len(tools)} (da shared/tools_spec.py)\n")

    latencies: list[float] = []
    failures = 0

    # --- 1. Il modello emette tool_calls? -------------------------------
    print("[1] Il modello emette tool_calls invece di descrivere a parole?")
    try:
        body, dt = chat(
            [{"role": "user", "content": "Quali eventi sono disponibili?"}], tools
        )
    except httpx.HTTPStatusError as exc:
        print(f"  {KO} HTTP {exc.response.status_code}: {exc.response.text[:300]}")
        # Anche — anzi soprattutto — l'esecuzione fallita va salvata: un
        # fallimento dopo i ritentativi esauriti e' il caso piu'
        # informativo sulla stabilita' dell'endpoint condiviso.
        _persist(latencies, failures + 1, error=f"HTTP {exc.response.status_code}")
        return 1
    except httpx.HTTPError as exc:
        print(f"  {KO} Errore di rete: {exc}")
        _persist(latencies, failures + 1, error=f"{type(exc).__name__}: {exc}")
        return 1

    latencies.append(dt)
    choice = body["choices"][0]
    message = choice["message"]
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        print(f"  {KO} Nessun tool_call. finish_reason={choice.get('finish_reason')!r}")
        print(f"     Il modello ha risposto a parole: {(message.get('content') or '')[:200]!r}")
        print("     Il tool calling non e' attivo su questo modello/endpoint.")
        failures += 1
    else:
        call = tool_calls[0]
        name = call["function"]["name"]
        print(f"  {OK} tool_call ricevuto: {name}  ({dt:.2f}s, finish_reason={choice.get('finish_reason')!r})")
        if name != "list_events":
            print(f"  ! atteso list_events, ottenuto {name} (non bloccante)")
        if len(tool_calls) > 1:
            print(f"  i Chiamate parallele nella stessa risposta: {len(tool_calls)}")

    # --- 2. Estrazione degli argomenti ----------------------------------
    print("\n[2] Sa estrarre gli argomenti da una richiesta in italiano?")
    body2, dt2 = chat(
        [
            {
                "role": "user",
                "content": (
                    "Iscrivi Marco Rossi, username mrossi, email "
                    "marco.rossi@example.it, all'evento con id 1."
                ),
            }
        ],
        tools,
    )
    latencies.append(dt2)
    calls2 = body2["choices"][0]["message"].get("tool_calls") or []
    if not calls2:
        print(f"  {KO} Nessun tool_call sulla richiesta con argomenti.")
        failures += 1
    else:
        fn = calls2[0]["function"]
        try:
            args = json.loads(fn["arguments"])
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"  {KO} arguments non e' JSON valido: {exc}")
            print(f"     grezzo: {fn['arguments']!r}")
            failures += 1
        else:
            print(f"  {OK} {fn['name']}({json.dumps(args, ensure_ascii=False)})  ({dt2:.2f}s)")
            expected = {"event_id": 1, "username": "mrossi"}
            wrong = {k: (args.get(k), v) for k, v in expected.items() if args.get(k) != v}
            if wrong:
                print(f"  ! argomenti inattesi: {wrong} (non bloccante, ma da annotare)")
                for key, (got, want) in wrong.items():
                    anomalies.append(
                        {
                            "argument": key,
                            "observed": got,
                            "observed_type": type(got).__name__,
                            "expected": want,
                            "expected_type": type(want).__name__,
                        }
                    )

    # --- 3. Chiusura del giro -------------------------------------------
    print("\n[3] Chiude il giro quando riceve il risultato dello strumento?")
    if not tool_calls:
        print("  - saltato: nessun tool_call dal passo 1")
    else:
        call = tool_calls[0]
        fake_result = json.dumps(
            [{"id": 1, "title": "Conferenza sull'Intelligenza Artificiale", "location": "Cagliari"}],
            ensure_ascii=False,
        )
        body3, dt3 = chat(
            [
                {"role": "user", "content": "Quali eventi sono disponibili?"},
                message,
                {"role": "tool", "tool_call_id": call["id"], "content": fake_result},
            ],
            tools,
        )
        latencies.append(dt3)
        final = body3["choices"][0]["message"].get("content") or ""
        if final.strip():
            print(f"  {OK} Risposta finale ({dt3:.2f}s): {final.strip()[:180]}")
        else:
            print(f"  {KO} Nessuna risposta testuale finale.")
            failures += 1

    # --- Riepilogo -------------------------------------------------------
    print("\n" + "-" * 60)
    if latencies:
        print(
            f"Latenze osservate: min {min(latencies):.2f}s  "
            f"max {max(latencies):.2f}s  ({len(latencies)} chiamate riuscite)"
        )
        print("  (indicativo: la caratterizzazione del rumore e' la fase 5)")
    if retries_observed:
        codes = ", ".join(str(code) for code, _ in retries_observed)
        print(
            f"Ritentativi: {len(retries_observed)} su {len(latencies) + len(retries_observed)} "
            f"richieste totali (codici: {codes})"
        )
        print("  L'endpoint condiviso e' instabile: la campagna richiede backoff e checkpoint.")
    else:
        print("Ritentativi: nessuno.")

    _persist(latencies, failures)

    if failures:
        print(f"{KO} Fase 0 NON superata: {failures} verifica/he fallita/e.")
        return 1
    print(f"{OK} Fase 0 superata: si puo' procedere con i loop agentici.")
    return 0


def _persist(
    latencies: list[float], failures: int, error: str | None = None
) -> None:
    """Salva l'esito su disco, una riga per esecuzione.

    Le esecuzioni si accumulano: e' il confronto *fra* esecuzioni dello
    stesso test a documentare quanto vari l'endpoint condiviso a parita'
    di codice, ed e' quel dato a sostenere la scelta di non attribuire
    ``t_llm`` al protocollo.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "experiment": "phase0_smoke",
        "timestamp_utc": timestamp,
        "endpoint": BASE_URL,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "n_tools": len(openai_tools_format()),
        "latencies_s": latencies,
        "retries": [{"status_code": c, "backoff_s": d} for c, d in retries_observed],
        "argument_anomalies": anomalies,
        "failures": failures,
        "error": error,
    }
    out_path = RESULTS_DIR / f"smoke_{timestamp}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"Esito salvato in {out_path.relative_to(RESULTS_DIR.parent)}")

    trace.finish(status="ok" if not failures else "fallita")
    trace_path = trace.save()
    print(f"Traccia integrale in  {trace_path.relative_to(RESULTS_DIR.parent)}")

    # Risponde a una domanda rimasta aperta: l'endpoint riporta il
    # conteggio dei token, oppure vanno stimati a posteriori dai testi?
    m = trace.metrics()
    if m["tokens_reported_by_endpoint"]:
        print(
            f"Token riportati dall'endpoint: {m['prompt_tokens']} in ingresso, "
            f"{m['completion_tokens']} in uscita."
        )
    else:
        print(
            "L'endpoint NON riporta i token: andranno stimati dai testi salvati "
            f"({m['chars_sent']} caratteri inviati, {m['chars_received']} ricevuti)."
        )


if __name__ == "__main__":
    sys.exit(main())
