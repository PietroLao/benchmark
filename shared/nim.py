"""Client minimo per l'endpoint NVIDIA, usato dal braccio MCP.

Il braccio LangChain non passa di qui: interroga il modello attraverso
``ChatNVIDIA``, come farebbe chiunque usi quel framework. Questo modulo
serve all'host MCP, che il ciclo di conversazione se lo gestisce da solo
e ha quindi bisogno di parlare direttamente con l'API.

La duplicazione e' voluta. Far passare anche il braccio MCP per LangChain
significherebbe misurare LangChain-che-parla-MCP, che e' precisamente il
confondimento che il disegno sperimentale esiste per evitare.

Le due strade convergono comunque sullo stesso formato di richiesta:
l'endpoint e' compatibile con l'API OpenAI, e ``ChatNVIDIA`` vi si
adegua. E' il substrato condiviso descritto nel Capitolo 2 della tesi.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from shared.env import ENV_PATH  # carica .env, se presente  # noqa: F401

BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

#: I modelli ospitati hanno una data di fine vita e vengono ritirati senza
#: preavviso utile: ``meta/llama-3.3-70b-instruct``, usato nella prima
#: campagna, e' stato dismesso il 2026-08-26 e da allora l'endpoint
#: risponde 410. Il nome del modello va quindi considerato una
#: configurazione volatile, e ogni risultato va riportato indicando su
#: quale modello e in quale data e' stato ottenuto.


def _api_key() -> str:
    """Legge la chiave al momento dell'uso, non all'import del modulo.

    Leggerla all'import la fisserebbe al valore presente in quell'istante:
    chi importasse il modulo prima di impostare la variabile — da un
    notebook, o caricando un file di ambiente — si troverebbe la chiave
    vuota senza capirne il motivo. La chiave non compare comunque mai
    nelle tracce salvate, perche' viaggia nelle intestazioni HTTP, che
    non vengono registrate.
    """
    return os.environ.get("NVIDIA_API_KEY", "")

#: Codici su cui ha senso ritentare: sovraccarico o indisponibilita'
#: temporanea dell'endpoint condiviso, non errori della nostra richiesta.
RETRYABLE = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4

#: Un tentativo osservato ha superato i 180 secondi. Il limite e' alzato
#: per non troncare le code lunghe legittime, ma resta finito: oltre
#: questa soglia conviene ritentare piuttosto che attendere.
TIMEOUT_S = 240.0


class NimError(RuntimeError):
    """Errore non ritentabile, o esaurimento dei tentativi."""


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    trace: Any | None = None,
) -> tuple[dict[str, Any], float]:
    """Interroga il modello e restituisce (risposta, secondi trascorsi).

    Se ``trace`` e' fornito, ogni tentativo viene registrato: le richieste
    riuscite come chiamate al modello, quelle fallite come eventi. Serve a
    poter distinguere a posteriori una esecuzione lenta da una che ha
    dovuto ritentare.
    """
    key = _api_key()
    if not key:
        raise NimError("NVIDIA_API_KEY non impostata.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        # Il modello puo' emettere piu' chiamate in una sola risposta, ma
        # l'endpoint le rifiuta con un 500 "This model only supports
        # single tool-calls at once!". L'errore non si manifesta subito:
        # arriva alla richiesta *successiva*, quando la conversazione
        # contenente le chiamate parallele viene reinviata, e fa fallire
        # l'intera esecuzione. Disattivarle serializza il ciclo e rende
        # inoltre i due bracci confrontabili passo per passo.
        payload["parallel_tool_calls"] = False

    last_status: int | None = None
    for attempt in range(MAX_ATTEMPTS):
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=TIMEOUT_S,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Un timeout non e' un codice di stato, quindi senza questo
            # ramo uscirebbe dal ciclo dei ritentativi e farebbe perdere
            # l'intera esecuzione. Osservato: una chiamata oltre i 180 s
            # ha fatto fallire una fase 0 che, rilanciata identica, e'
            # andata a buon fine.
            elapsed = time.perf_counter() - start
            delay = min(2**attempt, 30)
            if trace is not None:
                trace.event(
                    "retry_timeout",
                    detail=f"{type(exc).__name__}: {exc}",
                    elapsed_s=elapsed,
                    backoff_s=delay,
                    attempt=attempt + 1,
                )
            if attempt == MAX_ATTEMPTS - 1:
                raise NimError(
                    f"Esauriti {MAX_ATTEMPTS} tentativi, ultimo: {type(exc).__name__}"
                ) from exc
            time.sleep(delay)
            continue
        elapsed = time.perf_counter() - start

        if response.status_code < 400:
            body = response.json()
            if trace is not None:
                trace.llm_call(payload, body, elapsed, attempt=attempt + 1)
            return body, elapsed

        last_status = response.status_code
        # Alcuni 5xx sono permanenti: un prompt che l'endpoint non sa
        # elaborare non diventa valido riprovando, e ritentarlo brucia
        # quattro timeout da 240 s per nulla.
        permanente = any(
            s in response.text
            for s in ("invalid operation", "Failed to apply prompt template")
        )
        if response.status_code not in RETRYABLE or permanente:
            if trace is not None:
                trace.llm_call(
                    payload,
                    None,
                    elapsed,
                    attempt=attempt + 1,
                    error=f"HTTP {response.status_code}: {response.text[:300]}",
                )
            raise NimError(f"HTTP {response.status_code}: {response.text[:300]}")

        delay = min(2**attempt, 30)
        if trace is not None:
            trace.event(
                "retry", status_code=response.status_code, backoff_s=delay,
                attempt=attempt + 1,
            )
        time.sleep(delay)

    raise NimError(f"Esauriti {MAX_ATTEMPTS} tentativi (ultimo stato {last_status}).")
