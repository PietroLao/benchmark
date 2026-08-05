"""Implementazione unica delle operazioni sul sistema sotto test.

Questo modulo e' l'unico punto in cui vengono effettuate chiamate REST
verso l'Event Manager. Entrambi i bracci sperimentali lo invocano: il
server MCP lo espone tramite ``tools/call``, l'agente LangChain lo
avvolge con il decoratore ``@tool``.

Tenere una sola implementazione garantisce che i bracci differiscano
esclusivamente per il modo in cui lo strumento viene reso disponibile al
modello, mai per cosa lo strumento fa.
"""

from __future__ import annotations

import contextvars
import os
from typing import Any

import httpx

from shared.tools_spec import coerce_arguments

BASE_URL = os.environ.get("BENCH_SERVER_URL", "http://127.0.0.1:8000")

#: Identificativo della run corrente, propagato al server come header
#: ``X-Run-Id`` per attribuire ogni chiamata REST alla run che l'ha
#: generata senza mescolare i conteggi tra run diverse.
current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)

_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    """Restituisce il client HTTP condiviso, creandolo alla prima chiamata.

    Il client e' unico e riusa le connessioni: senza pooling, ogni
    operazione pagherebbe un handshake TCP e la latenza REST misurata
    sarebbe dominata dal costo di connessione anziche' dal servizio.
    """
    global _client
    if _client is None:
        _client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    return _client


def close_client() -> None:
    """Chiude il client condiviso, se aperto."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def _headers() -> dict[str, str]:
    run_id = current_run_id.get()
    return {"X-Run-Id": run_id} if run_id else {}


def _request(method: str, path: str, **kwargs: Any) -> Any:
    """Effettua una richiesta e restituisce il corpo decodificato.

    Gli errori HTTP non vengono sollevati come eccezioni ma restituiti
    come dizionari con chiave ``error``: l'agente deve poterli leggere e
    correggersi, esattamente come farebbe con un tool result di errore.
    """
    response = get_client().request(method, path, headers=_headers(), **kwargs)
    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"error": True, "status_code": response.status_code, "detail": detail}
    if not response.content:
        return {"ok": True}
    return response.json()


# --- Operazioni esposte come strumenti ------------------------------------


def list_events() -> Any:
    """Restituisce tutti gli eventi."""
    return _request("GET", "/events")


def get_event(event_id: int) -> Any:
    """Restituisce l'evento con l'id indicato."""
    return _request("GET", f"/events/{event_id}")


def create_event(title: str, description: str, date: str, location: str) -> Any:
    """Crea un nuovo evento."""
    return _request(
        "POST",
        "/events",
        json={
            "title": title,
            "description": description,
            "date": date,
            "location": location,
        },
    )


def list_users() -> Any:
    """Restituisce tutti gli utenti."""
    return _request("GET", "/users/")


def register_user_to_event(
    event_id: int, username: str, name: str, email: str
) -> Any:
    """Iscrive un utente a un evento, creandolo se non esiste."""
    return _request(
        "POST",
        f"/events/{event_id}/register",
        json={"username": username, "name": name, "email": email},
    )


def list_registrations() -> Any:
    """Restituisce tutte le iscrizioni."""
    return _request("GET", "/registrations")


def delete_registration(username: str, event_id: int) -> Any:
    """Cancella l'iscrizione di un utente a un evento."""
    return _request(
        "DELETE",
        "/registrations",
        params={"username": username, "event_id": event_id},
    )


def _bench_echo(payload: str = "ping") -> Any:
    """Restituisce l'argomento senza toccare la rete.

    Non compare in ``TOOL_SPECS`` e non viene pubblicato da
    ``tools/list``: il modello non lo vede mai. Serve unicamente al
    microbenchmark per isolare l'overhead di protocollo dalla latenza
    REST, misurando un'operazione il cui costo intrinseco e' nullo.
    """
    return {"echo": payload}


#: Tabella di dispatch usata dal server MCP e dal microbenchmark.
DISPATCH = {
    "_bench_echo": _bench_echo,
    "list_events": list_events,
    "get_event": get_event,
    "create_event": create_event,
    "list_users": list_users,
    "register_user_to_event": register_user_to_event,
    "list_registrations": list_registrations,
    "delete_registration": delete_registration,
}


def call(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Invoca un'operazione per nome con gli argomenti indicati.

    Gli argomenti vengono normalizzati ai tipi dichiarati nello schema
    prima dell'invocazione: entrambi i bracci passano di qui, quindi
    reagiscono in modo identico allo stesso output del modello. Vedi
    ``tools_spec.coerce_arguments``.
    """
    if name not in DISPATCH:
        return {"error": True, "detail": f"Strumento sconosciuto: {name}"}
    normalized = coerce_arguments(name, arguments)
    try:
        return DISPATCH[name](**normalized)
    except TypeError as exc:
        # Argomenti mancanti o inattesi: va restituito al modello come
        # risultato leggibile, non sollevato come eccezione che
        # interromperebbe la run.
        return {"error": True, "detail": f"Argomenti non validi per {name}: {exc}"}
