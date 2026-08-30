"""Implementazione unica delle operazioni sul sistema sotto test.

Questo modulo e' l'unico punto in cui vengono effettuate chiamate REST
verso l'Event Manager, ed e' la sola definizione condivisa dai due bracci:
ciascuno registra queste funzioni presso la propria API di alto livello,
che ne deriva lo schema.

Tenere una sola implementazione garantisce che i bracci differiscano
esclusivamente per il modo in cui lo strumento viene reso disponibile al
modello, mai per cosa lo strumento fa. Come **appare** invece puo'
differire, ed e' cio' che si misura.
"""

from __future__ import annotations

import contextvars
import os
from typing import Any

import httpx

BASE_URL = os.environ.get("BENCH_SERVER_URL", "http://127.0.0.1:8000")


class OperationError(RuntimeError):
    """Operazione rifiutata dal sistema sotto test.

    Viene **sollevata**, non restituita. E' la scelta che permette a
    ciascun ecosistema di applicare la propria convenzione sugli errori,
    che e' precisamente cio' che ``harness/error_paths.py`` misura: MCP la
    traduce in ``isError`` sul risultato del protocollo, mentre LangChain
    la consegna al modello solo se il braccio la converte prima in
    ``ToolException``.

    Restituirla come dizionario, com'era prima, faceva arrivare a
    entrambi i bracci un successo con del testo d'errore dentro:
    identici, e quindi ciechi proprio alla differenza da misurare.
    """

    def __init__(self, detail: Any, status_code: int | None = None) -> None:
        super().__init__(str(detail))
        self.detail = detail
        self.status_code = status_code


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

    Gli errori HTTP diventano ``OperationError``. L'agente deve comunque
    poterli leggere e correggersi, ed e' cio' che accade: entrambi gli
    ecosistemi consegnano al modello il testo dell'eccezione. Ma lo fanno
    per vie diverse, ed e' la via che qui interessa misurare.
    """
    response = get_client().request(method, path, headers=_headers(), **kwargs)
    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise OperationError(detail, response.status_code)
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
    """Crea un nuovo evento e restituisce l'evento creato con l'id assegnato.

    Args:
        title: Titolo dell'evento.
        description: Descrizione dell'evento.
        date: Data e ora in formato ISO 8601, per esempio
            ``2026-06-15T18:00:00``.
        location: Luogo dell'evento.
    """
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

    Non compare in ``TOOL_FUNCTIONS`` e viene registrato solo con
    ``BENCH_EXPOSE_ECHO=1``: il modello non lo vede mai. Serve unicamente al
    microbenchmark per isolare l'overhead di protocollo dalla latenza
    REST, misurando un'operazione il cui costo intrinseco e' nullo.
    """
    return {"echo": payload}


#: Le operazioni esposte al modello come strumenti, nell'ordine in cui
#: vengono pubblicate.
#:
#: E' l'unica definizione condivisa dai due bracci, e definisce **cosa lo
#: strumento fa**, non come appare. Lo schema che il modello vede lo
#: deriva ciascun ecosistema dalla firma e dalla docstring di queste
#: funzioni, con la propria API di alto livello: e' il modo in cui questi
#: strumenti vengono scritti nella pratica, ed e' quello che il confronto
#: deve misurare. Le differenze fra i due schemi derivati sono un
#: risultato, non un difetto da correggere.
TOOL_FUNCTIONS = (
    list_events,
    get_event,
    create_event,
    list_users,
    register_user_to_event,
    list_registrations,
    delete_registration,
)


#: Tabella di dispatch usata dal microbenchmark.
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
    """Invoca un'operazione per nome, senza alcuna mediazione.

    E' la condizione ``diretto`` dell'esperimento A: il riferimento
    inferiore, il costo dell'operazione quando nessun meccanismo di
    esposizione e' presente. I bracci sperimentali **non** passano di
    qui: registrano le funzioni direttamente presso la propria API di
    alto livello, che si occupa di validare gli argomenti e di tradurre
    le eccezioni secondo la propria convenzione.

    Non normalizza i tipi degli argomenti. Serviva quando il server MCP
    usava l'API di basso livello e inoltrava gli argomenti cosi' come
    arrivavano, mentre LangChain li validava con Pydantic; con le API di
    alto livello entrambi costruiscono un modello Pydantic dalla firma e
    convertono allo stesso modo. Verificato: ``"7"`` su un parametro
    ``int`` arriva come ``7`` in entrambi.
    """
    if name not in DISPATCH:
        raise OperationError(f"Strumento sconosciuto: {name}")
    return DISPATCH[name](**(arguments or {}))
