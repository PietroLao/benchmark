"""Definizione unica degli strumenti esposti al modello.

Entrambi i bracci sperimentali — il server MCP e l'agente LangChain —
derivano nome, descrizione e schema degli argomenti da questo modulo.

È il controllo metodologico su cui poggia l'intero confronto: se i due
bracci presentassero al modello schemi anche solo leggermente diversi,
le differenze misurate in numero di iterazioni e di chiamate REST non
sarebbero piu' attribuibili al protocollo. Vedi il diff dei body HTTP
descritto nel README.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Descrizione di uno strumento, indipendente dal meccanismo di esposizione."""

    name: str
    description: str
    input_schema: dict[str, Any]


def _obj(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    """Costruisce uno schema JSON di tipo object con chiavi ordinate.

    L'ordine delle chiavi e' deterministico perche' la serializzazione
    dello schema entra nel prompt: una variazione d'ordine cambierebbe i
    byte inviati al modello e invaliderebbe il confronto tra i bracci.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


_STR = {"type": "string"}
_INT = {"type": "integer"}


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_events",
        description=(
            "Restituisce la lista completa degli eventi esistenti, ciascuno "
            "con id, titolo, descrizione, data e luogo. Non accetta filtri: "
            "per trovare un evento specifico occorre esaminare la lista."
        ),
        input_schema=_obj({}),
    ),
    ToolSpec(
        name="get_event",
        description="Restituisce i dettagli dell'evento con l'id indicato.",
        input_schema=_obj(
            {"event_id": {**_INT, "description": "Identificativo dell'evento"}},
            ["event_id"],
        ),
    ),
    ToolSpec(
        name="create_event",
        description="Crea un nuovo evento e restituisce l'evento creato con l'id assegnato.",
        input_schema=_obj(
            {
                "title": {**_STR, "description": "Titolo dell'evento"},
                "description": {**_STR, "description": "Descrizione dell'evento"},
                "date": {
                    **_STR,
                    "description": "Data e ora in formato ISO 8601, es. 2026-06-15T18:00:00",
                },
                "location": {**_STR, "description": "Luogo dell'evento"},
            },
            ["title", "description", "date", "location"],
        ),
    ),
    ToolSpec(
        name="list_users",
        description="Restituisce la lista completa degli utenti registrati nel sistema.",
        input_schema=_obj({}),
    ),
    ToolSpec(
        name="register_user_to_event",
        description=(
            "Iscrive un utente a un evento. Se l'utente non esiste ancora viene "
            "creato automaticamente. Fallisce se l'utente e' gia' iscritto a "
            "quell'evento."
        ),
        input_schema=_obj(
            {
                "event_id": {**_INT, "description": "Identificativo dell'evento"},
                "username": {**_STR, "description": "Username dell'utente"},
                "name": {**_STR, "description": "Nome completo dell'utente"},
                "email": {**_STR, "description": "Indirizzo email dell'utente"},
            },
            ["event_id", "username", "name", "email"],
        ),
    ),
    ToolSpec(
        name="list_registrations",
        description=(
            "Restituisce tutte le iscrizioni esistenti come coppie "
            "(username, event_id)."
        ),
        input_schema=_obj({}),
    ),
    ToolSpec(
        name="delete_registration",
        description="Cancella l'iscrizione di un utente a un evento.",
        input_schema=_obj(
            {
                "username": {**_STR, "description": "Username dell'iscritto"},
                "event_id": {**_INT, "description": "Identificativo dell'evento"},
            },
            ["username", "event_id"],
        ),
    ),
)


TOOL_SPECS_BY_NAME: dict[str, ToolSpec] = {s.name: s for s in TOOL_SPECS}


def coerce_arguments(tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Converte gli argomenti ai tipi dichiarati nello schema.

    Llama 3.3 restituisce talvolta i numeri come stringhe (``"1"`` invece
    di ``1``) anche quando lo schema dichiara ``integer``. Senza questa
    normalizzazione i due bracci divergerebbero sullo stesso identico
    output del modello: LangChain converte o rifiuta tramite la
    validazione Pydantic dei suoi strumenti, mentre un server MCP
    minimale inoltra gli argomenti cosi' come arrivano.

    La differenza sarebbe reale — la validazione degli argomenti e' una
    cosa che il framework offre e che con MCP va implementata — ma e' un
    fatto qualitativo da discutere nell'asse "overhead di
    implementazione", non un effetto che deve inquinare i conteggi di
    iterazioni attribuendoli al protocollo.
    """
    spec = TOOL_SPECS_BY_NAME.get(tool_name)
    if spec is None or not arguments:
        return dict(arguments or {})

    properties = spec.input_schema.get("properties", {})
    coerced: dict[str, Any] = {}
    for key, value in arguments.items():
        declared = properties.get(key, {}).get("type")
        try:
            if declared == "integer" and not isinstance(value, bool):
                coerced[key] = int(value)
            elif declared == "number" and not isinstance(value, bool):
                coerced[key] = float(value)
            elif declared == "string" and not isinstance(value, str):
                coerced[key] = str(value)
            else:
                coerced[key] = value
        except (TypeError, ValueError):
            # Valore non convertibile: si inoltra invariato e sara'
            # l'operazione a produrre un errore leggibile dal modello.
            coerced[key] = value
    return coerced


def openai_tools_format() -> list[dict[str, Any]]:
    """Converte le specifiche nel formato ``tools`` dell'API OpenAI.

    E' il formato che NIM accetta su ``/v1/chat/completions``: entrambi i
    bracci arrivano qui, per vie diverse.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.input_schema,
            },
        }
        for s in TOOL_SPECS
    ]
