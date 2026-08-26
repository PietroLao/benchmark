"""Intercetta il payload HTTP realmente trasmesso da ``ChatNVIDIA``.

Il braccio LangChain non espone la richiesta inviata al modello: il
callback riceve i messaggi come oggetti del framework, e ricostruirli con
``convert_to_openai_messages`` produce qualcosa di *simile* al payload,
non il payload. La differenza non e' teorica. Confrontando la
ricostruzione con il filo sono emerse due divergenze che il gate sugli
input non poteva vedere, perche' confrontava due tracce entrambe
ricostruite:

* l'assistant message trasmesso porta ``reasoning`` e ``reasoning_content``
  — il framework rimanda al modello, a ogni giro, il ragionamento dei
  giri precedenti — e la ricostruzione li perde;
* il messaggio di strumento trasmesso **non** porta ``name``, mentre la
  ricostruzione lo mostra.

Registrare il filo elimina la classe di errore alla radice: qualunque
campo il framework aggiunga o tolga in futuro finisce nella traccia senza
che occorra prevederlo.

L'aggancio e' su ``aiohttp.ClientSession._request`` perche' e' il punto
piu' basso ancora in Python, dopo qualunque trasformazione della libreria
del fornitore. Non altera la richiesta: la osserva soltanto.
"""

from __future__ import annotations

import json
from typing import Any

import aiohttp

_installato = False

#: Ultimo payload osservato. Le esecuzioni sono sequenziali e un
#: ritentativo sovrascrive il tentativo fallito, quindi al momento di
#: ``on_llm_end`` questo e' il payload della richiesta andata a buon fine.
_ultimo: dict[str, Any] | None = None


def installa() -> None:
    """Applica l'aggancio una sola volta per processo."""
    global _installato
    if _installato:
        return

    originale = aiohttp.ClientSession._request

    async def osserva(self: Any, method: str, url: Any, **kwargs: Any) -> Any:
        global _ultimo
        payload = kwargs.get("json")
        if payload is not None and "chat/completions" in str(url):
            # Copia profonda: il chiamante potrebbe riusare la struttura.
            _ultimo = json.loads(json.dumps(payload, default=str))
        return await originale(self, method, url, **kwargs)

    aiohttp.ClientSession._request = osserva  # type: ignore[method-assign]
    _installato = True


def ultimo() -> dict[str, Any] | None:
    """Il payload dell'ultima richiesta al modello, o ``None``."""
    return _ultimo


def azzera() -> None:
    """Dimentica il payload osservato, prima di una nuova esecuzione."""
    global _ultimo
    _ultimo = None
