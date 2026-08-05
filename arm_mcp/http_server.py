"""Lo stesso server MCP, servito su trasporto Streamable HTTP.

Espone esattamente gli strumenti di ``arm_mcp.server`` — stesso oggetto
``Server``, stessi schemi, stesso dispatch — cambiando unicamente il
trasporto. E' cio' che rende interpretabile il confronto fra i due
trasporti nell'esperimento A.

Serve a separare due costi che su stdio risultano inscindibili. Aprire
una nuova sessione su stdio significa avviare un nuovo processo Python,
e la misura era dominata dal tempo di import dell'interprete (~470 ms),
una proprieta' dell'implementazione e non del protocollo. Su HTTP il
processo server e' gia' in esecuzione: aprire una nuova sessione costa
solo l'handshake di inizializzazione. La differenza fra le due condizioni
``new_session`` isola quindi il costo di protocollo da quello di avvio.

Avvio::

    uv run uvicorn arm_mcp.http_server:app --port 8100

L'endpoint e' ``/mcp/``, **con la barra finale**: ``Mount`` risponde a
``/mcp`` con un redirect 307 che il client segue in modo trasparente,
raddoppiando i round trip di ogni misura.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from arm_mcp.server import server

#: ``json_response=True`` evita lo streaming SSE per le risposte: le
#: chiamate misurate sono richiesta-risposta singole, e lo stream
#: aggiungerebbe alla misura un costo di framing che nessuno dei due
#: bracci paga in esercizio.
session_manager = StreamableHTTPSessionManager(app=server, json_response=True)


async def handle(scope: Scope, receive: Receive, send: Send) -> None:
    await session_manager.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(_: Starlette) -> AsyncIterator[None]:
    """Il gestore di sessioni va avviato una volta sola per processo."""
    async with session_manager.run():
        yield


app = Starlette(routes=[Mount("/mcp", app=handle)], lifespan=lifespan)
