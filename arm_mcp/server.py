"""Server MCP che espone le operazioni sull'Event Manager.

Usa ``MCPServer``, l'API di alto livello dell'SDK: si registra la
funzione e il server ne deriva nome, descrizione e schema degli argomenti
dalla firma e dalla docstring.

E' la sostituzione dell'API di basso livello che il banco di prova usava
prima. Quella pubblicava schemi scritti a mano, identici byte per byte a
quelli del braccio LangChain, e garantiva cosi' che il modello ricevesse
la stessa identica descrizione degli strumenti. Il prezzo era che nessuno
dei due ecosistemi esercitava mai la propria generazione di schemi,
mentre e' proprio quella che usa chi adotta l'uno o l'altro — e nel
modello MCP il server lo scrive spesso qualcun altro rispetto a chi
costruisce l'agente, che e' l'intero senso del problema N x M.

I due bracci possono percio' presentare al modello schemi **diversi**.
La differenza e' una proprieta' dei due ecosistemi e va misurata:
``harness/schema_gate.py`` la riporta invece di pretendere che sia nulla.

Gli errori delle operazioni sono ``OperationError`` sollevate da
``shared/operations.py``. Non vengono catturate qui: l'API di alto
livello le traduce in un risultato con ``isError`` a vero, che e' la
convenzione del protocollo, e vale per qualunque eccezione senza che il
codice dello strumento sappia di MCP. ``harness/error_paths.py`` confronta
questa via con quella di LangChain.

Avvio (trasporto stdio, tipicamente lanciato da un client)::

    python -m arm_mcp.server
"""

from __future__ import annotations

import asyncio
import os

from mcp.server import MCPServer

from shared import operations

SERVER_NAME = "event-manager"
SERVER_VERSION = "0.1.0"

#: Il microbenchmark (esperimento A) misura anche uno strumento fittizio
#: che non tocca la rete, per isolare l'overhead di protocollo dalla
#: latenza REST. Viene pubblicato solo su richiesta esplicita: negli
#: esperimenti che coinvolgono il modello la variabile non e' impostata e
#: ``tools/list`` espone le sole operazioni reali.
EXPOSE_ECHO = os.environ.get("BENCH_EXPOSE_ECHO") == "1"


def build_server() -> MCPServer:
    """Registra le operazioni lasciando dedurre gli schemi al server."""
    srv: MCPServer = MCPServer(name=SERVER_NAME, version=SERVER_VERSION)
    for funzione in operations.TOOL_FUNCTIONS:
        srv.tool()(funzione)
    if EXPOSE_ECHO:
        srv.tool()(operations._bench_echo)
    return srv


server = build_server()


async def main() -> None:
    """Serve il protocollo su stdio finche' il client resta connesso."""
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
