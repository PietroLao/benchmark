"""Server MCP che espone le operazioni sull'Event Manager.

Usa l'API di basso livello dell'SDK (``mcp.server.lowlevel.Server``)
invece di ``MCPServer``: quest'ultimo deduce lo schema degli argomenti
dalla firma della funzione, mentre qui gli schemi pubblicati da
``tools/list`` sono presi letteralmente da
``shared.tools_spec.TOOL_SPECS``.

E' una scelta metodologica, non stilistica: garantisce che il modello
riceva nei due bracci sperimentali la stessa identica descrizione degli
strumenti, requisito senza il quale le differenze misurate non sarebbero
attribuibili al protocollo.

Avvio (trasporto stdio, tipicamente lanciato da un client)::

    python -m arm_mcp.server
"""

from __future__ import annotations

import asyncio
import json

import mcp.types as types
from mcp.server import InitializationOptions, NotificationOptions, ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

import os

from shared import operations
from shared.tools_spec import TOOL_SPECS

SERVER_NAME = "event-manager"
SERVER_VERSION = "0.1.0"

#: Il microbenchmark (esperimento A) misura anche uno strumento fittizio
#: che non tocca la rete, per isolare l'overhead di protocollo dalla
#: latenza REST. Viene pubblicato solo su richiesta esplicita: negli
#: esperimenti che coinvolgono il modello la variabile non e' impostata e
#: ``tools/list`` coincide esattamente con ``TOOL_SPECS``.
EXPOSE_ECHO = os.environ.get("BENCH_EXPOSE_ECHO") == "1"

_ECHO_TOOL = types.Tool(
    name="_bench_echo",
    description="Strumento di misura interno: restituisce l'argomento ricevuto.",
    inputSchema={
        "type": "object",
        "properties": {"payload": {"type": "string"}},
        "required": [],
    },
)


async def on_list_tools(
    context: ServerRequestContext[object],
    params: types.PaginatedRequestParams | None,
) -> types.ListToolsResult:
    """Pubblica la superficie degli strumenti definita una sola volta."""
    tools = [
        types.Tool(
            name=spec.name,
            description=spec.description,
            inputSchema=spec.input_schema,
        )
        for spec in TOOL_SPECS
    ]
    if EXPOSE_ECHO:
        tools.append(_ECHO_TOOL)
    return types.ListToolsResult(tools=tools)


async def on_call_tool(
    context: ServerRequestContext[object],
    params: types.CallToolRequestParams,
) -> types.CallToolResult:
    """Esegue l'operazione richiesta e ne restituisce il risultato JSON.

    L'esecuzione e' sincrona: le operazioni sono chiamate HTTP brevi
    verso un server locale, e delegarle a un thread introdurrebbe nella
    misura un costo di scheduling assente nel braccio nativo.
    """
    result = operations.call(params.name, params.arguments)
    is_error = isinstance(result, dict) and result.get("error") is True
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, default=str),
            )
        ],
        isError=is_error,
    )


server: Server[object] = Server(
    SERVER_NAME,
    version=SERVER_VERSION,
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)


async def main() -> None:
    """Serve il protocollo su stdio finche' il client resta connesso."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
