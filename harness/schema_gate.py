"""Misura delle differenze fra gli schemi che i due bracci pubblicano.

Era un gate: verificava che i due bracci presentassero al modello gli
stessi identici byte nel campo ``tools``, e falliva altrimenti. Poteva
farlo perche' entrambi derivavano da una definizione scritta a mano.

Ora ciascun braccio lascia derivare lo schema alla propria API di alto
livello — ``MCPServer.tool()`` da un lato, ``@tool`` dall'altro — che e'
il modo in cui questi strumenti vengono scritti nella pratica. I due
schemi possono quindi differire, e **devono poter differire**: la
differenza e' una proprieta' dei due ecosistemi, ed e' un risultato per
il capitolo sull'overhead di implementazione.

Questo modulo la riporta invece di bocciarla. Il confronto resta fatto
sul percorso reale del braccio MCP — ``types.Tool`` serializzato in
JSON-RPC, trasmesso, deserializzato e convertito nel formato OpenAI — e
non su definizioni in memoria, perche' e' in quel giro che compaiono i
campi aggiunti dal modello Pydantic dell'SDK.

Uso::

    BENCH_EXPOSE_ECHO=1 uv run uvicorn arm_mcp.http_server:app --port 8100
    uv run python -m harness.schema_gate
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from arm_langchain.tools import build_tools

MCP_HTTP_URL = os.environ.get("BENCH_MCP_HTTP_URL", "http://127.0.0.1:8100/mcp/")

OK = "✓"
KO = "✗"


def _mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Converte uno strumento MCP nel formato ``tools`` dell'API OpenAI.

    E' la conversione che l'host MCP standalone dovra' fare comunque:
    NIM non parla MCP, quindi il client traduce. E' il punto in cui la
    tesi osserva che il *function calling* resta il substrato condiviso.

    Il modello ``Tool`` dell'SDK porta anche ``title``, ``annotations``,
    ``output_schema``, ``icons`` e ``meta``. Non hanno corrispondente nel
    formato OpenAI e restano fuori dalla conversione: e' un esempio
    concreto di cio' che il substrato condiviso non sa rappresentare.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


async def collect_mcp() -> list[dict[str, Any]]:
    """Interroga davvero ``tools/list`` su un server MCP in esecuzione."""
    async with streamable_http_client(MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.list_tools()
    return [
        _mcp_tool_to_openai(t)
        for t in result.tools
        if not t.name.startswith("_bench_")
    ]


def collect_langchain() -> list[dict[str, Any]]:
    return [convert_to_openai_tool(t) for t in build_tools()]


def _canonical(tools: list[dict[str, Any]]) -> str:
    """Serializza in forma canonica: l'ordine degli strumenti non conta,
    quello delle chiavi nemmeno, il contenuto si'."""
    ordered = sorted(tools, key=lambda t: t["function"]["name"])
    return json.dumps(ordered, sort_keys=True, ensure_ascii=False, indent=2)


async def main() -> int:
    print(f"Server MCP: {MCP_HTTP_URL}\n")
    try:
        mcp_tools = await collect_mcp()
    except Exception as exc:  # noqa: BLE001
        print(f"{KO} Server MCP non raggiungibile: {type(exc).__name__}: {exc}")
        print("  Avvialo con:")
        print("  BENCH_EXPOSE_ECHO=1 uv run uvicorn arm_mcp.http_server:app --port 8100")
        return 1

    lc_tools = collect_langchain()

    print(f"strumenti da MCP (tools/list) : {len(mcp_tools)}")
    print(f"strumenti da LangChain        : {len(lc_tools)}\n")

    mcp_s, lc_s = _canonical(mcp_tools), _canonical(lc_tools)

    if mcp_s == lc_s:
        print(f"{OK} I due bracci pubblicano schemi identici.")
        print(f"  {len(mcp_s)} caratteri per parte.")
        return 0

    print("I due schemi differiscono. Non e' un errore: e' la misura.")
    print(f"  MCP      : {len(mcp_s)} caratteri")
    print(f"  LangChain: {len(lc_s)} caratteri")
    print(f"  scarto   : {len(mcp_s) - len(lc_s):+d} caratteri "
          f"({(len(mcp_s) - len(lc_s)) / len(lc_s):+.1%})")
    print()
    _print_diff(lc_s, mcp_s)
    # Esce con zero: una differenza va riportata, non trattata come guasto.
    return 0


def _print_diff(expected: str, actual: str) -> None:
    import difflib

    diff = difflib.unified_diff(
        expected.splitlines(), actual.splitlines(), "atteso", "ottenuto", lineterm="", n=2
    )
    for line in list(diff)[:60]:
        print(f"    {line}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
