"""Braccio LangChain — costruzione degli strumenti da ``TOOL_SPECS``.

Gli strumenti sono ``StructuredTool`` costruiti passando **direttamente**
lo schema JSON di ``shared/tools_spec.py`` come ``args_schema``, anziche'
derivandolo da un modello Pydantic o dalle annotazioni della funzione.

La scelta e' deliberata e ha una conseguenza non ovvia, verificata
sperimentalmente:

* con ``args_schema`` come **dizionario**, lo schema che LangChain invia
  al modello e' letteralmente quello di ``TOOL_SPECS``, e gli argomenti
  arrivano alla funzione **non convertiti**;
* con ``args_schema`` come **modello Pydantic**, LangChain converte i tipi
  al confine dello strumento (``"1"`` diventa ``1``) ma lo schema inviato
  al modello e' rigenerato da Pydantic, quindi puo' divergere dal nostro.

Per il confronto serve la prima: se i due bracci presentassero al modello
schemi diversi, le differenze misurate non sarebbero piu' attribuibili al
protocollo. La conversione dei tipi resta comunque garantita, perche'
entrambi i bracci invocano ``operations.call``, che applica
``coerce_arguments``.

Il fatto stesso e' un risultato da riportare in §3.2: la validazione degli
argomenti che LangChain sembra offrire gratuitamente si ottiene solo
definendo gli strumenti con modelli Pydantic; definirli dallo schema JSON
grezzo — necessario per garantire la parita' con un server MCP — vi
rinuncia. Non e' quindi una differenza fra i due approcci, ma una scelta
interna a LangChain.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from shared import operations
from shared.tools_spec import TOOL_SPECS


def make_tool(name: str, description: str, schema: dict[str, Any]) -> StructuredTool:
    """Costruisce un singolo strumento che inoltra a ``operations.call``."""

    async def _run(**kwargs: Any) -> Any:
        return operations.call(name, kwargs)

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=description,
        args_schema=schema,
    )


def build_tools() -> list[StructuredTool]:
    """Costruisce gli strumenti LangChain a partire dalle specifiche condivise.

    ``_bench_echo`` non compare: e' visibile solo al microbenchmark, che
    lo costruisce per conto proprio.
    """
    return [make_tool(s.name, s.description, s.input_schema) for s in TOOL_SPECS]


TOOLS_BY_NAME = {t.name: t for t in build_tools()}
