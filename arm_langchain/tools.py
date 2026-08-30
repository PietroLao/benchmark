"""Braccio LangChain — strumenti costruiti con il decoratore ``@tool``.

E' l'API di alto livello, quella che si usa nella pratica: si passa la
funzione, e il framework ne deriva lo schema dalle annotazioni di tipo e
dalla docstring.

Non c'e' piu' alcuno schema scritto a mano. Passarne uno — com'era prima,
sotto forma di dizionario in ``args_schema`` — otteneva che i due bracci
presentassero al modello schemi identici byte per byte, ma al prezzo di
non esercitare mai la generazione di schemi propria di LangChain, che e'
esattamente la parte che chi adotta il framework usa davvero.

La conseguenza va enunciata: i due bracci possono ora presentare al
modello schemi **diversi**, e la differenza non e' rumore da eliminare ma
una proprieta' dei due ecosistemi. ``harness/schema_gate.py`` non la
verifica piu', la misura.

Le funzioni sono quelle di ``shared/operations.py``, unica
implementazione delle chiamate REST: i bracci differiscono per come lo
strumento viene esposto, mai per cosa lo strumento fa. Vengono
registrate **sincrone**, come sono, in entrambi i bracci: qualunque cosa
ciascun ecosistema faccia per invocarle da un contesto asincrono fa parte
di cio' che si sta misurando, e sceglierla noi la nasconderebbe.

Gli errori richiedono un adattamento, ed e' esso stesso una misura.
Perche' l'agente possa riprendersi da uno strumento che fallisce, invece
di terminare, servono **due** condizioni insieme, verificate:

* lo strumento deve sollevare ``ToolException`` — l'eccezione del
  framework: una qualunque altra viene rilanciata e uccide l'esecuzione;
* lo strumento deve avere ``handle_tool_error`` a vero, che non e' il
  valore predefinito.

Ne manca una e l'esecuzione termina. ``create_agent`` non espone alcun
parametro per la seconda, e il gestore predefinito del suo ``ToolNode``
restituisce al modello i soli errori di *invocazione*, rilanciando tutto
cio' che lo strumento solleva al proprio interno.

Il costo di quell'adattamento e' il dato: per rendere ripristinabile uno
strumento, LangChain chiede che la logica applicativa importi un tipo di
eccezione del framework. ``shared/operations.py`` resta percio' agnostico
e solleva ``OperationError``; la conversione avviene qui, in un involucro
che appartiene al braccio. Il server MCP non ha bisogno di nulla di
simile: traduce qualunque eccezione in un risultato con ``isError``, e il
codice dello strumento ignora l'esistenza del protocollo. Il confronto fra
le due vie sta in ``harness/error_paths.py``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from langchain_core.tools import BaseTool, ToolException, tool

from shared.operations import TOOL_FUNCTIONS, OperationError


def _ripristinabile(funzione: Callable[..., Any]) -> Callable[..., Any]:
    """Converte ``OperationError`` in ``ToolException``.

    ``functools.wraps`` non e' cosmetico: ``@tool`` deriva lo schema con
    ``inspect.signature``, che segue ``__wrapped__``. Senza, l'involucro
    presenterebbe al modello ``(*args, **kwargs)`` e lo schema andrebbe
    perduto.
    """

    @functools.wraps(funzione)
    def _run(*args: Any, **kwargs: Any) -> Any:
        try:
            return funzione(*args, **kwargs)
        except OperationError as exc:
            raise ToolException(str(exc)) from exc

    return _run


def build_tools() -> list[BaseTool]:
    """Costruisce gli strumenti lasciando dedurre lo schema a LangChain."""
    strumenti = []
    for funzione in TOOL_FUNCTIONS:
        strumento = tool(_ripristinabile(funzione))
        strumento.handle_tool_error = True
        strumenti.append(strumento)
    return strumenti


TOOLS_BY_NAME = {t.name: t for t in build_tools()}
