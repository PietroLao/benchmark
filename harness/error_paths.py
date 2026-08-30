"""Come i due approcci segnalano il fallimento di uno strumento.

Non e' un compito di campagna, ed e' una scelta. La differenza qui
misurata e' **deterministica**: non serve un modello per osservarla, e
ripeterla su piu' esecuzioni non aggiungerebbe nulla. Un compito di
campagna che la esercitasse produrrebbe soltanto un esito binario, senza
alcun ciclo agentico da confrontare.

E' materiale per l'asse "overhead di implementazione", non per le metriche
di comportamento.

Cosa emerge
-----------

Il server MCP traduce **qualunque** eccezione in un risultato con
``isError`` a vero, e il codice dello strumento ignora l'esistenza del
protocollo.

LangChain sa fare altrettanto, ma a due condizioni congiunte: lo strumento
deve sollevare ``ToolException`` — la sua eccezione, non una qualsiasi — e
deve essere costruito con ``handle_tool_error`` a vero, che non e' il
valore predefinito. Ne manca una e l'esecuzione dell'agente termina.

``create_agent`` non espone alcun parametro per la seconda: costruisce il
proprio ``ToolNode`` con il gestore predefinito, che restituisce al
modello i soli errori di *invocazione* e rilancia tutto cio' che lo
strumento solleva al proprio interno. Per cambiarlo servono un middleware
``wrap_tool_call`` o la costruzione del grafo a mano — in entrambi i casi,
uscire dall'API di alto livello.

Il dato non e' quindi "LangChain non sa riprendersi", ma il prezzo che
chiede per farlo: perche' uno strumento sia ripristinabile, la logica
applicativa deve importare un tipo di eccezione del framework. E' un
accoppiamento che con MCP non esiste, ed e' precisamente quello che il
protocollo si propone di evitare.

``arm_langchain/tools.py`` paga quel prezzo in un involucro che appartiene
al braccio, cosi' che ``shared/operations.py`` resti agnostico e i due
bracci abbiano la stessa capacita' di ripresa. Senza, il confronto
sarebbe fra un agente che si riprende e uno che muore.

Uso::

    uv run uvicorn server.wrapper:app --port 8000
    uv run uvicorn arm_mcp.http_server:app --port 8100
    uv run python -m harness.error_paths
"""

from __future__ import annotations

import asyncio
import functools
import sys
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from shared import operations
from shared.operations import OperationError

MCP_HTTP_URL = "http://127.0.0.1:8100/mcp/"

#: ``lferrari`` risulta gia' iscritta all'evento 3 nei dati di prova:
#: l'operazione fallisce senza bisogno di alterare nulla.
CONFLITTO = {
    "event_id": 3,
    "username": "lferrari",
    "name": "Laura Ferrari",
    "email": "laura.ferrari@example.it",
}

#: Un argomento che nessuna conversione puo' salvare: verifica il secondo
#: percorso, quello della validazione, dove i due si comportano uguale.
ARGOMENTO_INVALIDO = {**CONFLITTO, "event_id": "abc"}


class _Stato(TypedDict):
    messages: Annotated[list, add_messages]


def _grafo(strumento: Any) -> Any:
    g = StateGraph(_Stato)
    g.add_node("tools", ToolNode([strumento]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


def _avvolgi(funzione: Callable[..., Any], eccezione: type[Exception] | None) -> Any:
    """Ricostruisce la funzione convertendo l'errore, o lasciandolo passare."""

    @functools.wraps(funzione)
    def _run(*args: Any, **kwargs: Any) -> Any:
        try:
            return funzione(*args, **kwargs)
        except OperationError as exc:
            if eccezione is None:
                raise
            raise eccezione(str(exc)) from exc

    return _run


async def _mcp(argomenti: dict[str, Any]) -> str:
    async with streamable_http_client(MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as sess:
            await sess.initialize()
            r = await sess.call_tool("register_user_to_event", argomenti)
    testo = r.content[0].text if r.content else ""
    return f"is_error={r.is_error}  {testo[:58]!r}"


async def _langchain(strumento: Any, argomenti: dict[str, Any]) -> str:
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "register_user_to_event",
                "args": argomenti,
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    try:
        out = await _grafo(strumento).ainvoke({"messages": [msg]})
    except Exception as exc:  # noqa: BLE001 — e' l'esito da osservare
        return f"ESECUZIONE TERMINATA — {type(exc).__name__}"
    tm = out["messages"][-1]
    return f"status={getattr(tm, 'status', '?')}  {str(tm.content)[:58]!r}"


def _configura(eccezione: type[Exception] | None, gestisci: bool) -> Any:
    strumento = tool(_avvolgi(operations.register_user_to_event, eccezione))
    strumento.handle_tool_error = gestisci
    return strumento


async def main() -> int:
    print("Errore sollevato DENTRO lo strumento (iscrizione gia' esistente)")
    print(f"  MCP, qualunque eccezione            {await _mcp(CONFLITTO)}")
    for etichetta, ecc, gest in (
        ("LangChain, eccezione propria        ", None, False),
        ("LangChain, ToolException            ", ToolException, False),
        ("LangChain, ToolException + handle   ", ToolException, True),
    ):
        print(f"  {etichetta}{await _langchain(_configura(ecc, gest), CONFLITTO)}")

    print()
    print("Argomento non convertibile (event_id='abc')")
    print(f"  MCP                                 {await _mcp(ARGOMENTO_INVALIDO)}")
    print(
        "  LangChain                           "
        f"{await _langchain(_configura(ToolException, True), ARGOMENTO_INVALIDO)}"
    )

    print()
    print("La configurazione del braccio e' l'ultima del primo blocco:")
    print("ToolException piu' handle_tool_error, cioe' il prezzo che LangChain")
    print("chiede perche' la logica applicativa sia ripristinabile.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
