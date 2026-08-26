"""Fase 3 — agente ReAct LangChain.

E' il braccio LangChain dell'esperimento con il modello: gli strumenti
sono ``StructuredTool`` costruiti da ``tools_spec``, il ciclo e' quello
di ``create_agent``, e il modello e' raggiunto tramite ``ChatNVIDIA``.

Il prompt di sistema, i compiti e il limite di iterazioni provengono da
``shared.tasks``, gli stessi che usa l'host MCP. E' la condizione perche'
le differenze misurate siano attribuibili al meccanismo di integrazione
anziche' a una diversa formulazione della richiesta.

Il framework non espone le chiamate al modello, che avvengono al suo
interno. Vengono percio' raccolte con un ``callback``, e i messaggi
normalizzati nel formato dell'API OpenAI — lo stesso in cui l'host MCP
li registra — cosi' che le tracce dei due bracci siano direttamente
confrontabili.

Uso::

    uv run python -m arm_langchain.agent --task t2_disambiguazione
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, convert_to_openai_messages
from langchain_core.outputs import LLMResult

from arm_langchain import wire
from arm_langchain.tools import build_tools
from shared.env import ENV_PATH  # carica .env, se presente  # noqa: F401
from harness.trace import RunTrace
from shared.tasks import MAX_ITERATIONS, SYSTEM_PROMPT, TASKS, TASKS_BY_ID, Task

ARM = "langchain"

#: Errori dell'endpoint condiviso che vale la pena ritentare. Il braccio
#: MCP li ritenta gia' tramite ``shared.nim``; ``ChatNVIDIA`` non espone
#: alcun parametro di ritentativo, quindi senza questa aggiunta i due
#: bracci avrebbero politiche diverse davanti agli stessi guasti. Non
#: sarebbe un risultato ma un artefatto: nella prima campagna il braccio
#: LangChain aveva fallito 7 esecuzioni su 10, quasi tutte per 429, 503 o
#: timeout che il braccio MCP superava semplicemente riprovando.
_TRANSITORI = ("[429]", "[502]", "[503]", "[504]", "Timeout", "timeout")

#: Un 500 puo' essere permanente: se l'endpoint non sa elaborare il
#: prompt, riprovare non lo rende valido.
_PERMANENTI = ("invalid operation", "Failed to apply prompt template")


def _e_transitorio(exc: BaseException) -> bool:
    testo = str(exc)
    if any(s in testo for s in _PERMANENTI):
        return False
    return any(s in testo for s in _TRANSITORI) or "[500]" in testo


def _messaggio_completo(message: BaseMessage) -> dict[str, Any]:
    """Normalizza la risposta del modello senza perderne i campi propri.

    ``convert_to_openai_messages`` restituisce solo i campi previsti dal
    formato OpenAI, mentre ``ChatNVIDIA`` deposita quelli specifici del
    fornitore in ``additional_kwargs``. Il ragionamento di ``gpt-oss``
    finisce li': senza questo recupero la traccia mostrerebbe zero
    caratteri di ragionamento in ogni chiamata — cosa che e' realmente
    accaduta, su trentatre' chiamate su trentatre', facendo sembrare
    assente un contenuto che il modello aveva prodotto e che i token in
    uscita gia' contavano.
    """
    fuori = convert_to_openai_messages([message])[0]
    extra = getattr(message, "additional_kwargs", None) or {}
    for campo in ("reasoning_content", "reasoning"):
        if extra.get(campo) is not None and campo not in fuori:
            fuori[campo] = extra[campo]
    return fuori


class TraceCallback(BaseCallbackHandler):
    """Registra nella traccia cio' che il framework fa internamente.

    Senza questo, del braccio LangChain si osserverebbe solo l'esito: le
    chiamate al modello e le invocazioni degli strumenti avvengono dentro
    il grafo e non sono altrimenti visibili.
    """

    def __init__(
        self,
        trace: RunTrace,
        tools_payload: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> None:
        self.trace = trace
        self.tools_payload = tools_payload
        # I parametri di generazione vivono nell'oggetto modello e non
        # transitano dal callback: vanno riportati a mano, altrimenti la
        # traccia di questo braccio risulterebbe priva di campi che
        # quella dell'host MCP registra, e il confronto fra le due
        # segnalerebbe una differenza inesistente.
        self.params = params or {}
        self._llm_started: dict[UUID, float] = {}
        self._tool_started: dict[UUID, tuple[float, str, Any]] = {}

    # --- modello ---------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_started[run_id] = time.perf_counter()
        # Conservato per l'``on_llm_end``, che non riceve i messaggi.
        self._pending_request = {
            "model": self.trace.model,
            "messages": convert_to_openai_messages(messages[0]),
            "tools": self.tools_payload,
            **self.params,
        }

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._llm_started.pop(run_id, time.perf_counter())
        # Il payload realmente trasmesso ha la precedenza sulla
        # ricostruzione: e' l'unico confrontabile con quello dell'host MCP.
        richiesta = wire.ultimo() or getattr(self, "_pending_request", {})
        usage = (response.llm_output or {}).get("token_usage") or {}
        if not usage:
            # Alcune integrazioni riportano il consumo sul messaggio
            # anziche' nel campo di riepilogo.
            try:
                usage = response.generations[0][0].message.usage_metadata or {}
                usage = {
                    "prompt_tokens": usage.get("input_tokens"),
                    "completion_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            except (AttributeError, IndexError, TypeError):
                usage = {}

        generation = response.generations[0][0] if response.generations else None
        body = {
            "choices": [
                {
                    "message": (
                        _messaggio_completo(generation.message)
                        if generation is not None
                        and hasattr(generation, "message")
                        else {}
                    ),
                    "finish_reason": (
                        (generation.generation_info or {}).get("finish_reason")
                        if generation is not None
                        else None
                    ),
                }
            ],
            "usage": usage,
        }
        self.trace.llm_call(richiesta, body, elapsed)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._llm_started.pop(run_id, time.perf_counter())
        self.trace.llm_call(
            getattr(self, "_pending_request", {}), None, elapsed, error=str(error)
        )

    # --- strumenti -------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "?")
        self._tool_started[run_id] = (time.perf_counter(), name, inputs or input_str)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        start, name, args = self._tool_started.pop(
            run_id, (time.perf_counter(), "?", None)
        )
        content = getattr(output, "content", output)
        self.trace.tool_call(
            name,
            args,
            content,
            (time.perf_counter() - start) * 1000,
            via="langchain/ToolNode",
            is_error=getattr(output, "status", None) == "error",
        )


def _modello_con_ritentativi(base_cls: type) -> type:
    """Crea una sottoclasse di ``ChatNVIDIA`` con ritentativi e chiamate
    a strumento serializzate.

    Si interviene per sottoclasse e non con ``with_retry`` perche'
    quest'ultimo restituisce un ``RunnableRetry``, che non espone
    ``bind_tools``: ``create_agent`` non potrebbe piu' collegargli gli
    strumenti. La sottoclasse eredita tutto e resta un modello a tutti
    gli effetti.
    """

    class ChatNVIDIARobusto(base_cls):  # type: ignore[misc, valid-type]
        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            # Serializza le chiamate a strumento: l'endpoint rifiuta con
            # un 500 le risposte che ne contengono piu' di una.
            kwargs.setdefault("parallel_tool_calls", False)
            for attempt in range(MAX_ATTEMPTS_LC):
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not _e_transitorio(exc) or attempt == MAX_ATTEMPTS_LC - 1:
                        raise
                    await asyncio.sleep(min(2**attempt, 30))
            raise AssertionError("irraggiungibile")

    return ChatNVIDIARobusto


#: Stesso numero di tentativi di ``shared.nim.MAX_ATTEMPTS``: i due
#: bracci devono reagire agli stessi guasti nello stesso modo.
MAX_ATTEMPTS_LC = 4


async def run_task(
    task: Task,
    *,
    model: str,
    max_iterations: int = MAX_ITERATIONS,
) -> RunTrace:
    """Esegue un compito e restituisce la traccia completa dell'esecuzione.

    L'esecuzione e' asincrona perche' gli strumenti sono definiti come
    corotine: e' la forma in cui ``ToolNode`` li invoca quando il grafo
    gira in modo asincrono, ed e' la stessa misurata nell'esperimento A.
    Invocarli per via sincrona fallirebbe.
    """
    from langchain.agents import create_agent
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    wire.installa()
    wire.azzera()

    tools = build_tools()
    tools_payload = [convert_to_openai_tool(t) for t in tools]

    trace = RunTrace(
        arm=ARM,
        model=model,
        task_id=task.task_id,
        prompt=task.prompt,
        config={
            "runtime": "create_agent",
            "max_iterations": max_iterations,
            "system_prompt": SYSTEM_PROMPT,
            "n_tools": len(tools),
        },
    )

    params = {
        "temperature": 0.0,
        "max_completion_tokens": 1024,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    llm = _modello_con_ritentativi(ChatNVIDIA)(
        model=model, temperature=params["temperature"],
        max_completion_tokens=params["max_completion_tokens"],
    )
    agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    callback = TraceCallback(trace, tools_payload, params)
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": task.prompt}]},
            config={"callbacks": [callback], "recursion_limit": max_iterations * 2},
        )
    except Exception as exc:  # noqa: BLE001 — va registrato, non propagato
        trace.event("errore_agente", detail=f"{type(exc).__name__}: {exc}")
        trace.finish(status="errore_agente")
        return trace

    final = result["messages"][-1]
    answer = getattr(final, "content", "") or ""
    trace.finish(
        final_answer=answer,
        status="ok" if task.check(answer) else "risposta_errata",
    )
    return trace


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="id del compito; vuoto = tutti")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parsed = parser.parse_args()

    for task in [TASKS_BY_ID[parsed.task]] if parsed.task else list(TASKS):
        trace = await run_task(task, model=parsed.model)
        m = trace.metrics()
        print(
            f"[{ARM}] {task.task_id:<22} {trace.status:<16} "
            f"iterazioni={m['n_llm_calls']} strumenti={m['n_tool_calls']} "
            f"tempo={m['latency_llm_s']:.1f}s"
        )
        print(f"        risposta: {(trace.final_answer or '')[:100]}")
        print(f"        traccia:  {trace.save().name}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
