"""Fase 2 — host agentico MCP autonomo.

Realizza il ciclo osserva-decidi-agisci parlando direttamente con il
protocollo e con l'API del modello, senza alcun framework di
orchestrazione. E' il braccio MCP dell'esperimento con il modello.

Gli strumenti vengono **scoperti** tramite ``tools/list`` e non importati
da ``shared.tools_spec``: il gate sugli schemi ha gia' verificato che le
due vie coincidono byte per byte, quindi passare per il protocollo non
introduce differenze e rende il braccio autentico — un host MCP reale non
sa nulla di come il server e' stato scritto.

La sessione MCP e' **persistente** per l'intera esecuzione: si apre una
volta e si riusa per tutte le invocazioni. E' la configurazione sensata,
ed e' anche quella che l'esperimento A ha mostrato costare circa otto
volte meno di riaprirla a ogni chiamata.

Uso::

    uv run python -m arm_mcp.host --task t2_disambiguazione
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from harness.trace import RunTrace
from shared import nim
from shared.tasks import MAX_ITERATIONS, SYSTEM_PROMPT, TASKS, TASKS_BY_ID, Task

MCP_HTTP_URL = "http://127.0.0.1:8100/mcp/"
ARM = "mcp"


def to_openai_tools(tools: list[types.Tool]) -> list[dict[str, Any]]:
    """Converte gli strumenti scoperti nel formato accettato dall'endpoint.

    E' il passaggio obbligato di cui parla la Sezione 2.2 della tesi: il
    modello non conosce MCP, quindi anche un host che non usa alcun
    framework deve tradurre verso il meccanismo di tool calling del
    fornitore.

    Il campo si legge come ``input_schema``: l'SDK 2.x espone i campi in
    forma snake_case, e accetta la grafia del protocollo (``inputSchema``)
    solo come alias in scrittura.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


async def run_task(
    task: Task,
    *,
    model: str,
    url: str = MCP_HTTP_URL,
    max_iterations: int = MAX_ITERATIONS,
) -> RunTrace:
    """Esegue un compito e restituisce la traccia completa dell'esecuzione."""
    trace = RunTrace(
        arm=ARM,
        model=model,
        task_id=task.task_id,
        prompt=task.prompt,
        config={
            "transport": "streamable_http",
            "session": "persistent",
            "max_iterations": max_iterations,
            "system_prompt": SYSTEM_PROMPT,
            "url": url,
        },
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]

    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            listed = await session.list_tools()

            # Gli strumenti che iniziano per "_" sono interni al banco di
            # prova e non devono mai raggiungere il modello. Il server li
            # pubblica solo con BENCH_EXPOSE_ECHO=1, impostata dal
            # microbenchmark; se pero' quel server restasse acceso e lo si
            # riusasse per l'esperimento, il modello vedrebbe uno
            # strumento in piu' e il confronto sarebbe falsato senza alcun
            # segnale. Il filtro rende l'esito indipendente da come il
            # server e' stato avviato.
            interni = [t.name for t in listed.tools if t.name.startswith("_")]
            if interni:
                trace.event("strumenti_interni_esclusi", names=interni)
                print(
                    f"  ! il server pubblica strumenti interni {interni}: esclusi",
                    file=sys.stderr,
                )

            tools = to_openai_tools(
                [t for t in listed.tools if not t.name.startswith("_")]
            )
            trace.config["n_tools"] = len(tools)
            trace.config["tool_names"] = [t["function"]["name"] for t in tools]

            for _ in range(max_iterations):
                try:
                    body, _ = nim.chat(
                        messages, tools, model=model, trace=trace
                    )
                except nim.NimError as exc:
                    trace.event("errore_llm", detail=str(exc))
                    trace.finish(status="errore_llm")
                    return trace

                message = body["choices"][0]["message"]
                tool_calls = message.get("tool_calls") or []

                # Si riaccoda una forma **canonica** del messaggio, non
                # l'oggetto grezzo restituito dall'API, che porta anche
                # ``annotations``, ``audio`` e ``refusal``. L'insieme dei
                # campi riprodotto qui e' esattamente quello che LangChain
                # trasmette sul filo, verificato intercettando il payload
                # HTTP reale del braccio: ``content``, ``reasoning``,
                # ``reasoning_content``, ``role``, ``tool_calls``.
                #
                # I due campi di ragionamento vanno riaccodati anche se
                # sembrano superflui. ``ChatNVIDIA`` li conserva in
                # ``additional_kwargs`` e li rimanda al modello a ogni
                # giro; ometterli qui significherebbe presentare al
                # modello un contesto piu' povero nel braccio MCP. La
                # divergenza era misurabile e cresceva con la profondita'
                # della conversazione: +33 token in ingresso alla seconda
                # interrogazione, +84 alla terza.
                canonico: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content"),
                }
                for campo in ("reasoning_content", "reasoning"):
                    if message.get(campo) is not None:
                        canonico[campo] = message[campo]
                if tool_calls:
                    canonico["tool_calls"] = tool_calls
                messages.append(canonico)

                if not tool_calls:
                    answer = message.get("content") or ""
                    trace.finish(
                        final_answer=answer,
                        status="ok" if task.check(answer) else "risposta_errata",
                    )
                    return trace

                for call in tool_calls:
                    fn = call["function"]
                    try:
                        args = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                        trace.event("argomenti_illeggibili", raw=fn["arguments"])

                    start = time.perf_counter_ns()
                    result = await session.call_tool(fn["name"], args)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6

                    text = result.content[0].text if result.content else ""
                    trace.tool_call(
                        fn["name"],
                        args,
                        text,
                        elapsed_ms,
                        via="mcp/tools_call",
                        is_error=result.is_error,
                    )
                    messages.append(
                        {
                            # ``name`` e' facoltativo nel formato OpenAI e
                            # non va incluso: il payload HTTP reale di
                            # LangChain porta solo ``content``, ``role`` e
                            # ``tool_call_id``. La ricostruzione prodotta
                            # dal callback lo mostrava, ma non e' cio' che
                            # il framework trasmette.
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": text,
                        }
                    )

            trace.event("limite_iterazioni", limit=max_iterations)
            trace.finish(status="limite_iterazioni")
            return trace


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="id del compito; vuoto = tutti")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--url", default=MCP_HTTP_URL)
    parsed = parser.parse_args()

    tasks = [TASKS_BY_ID[parsed.task]] if parsed.task else list(TASKS)
    for task in tasks:
        trace = await run_task(task, model=parsed.model, url=parsed.url)
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
