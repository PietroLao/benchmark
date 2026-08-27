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

Due condizioni
--------------

L'host si esegue in due forme, e la distinzione e' metodologica.

``mcp`` e' la forma **idiomatica**: quella che scriverebbe chi costruisce
un host senza avere un framework da imitare. Non rimanda al modello il
ragionamento dei giri precedenti e include ``name`` sui messaggi di
strumento.

``mcp-conforme`` riproduce sul filo **esattamente** cio' che trasmette
LangChain, verificato con ``arm_langchain/wire.py``: ritrasmette
``reasoning`` e ``reasoning_content`` a ogni giro e omette ``name``.

Servono entrambe perche' pareggiare il contesto e' l'unico modo di
attribuire al meccanismo le differenze osservate, ma pareggiarlo cancella
una differenza che e' reale e che chi adotta LangChain subisce davvero.
Il confronto ``mcp-conforme`` contro ``langchain`` isola il meccanismo;
``mcp`` contro ``langchain`` dice cosa si ottiene in pratica; ``mcp``
contro ``mcp-conforme`` attribuisce lo scarto alla gestione del contesto
del framework. Misurato in precedenza: +33 token in ingresso alla seconda
interrogazione, +84 alla terza.

Uso::

    uv run python -m arm_mcp.host --task t3_join_titoli
    uv run python -m arm_mcp.host --task t3_join_titoli --conforme
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
ARM_CONFORME = "mcp-conforme"


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
    conforme: bool = False,
) -> RunTrace:
    """Esegue un compito e restituisce la traccia completa dell'esecuzione.

    ``conforme`` sceglie fra le due condizioni descritte in cima al modulo:
    a ``False`` l'host si comporta come lo scriverebbe il suo autore, a
    ``True`` riproduce sul filo esattamente cio' che trasmette LangChain.
    """
    trace = RunTrace(
        arm=ARM_CONFORME if conforme else ARM,
        model=model,
        task_id=task.task_id,
        prompt=task.prompt,
        config={
            "transport": "streamable_http",
            "session": "persistent",
            "max_iterations": max_iterations,
            "system_prompt": SYSTEM_PROMPT,
            "url": url,
            "conforme": conforme,
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
                # ``annotations``, ``audio`` e ``refusal``.
                #
                # Nella condizione conforme si aggiungono i due campi di
                # ragionamento, perche' ``ChatNVIDIA`` li conserva in
                # ``additional_kwargs`` e li ritrasmette a ogni giro. Nella
                # condizione idiomatica si omettono: e' testo che il
                # modello ha gia' prodotto e che nessun autore di host
                # rimanderebbe indietro. Lo scarto fra le due misura il
                # costo di quella scelta del framework, ed e' il motivo per
                # cui entrambe vengono eseguite.
                canonico: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content"),
                }
                if conforme:
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
                    # ``name`` e' facoltativo nel formato OpenAI. Il payload
                    # reale di LangChain porta solo ``content``, ``role`` e
                    # ``tool_call_id``, quindi nella condizione conforme si
                    # omette; nella condizione idiomatica si include, che e'
                    # la forma piu' esplicita e quella che si scriverebbe
                    # non avendo un framework da imitare.
                    messaggio_strumento: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": text,
                    }
                    if not conforme:
                        messaggio_strumento["name"] = fn["name"]
                    messages.append(messaggio_strumento)

            trace.event("limite_iterazioni", limit=max_iterations)
            trace.finish(status="limite_iterazioni")
            return trace


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="id del compito; vuoto = tutti")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--url", default=MCP_HTTP_URL)
    parser.add_argument(
        "--conforme",
        action="store_true",
        help="riproduce sul filo cio' che trasmette LangChain",
    )
    parsed = parser.parse_args()

    arm = ARM_CONFORME if parsed.conforme else ARM
    tasks = [TASKS_BY_ID[parsed.task]] if parsed.task else list(TASKS)
    for task in tasks:
        trace = await run_task(
            task, model=parsed.model, url=parsed.url, conforme=parsed.conforme
        )
        m = trace.metrics()
        print(
            f"[{arm}] {task.task_id:<22} {trace.status:<16} "
            f"iterazioni={m['n_llm_calls']} strumenti={m['n_tool_calls']} "
            f"tempo={m['latency_llm_s']:.1f}s"
        )
        print(f"        risposta: {(trace.final_answer or '')[:100]}")
        print(f"        traccia:  {trace.save().name}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
