# Benchmark MCP vs LangChain

Codice della valutazione sperimentale (Capitolo 4 della tesi). Confronta
due approcci all'integrazione di strumenti esterni in un agente basato su
LLM, tenendo fissi il modello e la risorsa esterna:

* **braccio MCP** — server MCP conforme + host agentico standalone;
* **braccio LangChain** — strumenti `@tool` + agente ReAct.

I due bracci sono implementati ciascuno nel modo idiomatico del proprio
ecosistema. Il confronto e' quindi tra *approcci di orchestrazione*, non
tra due modi di usare lo stesso framework: instradare MCP attraverso
`langchain-mcp-adapters` misurerebbe LangChain-che-parla-MCP, non MCP.

## Il controllo su cui poggia tutto

Il rischio del disegno a due loop e' che i bracci presentino al modello
input diversi, rendendo le differenze misurate non attribuibili al
protocollo. Tre presidi:

1. **`shared/tools_spec.py`** definisce nome, descrizione e schema degli
   argomenti **una sola volta**. Il server MCP li pubblica letteralmente
   (per questo usa l'API di basso livello dell'SDK e non `MCPServer`, che
   li dedurrebbe dalle firme); il braccio LangChain li passa come
   `args_schema` sotto forma di dizionario, per la stessa ragione.
   Verificato: gli schemi che i due bracci presentano al modello
   coincidono byte per byte.
2. **`shared/operations.py`** e' l'unica implementazione delle chiamate
   REST. I bracci cambiano come lo strumento e' *esposto*, mai cosa *fa*.
3. **Diff dei body HTTP** inviati a `/v1/chat/completions` nei due
   bracci: se `tools` e `messages` coincidono, il modello riceve lo
   stesso input.

`harness/schema_gate.py` verifica il campo `tools` e **passa**: gli
schemi vengono raccolti da un vero `tools/list` su un server MCP in
esecuzione, quindi dopo serializzazione JSON-RPC, trasmissione e
deserializzazione — non confrontando definizioni in memoria, che
sarebbero identiche per costruzione. Resta da verificare `messages`, che
dipende dai loop agentici e quindi dalle fasi 2 e 3.

```bash
uv run python -m harness.schema_gate
```

## Stato

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | Smoke test tool calling su NIM | **superata** |
| 1 | Server strumentato, fixture, esperimento A | **completata** |
| 2 | Host MCP standalone con LLM | da fare |
| 3 | Braccio LangChain | da fare |
| 4 | Diff degli schemi (gate) | **verde** sugli schemi; `messages` da fare |
| 5 | Caratterizzazione rumore `t_llm` | da fare |
| 6 | Campagna completa | da fare |
| 7 | Analisi e tabelle LaTeX | da fare |

## Struttura

```
shared/tools_spec.py     definizione unica degli strumenti
shared/operations.py     unica implementazione delle chiamate REST
server/wrapper.py        avvolge l'Event Manager: conteggio REST + reset
server/fixture.py        dataset deterministico e leggibile
arm_mcp/server.py        server MCP (API di basso livello, schemi espliciti)
arm_mcp/http_server.py   lo stesso server su trasporto Streamable HTTP
arm_langchain/tools.py   strumenti LangChain derivati da tools_spec
harness/smoke_nim.py     fase 0: verifica del tool calling su NIM
microbench/transport.py  esperimento A: overhead di trasporto, senza LLM
results/                 un JSON per esecuzione
```

## Requisiti

Il progetto Event Manager non viene modificato: e' importato dall'esterno.
Percorso configurabile con `EVENT_MANAGER_ROOT` (default:
`~/Desktop/ProgettoProgrammazioneWeb2026`).

```bash
uv sync
```

## Uso

Avviare il server strumentato:

```bash
uv run uvicorn server.wrapper:app --port 8000
```

Ripristinare lo stato iniziale e leggere i conteggi:

```bash
curl -X POST http://127.0.0.1:8000/__bench__/reset
curl http://127.0.0.1:8000/__bench__/counters/<run_id>
```

Avviare il server MCP su HTTP (necessario per le condizioni HTTP
dell'esperimento A; senza, quelle condizioni vengono saltate con un
avviso invece di far fallire l'esecuzione):

```bash
BENCH_EXPOSE_ECHO=1 uv run uvicorn arm_mcp.http_server:app --port 8100
```

Eseguire l'esperimento A:

```bash
uv run python -m microbench.transport --iterations 100 --warmup 20
```

Eseguire la fase 0 (richiede la chiave, mai scritta su file):

```bash
uv run python -m harness.smoke_nim
```

## Note metodologiche

**Alternanza delle condizioni e riscaldamento.** Le misure girano su un
portatile senza ventola. Le condizioni sono alternate a rotazione anziche'
eseguite in blocco, cosi' che l'eventuale deriva termica colpisca tutte le
condizioni allo stesso modo; le prime iterazioni sono scartate.

**Statistiche robuste.** Mediana e IQR, non media e deviazione standard:
la distribuzione delle latenze e' asimmetrica e con code lunghe.

**Alternanza dei bracci nella campagna (fase 6).** Due esecuzioni
identiche della fase 0 hanno dato 18–164 s con un ritentativo 503 e
8.9–25.7 s senza ritentativi. La dispersione *fra* sessioni supera quindi
quella *entro* una sessione, e non riguarda solo la latenza: un endpoint
carico cambia anche la frequenza dei ritentativi e il troncamento delle
risposte, quindi puo' toccare il **numero di iterazioni**, che e' una
delle metriche effettivamente rivendicate. Eseguire un braccio in un
momento e l'altro in un altro non e' rumore ma un fattore confondente.
La campagna alterna percio' i bracci run per run, con la stessa
disciplina di rotazione dell'esperimento A, e i checkpoint cadono solo su
cicli completi. Ogni run registra istante di esecuzione e numero di
ritentativi, cosi' che un effetto di sessione resti rilevabile a
posteriori.

**Lo strumento `_bench_echo`.** Non compare in `TOOL_SPECS` e viene
pubblicato da `tools/list` solo con `BENCH_EXPOSE_ECHO=1`, che imposta il
solo microbenchmark. Serve a isolare l'overhead di protocollo dalla
latenza REST misurando un'operazione a costo intrinseco nullo. Negli
esperimenti con il modello resta invisibile, quindi la parita' degli
schemi non ne e' toccata.

**Il percorso `/users/`.** Il router utenti usa `@router.get("/")` con
prefisso `/users`: chiamare `/users` produrrebbe un redirect 307, cioe'
**due** richieste nel conteggio. `operations.py` usa il percorso esatto.

**Echo SQL disattivato.** L'engine del progetto d'esame e' creato con
`echo=True`; il wrapper lo disattiva a runtime, senza toccare il file,
perche' stamperebbe ogni query su stdout introducendo rumore nella
latenza.

**La coercizione dei tipi, e dove va applicata.** Llama 3.3 restituisce
in modo riproducibile `"event_id": "1"` — una stringa — benche' lo schema
dichiari `integer`. Verificato sperimentalmente: `StructuredTool` converte
la stringa in intero se `args_schema` e' un modello Pydantic, ma **non**
se e' un dizionario di schema JSON. Poiche' la parita' degli schemi
impone il dizionario (vedi sopra), la conversione e' applicata nel
percorso condiviso da `operations.call` tramite `coerce_arguments`, e i
due bracci reagiscono in modo identico allo stesso output del modello.

E' anche un risultato per §3.2: la validazione degli argomenti che
LangChain sembra offrire gratuitamente si ottiene solo definendo gli
strumenti con modelli Pydantic; definirli dallo schema JSON grezzo —
necessario per garantire la parita' con un server MCP — vi rinuncia.

**Divergenza residua da verificare nel gate.** Su argomenti *non*
convertibili (es. `"abc"` per un intero) i bracci non coincidono:
`StructuredTool` con schema Pydantic solleva `ValidationError`, mentre il
percorso condiviso inoltra il valore e lascia che sia l'operazione a
produrre un errore leggibile. Con `args_schema` dizionario — la
configurazione adottata — il problema non si presenta, ma va riverificato
nella fase 4 se la configurazione dovesse cambiare.

## Limiti noti dell'esperimento A

**`t_rest` non e' identica fra i percorsi.** Nella condizione diretta la
chiamata REST parte dal processo principale; nelle condizioni MCP su stdio
parte dal sottoprocesso, con un proprio pool di connessioni. Per questo
l'overhead va letto sull'operazione `echo`, che elimina del tutto la
componente REST; `list_events` serve a collocarlo in un contesto
realistico e a controllare la coerenza interna (vedi sotto).

**L'ordine delle condizioni era una distorsione sistematica, ora
corretta.** Con un ordine fisso, la prima condizione di ogni iterazione
veniva eseguita sempre subito dopo quelle che avviano processi e chiudono
connessioni, pagandone ogni volta gli strascichi. Il sintomo era che
`langchain_tool` risultava *piu' veloce* della chiamata diretta su
`list_events` (-0.108 ms), il che e' impossibile. Isolando le due
condizioni il segno si inverte e l'overhead diventa +0.141 ms, coerente
con i +0.148 ms misurati su `echo`. Le condizioni sono ora permutate a
caso a ogni iterazione, con seme fisso per riproducibilita'.

E' il motivo per cui la tabella qui sotto va letta come misura valida: i
valori assoluti restano piu' alti che in isolamento, perche' le
condizioni pesanti caricano comunque il processo, ma la distorsione non
cade piu' sempre sulla stessa cella.

## Cosa mostra l'esperimento A

Tre esecuzioni con ordine permutato: una da 40 iterazioni e **due
repliche indipendenti da 100** (semi 1001 e 2002), eseguite per
verificare che le cifre rivendicate non fossero un artefatto di sessione.
Mediane sull'operazione `echo`, che esclude la rete:

| condizione | 40 iter | replica 1 | replica 2 |
|---|---|---|---|
| `diretto` | 0.006 ms | 0.005 ms | 0.005 ms |
| `langchain_tool` | 0.395 ms | 0.448 ms | 0.434 ms |
| `mcp_stdio_persistent` | 1.006 ms | 1.051 ms | 1.002 ms |
| `mcp_http_persistent` (¹) | 1.420 ms | 1.404 ms | 1.259 ms |
| `mcp_http_new_session` | 7.571 ms | 8.033 ms | 8.103 ms |
| `mcp_stdio_new_process` | 459.0 ms | 481.0 ms | 459.3 ms |

**Le due cifre di testata si riproducono.**

| | 40 iter | replica 1 | replica 2 |
|---|---|---|---|
| MCP persistente − LangChain | +0.612 ms | +0.603 ms | +0.568 ms |
| avvio processo / totale stdio | 98.4 % | 98.3 % | 98.2 % |

Lo scarto MCP–LangChain varia di 0.044 ms su tre esecuzioni, contro un
effetto di ~0.6 ms: un ordine di grandezza di margine. La quota
attribuibile all'avvio del processo e' stabile entro due decimi di punto
percentuale.

**Il controllo di coerenza.** L'overhead di ciascuna condizione e'
sostanzialmente lo stesso sulle due operazioni, benche' `list_events`
costi ~200 volte `echo`. E' quello che deve accadere se si sta misurando
un costo fisso di protocollo, indipendente da cio' che lo strumento fa —
ed e' proprio cio' che *non* accadeva prima della permutazione.

(¹) `mcp_http_persistent` resta l'unica riga che non supera il controllo,
e le repliche lo confermano: mediana fra 1.259 e 1.420 ms, IQR su `echo`
fra 0.854 e 1.077 ms, cioe' da quattro a dieci volte quello delle altre
condizioni persistenti e piu' ampio dell'effetto rivendicato qui sopra.
Non tocca nessuna delle due conclusioni — la prima usa
`mcp_stdio_persistent`, la seconda le due condizioni a nuova sessione —
ma questa riga non va presentata con la stessa precisione delle altre.

Tre conclusioni.

**Il confronto corretto e' ~+0.6 ms, non ~+1.0 ms.** Rapportare MCP alla
chiamata di funzione nuda ne sovrastima l'overhead di oltre il 60%,
perche' gli attribuisce anche il costo che LangChain paga comunque
(~+0.44 ms). Con una sessione persistente, un `tools/call` MCP costa
circa sei decimi di millisecondo in piu' di uno `StructuredTool`
LangChain.

**Il costo dell'avvio del processo e' il 98% della condizione stdio a
nuova sessione.** La differenza fra `mcp_stdio_new_process` e
`mcp_http_new_session` misura esattamente questo, perche' le due
condizioni differiscono solo per la presenza di un processo da avviare.
Stabilire una nuova sessione MCP costa ~8 ms; i restanti ~460 sono avvio
dell'interprete Python e import dell'SDK. Attribuirli al protocollo
sarebbe un errore, ed e' l'errore che la prima versione dell'esperimento
— priva del trasporto HTTP — induceva.

Quei ~460 ms si decompongono ulteriormente (mediana di 7 avvii di
sottoprocesso):

| cosa viene importato | costo |
|---|---|
| interprete nudo (`pass`) | 11.8 ms |
| `pydantic` | 38.3 ms |
| `httpx` | 101.0 ms |
| `shared.operations` (cioe' httpx + il nostro codice) | 107.1 ms |
| **`mcp.types`** | **437.3 ms** |
| server MCP completo | 489.4 ms |

L'avvio di Python in se' costa 11.8 ms: e' trascurabile. Il costo sta
quasi tutto nell'import di `mcp.types`, cioe' nella **definizione delle
classi Pydantic dei tipi del protocollo**. Pydantic v2 compila i
validatori al momento della definizione della classe, e il sistema di
tipi di MCP e' ampio. `python -X importtime` lo conferma: 371 ms
cumulativi sull'albero `mcp`, di cui ~52 ms di tempo proprio in
`mcp_types._types`.

E' quindi una proprieta' **dell'SDK Python di MCP**, non del protocollo
ne' del linguaggio: un SDK con import pigri dei tipi, o un server scritto
in Go o TypeScript, non pagherebbe questa voce. Nella tesi la cifra va
etichettata cosi', altrimenti afferma su MCP qualcosa che riguarda una
sua implementazione.

**Non mantenere la sessione costa piu' del protocollo stesso.** Anche
scontato l'avvio del processo, ~8 ms contro ~1 ms significa che la
politica di gestione della sessione pesa circa otto volte il costo di
un'invocazione a sessione aperta. E' rilevante perche' e' il
comportamento predefinito di `MultiServerMCPClient`.
