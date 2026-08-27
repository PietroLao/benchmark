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
3. **`shared/tasks.py`** definisce il prompt di sistema e i compiti, di
   nuovo una sola volta. Nulla in MCP impone un certo prompt e nulla in
   LangChain lo vieta: e' una scelta dello sviluppatore, non una
   proprieta' dei due approcci, e lasciarla divergere metterebbe nei
   conteggi di iterazioni una differenza arbitraria indistinguibile da un
   effetto del protocollo. Verificato: senza `system_prompt` esplicito
   `create_agent` invia al modello il solo messaggio dell'utente, quindi
   il framework non ne aggiunge di suo.
4. **Diff dei body HTTP** inviati a `/v1/chat/completions` nei due
   bracci: se `tools` e `messages` coincidono, il modello riceve lo
   stesso input.

`harness/schema_gate.py` verifica il campo `tools` e **passa**: gli
schemi vengono raccolti da un vero `tools/list` su un server MCP in
esecuzione, quindi dopo serializzazione JSON-RPC, trasmissione e
deserializzazione — non confrontando definizioni in memoria, che
sarebbero identiche per costruzione. Resta da verificare `messages`, che
dipende dai loop agentici.

`harness/messages_gate.py` verifica `messages`, confrontando due tracce
reali dello stesso compito. Il confronto non e' letterale: identificativi
di chiamata generati a caso, `content: null` contro campo assente e campi
in piu' nella ricostruzione LangChain sono differenze prive di
significato semantico, e vengono normalizzate. Resta come prova
indipendente il **conteggio dei token in ingresso** riportato
dall'endpoint, che non dipende dal nostro codice di registrazione: se
coincide, il modello ha ricevuto lo stesso input. Verificato su
`t1_conteggio`: 1173 e 1735 token, identici in entrambe le interrogazioni.

```bash
uv run python -m harness.schema_gate
uv run python -m harness.messages_gate --task t1_conteggio
```

## Stato

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | Smoke test tool calling su NIM | **superata** |
| 1 | Server strumentato, fixture, esperimento A | **completata** |
| 2 | Host MCP standalone con LLM | **implementata**, da eseguire |
| 3 | Braccio LangChain | **implementata**, da eseguire |
| 4 | Parita' dell'input (gate) | **verde** su `tools`; da riscrivere su `messages` |
| 5 | Caratterizzazione rumore `t_llm` | da fare |
| 6 | Campagna completa | **implementata**, da eseguire |
| 6b | Riepilogo leggibile della campagna | **implementata** |
| 7 | Analisi e tabelle LaTeX | da fare |

## Struttura

```
shared/tools_spec.py     definizione unica degli strumenti
shared/operations.py     unica implementazione delle chiamate REST
shared/tasks.py          prompt di sistema condiviso e cinque compiti
server/wrapper.py        avvolge l'Event Manager: conteggio REST + reset
server/fixture.py        dataset deterministico e leggibile
arm_mcp/server.py        server MCP (API di basso livello, schemi espliciti)
arm_mcp/http_server.py   lo stesso server su trasporto Streamable HTTP
arm_mcp/host.py          fase 2: host agentico autonomo (niente framework)
arm_langchain/tools.py   strumenti LangChain derivati da tools_spec
arm_langchain/agent.py   fase 3: agente ReAct + cattura della traccia
arm_langchain/wire.py    intercetta il payload HTTP realmente trasmesso
shared/nim.py            client per l'endpoint, usato dal solo host MCP
shared/env.py            caricamento di .env (chiave API)
harness/smoke_nim.py     fase 0: verifica del tool calling su NIM
harness/trace.py         registrazione integrale delle esecuzioni
harness/schema_gate.py   fase 4a: parita' degli schemi (tools)
harness/messages_gate.py fase 4b: parita' della conversazione (messages)
harness/campaign.py      fase 6: campagna, bracci alternati e riprendibile
harness/summary.py       fase 6b: riepilogo Markdown della campagna
microbench/transport.py  esperimento A: overhead di trasporto, senza LLM
results/                 un JSON per esecuzione, piu' il riepilogo
```

Le tracce servono a ricostruire un'esecuzione, non a leggerla: contengono i
messaggi e gli schemi integrali. Il riepilogo estrae da un'intera cartella
di tracce le sole grandezze su cui la tesi si pronuncia, con i due bracci
su righe adiacenti e un valore per ogni ripetizione. Viene generato in coda
alla campagna, ed e' rigenerabile in qualunque momento, anche su una
campagna interrotta a meta':

```bash
uv run python -m harness.summary --dir results/campagna
```

## Requisiti

Il progetto Event Manager non viene modificato: e' importato dall'esterno.
Percorso configurabile con `EVENT_MANAGER_ROOT` (default:
`~/Desktop/ProgettoProgrammazioneWeb2026`).

```bash
uv sync
cp .env.example .env      # poi inserire la propria chiave NVIDIA
```

La chiave sta in `.env`, che **non e' versionato**: `.gitignore` lo
esclude, e nel repository compare solo il modello `.env.example` con i
soli nomi delle variabili. Una variabile esportata a mano nel terminale
ha comunque la precedenza sul file, cosi' si puo' eseguire una campagna
con una chiave o un modello diversi senza modificarlo. La chiave non
compare mai nelle tracce salvate, perche' viaggia nelle intestazioni
HTTP, che non vengono registrate.

Il comando installa entrambi i bracci. LangChain non e' una dipendenza
opzionale: l'esperimento A misura una condizione LangChain e il gate
sugli schemi confronta i due bracci fra loro, quindi entrambi la
importano. Le versioni esatte sono fissate in `uv.lock`, che va tenuto
sotto controllo di versione: i risultati misurati dipendono da esse — i
437 ms di import dell'SDK MCP e la quota di introspezione nel percorso
LangChain sono proprieta' di versioni precise, e senza il lock non
sarebbero riproducibili.

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

Avviare il server MCP su HTTP. La variabile `BENCH_EXPOSE_ECHO` serve
alle condizioni HTTP dell'esperimento A e puo' restare impostata per
tutto: sia il gate sia l'host agentico scartano gli strumenti interni,
quindi il modello non li vede in nessun caso. Senza il server, le
condizioni HTTP del microbenchmark vengono saltate con un avviso invece
di far fallire l'esecuzione.

```bash
BENCH_EXPOSE_ECHO=1 uv run uvicorn arm_mcp.http_server:app --port 8100
```

Eseguire l'esperimento A:

```bash
uv run python -m microbench.transport --repetitions 100 --warmup 20
```

Eseguire la fase 0 (richiede la chiave, mai scritta su file):

```bash
uv run python -m harness.smoke_nim
```

Eseguire un compito con l'uno o l'altro braccio (senza `--task` li esegue
tutti):

```bash
uv run python -m arm_mcp.host --task t2_disambiguazione
uv run python -m arm_langchain.agent --task t2_disambiguazione
```

Eseguire la campagna. E' riprendibile: rilanciandola sulla stessa
cartella salta le esecuzioni gia' presenti.

```bash
uv run python -m harness.campaign --repetitions 3
```

## Note metodologiche

**Alternanza delle condizioni e riscaldamento.** Le misure girano su un
portatile senza ventola. Le condizioni sono alternate a rotazione anziche'
eseguite in blocco, cosi' che l'eventuale deriva termica colpisca tutte le
condizioni allo stesso modo; le prime ripetizioni sono scartate.

**Statistiche robuste.** Mediana e IQR, non media e deviazione standard:
la distribuzione delle latenze e' asimmetrica e con code lunghe.

**Ripetizioni, non iterazioni.** Nel microbenchmark il parametro
`--repetitions` indica quante volte la stessa misura viene ripetuta. Il
termine *iterazione* e' riservato al **numero di interrogazioni al
modello** entro una singola esecuzione agentica, che e' una delle metriche
della tesi: usare la stessa parola per due grandezze diverse ha gia'
generato confusione in sede di revisione.

**Tracce integrali.** Ogni esecuzione agentica scrive in `results/traces/`
un JSON con i payload inviati al modello, le risposte per intero, gli
argomenti e i risultati di ogni strumento, i tempi e i conteggi. Si
registra anche cio' che al momento non serve: interrogare il modello costa
minuti, quindi una metrica definita dopo l'esecuzione — token, caratteri
scambiati, ripetizioni di uno stesso strumento — deve essere ricalcolabile
leggendo i file invece di rieseguire. Il nome del file contiene braccio,
modello, compito, istante e identificativo: nessuna esecuzione puo'
sovrascriverne un'altra. Le intestazioni HTTP non sono mai registrate,
perche' conterrebbero la chiave API.

**Conteggio delle chiamate REST.** Il braccio MCP esegue le chiamate dal
processo del **server MCP**, non da quello dell'host: la variabile di
contesto che porta l'identificativo di esecuzione non lo raggiunge,
quindi l'intestazione `X-Run-Id` non viene apposta. Con il solo contatore
per identificativo il braccio MCP risultava avere **zero** chiamate REST,
in silenzio — una delle due metriche portanti azzerata. Far viaggiare
l'identificativo dentro il protocollo richiederebbe di aggiungere un
argomento agli strumenti, cioe' rompere la parita' degli schemi. Il
wrapper espone percio' anche un contatore incondizionato
(`/__bench__/global`), che la campagna azzera prima di ogni esecuzione e
legge dopo: essendo le esecuzioni seriali, il totale appartiene per
intero all'ultima.

**Il modello e' una configurazione volatile.** I modelli ospitati vengono
ritirati: `meta/llama-3.3-70b-instruct`, usato nella prima campagna, e'
stato dismesso il 26 agosto 2026 e da allora l'endpoint risponde 410. Ogni
risultato va percio' riportato indicando modello e data, e i dati raccolti
con un modello dismesso non sono piu' riproducibili. Il modello corrente e'
`openai/gpt-oss-120b`.

**Parita' dell'input: cosa e' esatto e cosa e' tollerato.** Il gate sui
messaggi confronta il **contenuto** in modo esatto, dopo aver escluso gli
identificativi delle chiamate a strumento, che l'endpoint genera a caso.
Il conteggio dei token e' invece un controllo secondario con tolleranza:
su contenuto identico carattere per carattere si e' osservata
un'oscillazione fra 1089 e 1122 token, perche' sequenze esadecimali
diverse si segmentano in modo diverso. La tolleranza e' calibrata su
quella variabilita' misurata.

**Campi non canonici nella conversazione.** L'host MCP riaccoda una forma
canonica del messaggio dell'assistente — solo `role`, `content` e
`tool_calls` — e non l'oggetto grezzo dell'API. Alcuni modelli vi
aggiungono campi propri: `gpt-oss` restituisce `reasoning_content` con la
catena di ragionamento, e riaccodarlo rimanderebbe al modello, a ogni
giro, testo estraneo alla conversazione, rompendo inoltre la parita' con
il braccio LangChain, che normalizza allo stesso modo.

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
corretta.** Con un ordine fisso, la prima condizione di ogni ripetizione
veniva eseguita sempre subito dopo quelle che avviano processi e chiudono
connessioni, pagandone ogni volta gli strascichi. Il sintomo era che
`langchain_tool` risultava *piu' veloce* della chiamata diretta su
`list_events` (-0.108 ms), il che e' impossibile. Isolando le due
condizioni il segno si inverte e l'overhead diventa +0.141 ms, coerente
con i +0.148 ms misurati su `echo`. Le condizioni sono ora permutate a
caso a ogni ripetizione, con seme fisso per riproducibilita'.

E' il motivo per cui la tabella qui sotto va letta come misura valida: i
valori assoluti restano piu' alti che in isolamento, perche' le
condizioni pesanti caricano comunque il processo, ma la distorsione non
cade piu' sempre sulla stessa cella.

## Cosa mostra l'esperimento A

Tre esecuzioni con ordine permutato: una da 40 ripetizioni e **due
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
