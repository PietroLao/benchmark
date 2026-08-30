"""Prompt di sistema e compiti, definiti una sola volta per entrambi i bracci.

Secondo punto di verità unica del banco di prova, accanto a
``operations`` — che definisce cosa lo strumento fa, e da cui ciascun
braccio deriva lo schema con la propria API: qui vive **cosa viene
chiesto al modello**.

Il prompt di sistema è condiviso deliberatamente. Nulla in MCP impone un
certo prompt e nulla in LangChain lo vieta: è una scelta dello
sviluppatore, non una proprietà dei due approcci. Lasciarlo diverso fra i
bracci introdurrebbe nei conteggi di iterazioni una differenza arbitraria,
indistinguibile da un effetto del protocollo. Il framework, di suo, non ne
aggiunge alcuno — verificato: senza ``system_prompt`` esplicito
``create_agent`` invia al modello il solo messaggio dell'utente.

I compiti sono sette: la valutazione è deliberatamente piccola, e serve a
osservare *come* i due bracci lavorano, non a coprire un dominio.

Sono scelti in modo che il **numero di invocazioni non sia determinato
dal testo della richiesta**. È il criterio che l'insieme precedente non
soddisfaceva: ``list_events`` restituisce ogni campo di ogni evento e non
accetta filtri, quindi contare gli eventi, trovare quello di Cagliari ed
elencare quelli di ottobre si risolvevano tutti e tre con una sola
lettura seguita dalla risposta. Erano lo stesso compito con tre domande
diverse, e su di essi il conteggio delle iterazioni misurava una
proprietà del compito anziché dell'approccio. Un solo compito di quella
forma è rimasto, come riferimento dichiarato.

L'escursione viene da tre fonti distinte: un numero di operazioni che si
scopre solo eseguendo (``t5``), una differenza insiemistica che il modello
deve calcolare prima di sapere quante operazioni servano (``t6``), e un
identificativo che non esiste finche' l'agente non lo produce (``t7``).

Nessun compito ripete la struttura di un altro, con una sola eccezione
voluta: ``t3`` e ``t4`` usano gli stessi strumenti nello stesso numero e
differiscono unicamente perche' nel secondo l'operazione fallisce. E' una
coppia minima, e serve perche' qualunque scarto fra i due sia
attribuibile all'errore e a nient'altro.

``t4`` chiede un'iscrizione gia' esistente. **Se** i due bracci sappiano
riprendersi da uno strumento che fallisce e' una differenza deterministica,
che si stabilisce senza modello e vive in ``harness/error_paths.py``; qui
si misura invece **come si comporta l'agente dopo l'errore**, e se il
diverso testo che i due gli consegnano cambi qualcosa.

La risposta attesa di ciascun compito è ricavabile dai dati di prova
(``server/fixture.py``) ed è unica per costruzione: un solo evento a
Cagliari, esattamente tre eventi in ottobre, ``svitale`` iscritta agli
eventi 1 e 7, ``mrossi`` ai tre eventi di ottobre, ``lferrari``
all'evento 1 e a uno solo dei tre di ottobre.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

#: Istruzione inviata al modello prima della richiesta dell'utente,
#: identica nei due bracci. È formulata in modo neutro rispetto al
#: meccanismo di integrazione: non nomina né MCP né alcun framework.
SYSTEM_PROMPT = (
    "Sei un assistente che opera su un sistema di gestione eventi "
    "attraverso gli strumenti a tua disposizione. "
    "Usa gli strumenti per ottenere o modificare i dati di cui hai bisogno: "
    "non inventare informazioni e non basarti su conoscenze pregresse, "
    "perché i dati del sistema sono gli unici validi. "
    "Quando hai raccolto quanto basta, rispondi in modo conciso e diretto."
)

#: Limite di interrogazioni al modello per singola esecuzione. Serve a
#: impedire che un ciclo non terminante blocchi la campagna; se viene
#: raggiunto, l'esecuzione va registrata come fallita anziché scartata in
#: silenzio.
#:
#: Il valore va tenuto ben sopra il minimo richiesto dal compito più
#: lungo, altrimenti smette di essere una rete di sicurezza e diventa
#: esso stesso la causa dei fallimenti. ``t5`` richiede almeno cinque
#: interrogazioni — una lettura, tre cancellazioni, la risposta — e le
#: cancellazioni non sono accorpabili, perché l'endpoint rifiuta le
#: risposte con più di una chiamata a strumento. Un margine di due o tre
#: giri non basta a distinguere un agente che sbaglia strada da uno a cui
#: è stata tolta la strada.
MAX_ITERATIONS = 14


def _normalize(text: str) -> str:
    """Riduce alla forma su cui si confronta.

    Oltre a minuscole e spazi, appiattisce le vocali accentate e la
    convenzione dattilografica che le rende con l'apostrofo. Il modello
    scrive ora ``già``, ora ``gia'``, ora ``gia``, e la differenza non ha
    alcun significato: senza questo, la verifica boccerebbe risposte
    corrette per il modo in cui e' stato reso un accento.

    L'apostrofo si toglie **solo dopo vocale**, che in italiano e'
    praticamente sempre l'accento reso a macchina. Dopo consonante
    l'apostrofo e' un'elisione vera (``dell'evento``, ``l'iscrizione``) e
    va conservato, altrimenti si salderebbero due parole distinte.
    """
    piatto = unicodedata.normalize("NFKD", text.lower())
    piatto = "".join(c for c in piatto if not unicodedata.combining(c))
    piatto = re.sub(r"(?<=[aeiou])'", "", piatto)
    return re.sub(r"\s+", " ", piatto).strip()


@dataclass(frozen=True)
class Task:
    """Un compito, con la sua risposta attesa e il criterio di verifica."""

    task_id: str
    prompt: str
    #: Perché il compito esiste: quale aspetto del ciclo agentico mette
    #: alla prova. Serve a giustificarne la scelta nella tesi.
    rationale: str
    #: Risposta corretta in forma leggibile, per la discussione.
    expected: str
    #: Frammenti che devono comparire tutti nella risposta finale. La
    #: verifica è deliberatamente meccanica: affidarla a un secondo
    #: modello introdurrebbe un giudice non validato, che è una delle
    #: debolezze riscontrate nei lavori correlati.
    must_contain: tuple[tuple[str, ...], ...] = ()
    #: Numero minimo di invocazioni di strumento se il compito è svolto
    #: per la via più breve. Non è un obiettivo: serve a riconoscere le
    #: esecuzioni che hanno preso strade più lunghe.
    min_tool_calls: int = 1
    #: Se vero, il compito modifica lo stato: richiede un ripristino dei
    #: dati prima di ogni esecuzione.
    mutates_state: bool = False
    #: Iscrizioni che devono risultare presenti al termine, come coppie
    #: ``(username, event_id)``.
    state_present: tuple[tuple[str, int], ...] = ()
    #: Iscrizioni che non devono risultare presenti. Un ``event_id`` a
    #: ``None`` vale come "nessuna iscrizione di questo utente".
    state_absent: tuple[tuple[str, int | None], ...] = ()
    #: Numero totale di iscrizioni atteso al termine. Intercetta gli
    #: effetti collaterali: un agente che cancella l'iscrizione giusta e
    #: per errore anche un'altra supererebbe i due controlli precedenti.
    state_total: int | None = None

    def check(self, answer: str) -> bool:
        """Verifica la risposta finale.

        Ogni elemento di ``must_contain`` è un gruppo di alternative
        equivalenti: ne basta una per gruppo. Serve ad accettare "3" e
        "tre" senza allargare la verifica fino a renderla inutile.

        Il confronto richiede un confine di parola **iniziale**, non una
        semplice sottostringa: senza di esso la risposta "ci sono 18
        eventi" verrebbe accettata per il compito la cui risposta e' 8,
        e "13 eventi" per quello la cui risposta e' 3. Il confine finale
        e' invece deliberatamente assente, cosi' che un frammento come
        "iscri" continui a corrispondere a "iscritto".
        """
        norm = _normalize(answer)
        return all(
            any(
                re.search(r"\b" + re.escape(_normalize(v)), norm)
                for v in alternatives
            )
            for alternatives in self.must_contain
        )


TASKS: tuple[Task, ...] = (
    Task(
        task_id="t1_conteggio",
        prompt="Quanti eventi sono presenti nel sistema?",
        rationale=(
            "Compito minimo: una sola lettura e nessuna disambiguazione. "
            "Stabilisce il costo di base del ciclo, contro cui si leggono "
            "gli altri. E' l'unico in cui la lunghezza della soluzione e' "
            "fissata dal compito, ed e' tenuto proprio per questo: serve da "
            "riferimento, non da misura."
        ),
        expected="8",
        must_contain=(("8", "otto"),),
        min_tool_calls=1,
    ),
    Task(
        task_id="t2_catena_lettura",
        prompt=(
            "Chi e' iscritto all'evento che si tiene a Cagliari? Riporta "
            "nome ed email di ciascuno."
        ),
        rationale=(
            "Tre letture obbligate su **tre strumenti diversi**, senza "
            "scorciatoie: ``list_events`` per sapere quale evento sia quello "
            "di Cagliari, ``list_registrations`` per sapere chi vi e' "
            "iscritto, ``list_users`` per nome ed email. Nessuno strumento "
            "restituisce due di questi tre livelli insieme. E' il compito "
            "con il profilo piu' lungo fra quelli di sola lettura, ed e' "
            "l'unico che eserciti ``list_users``."
            "\n\n"
            "Sostituisce un compito che chiedeva gli eventi di ottobre: "
            "quello si risolveva con una sola ``list_events``, cioe' con lo "
            "stesso strumento e lo stesso numero di iterazioni di ``t1``, e "
            "non aggiungeva nulla alla misura."
        ),
        expected="Laura Ferrari, Giulia Bianchi e Sara Vitale con le loro email",
        must_contain=(
            ("ferrari",),
            ("bianchi",),
            ("vitale",),
            ("@example.it",),
        ),
        min_tool_calls=3,
    ),
    Task(
        task_id="t3_catena_scrittura",
        prompt=(
            "Iscrivi Paolo Greco, username pgreco, email paolo.greco@example.it, "
            "all'evento che si tiene a Cagliari."
        ),
        rationale=(
            "Due invocazioni in sequenza dipendente: l'identificativo da "
            "passare alla seconda si conosce solo dopo la prima. E' il caso "
            "elementare in cui un ciclo agentico serve davvero."
        ),
        expected="iscrizione di pgreco all'evento 1",
        must_contain=(("iscri",),),
        min_tool_calls=2,
        mutates_state=True,
        state_present=(("pgreco", 1),),
        state_total=12,
    ),
    Task(
        task_id="t4_conflitto",
        prompt=(
            "Iscrivi Laura Ferrari, username lferrari, "
            "email laura.ferrari@example.it, all'Hackathon Open Source."
        ),
        rationale=(
            "L'iscrizione richiesta esiste gia', quindi lo strumento "
            "fallisce e il compito si risolve riconoscendo il fallimento e "
            "riferendolo, non riprovando. Entrambi i bracci sono in grado "
            "di riprendersi — il braccio LangChain dopo l'adattamento "
            "documentato in ``harness/error_paths.py`` — quindi qui non si "
            "misura *se* si riprendano, ma **come si comporta l'agente dopo "
            "un errore**: se riferisca, se ritenti, se tenti una strada "
            "diversa."
            "\n\n"
            "Resta una differenza in cio' che il modello legge, e non e' "
            "stata pareggiata perche' appartiene ai due ecosistemi: l'SDK "
            "MCP antepone ``Error executing tool <nome>:`` al messaggio, "
            "LangChain lo consegna nudo. Se quel prefisso cambi il "
            "comportamento e' osservabile solo con il modello nel ciclo, ed "
            "e' la ragione per cui questo compito sta in campagna e non fra "
            "i test deterministici."
        ),
        expected="fallimento riferito: lferrari e' gia' iscritta all'evento 3",
        must_contain=(
            ("lferrari", "laura"),
            # Le varianti con accento o apostrofo non servono: la
            # normalizzazione le riporta tutte a questa forma.
            (
                "gia iscritt",
                "gia registrat",
                "gia present",
                "esiste gia",
                "duplicat",
                "non e stato possibile",
                "fallit",
                "errore",
            ),
        ),
        min_tool_calls=2,
        # Nulla deve cambiare. Un agente che cancellasse l'iscrizione per
        # poi ricrearla lascerebbe comunque lo stato in questa forma, ed e'
        # una soluzione legittima: il controllo verifica l'esito, non la
        # strada.
        mutates_state=True,
        state_present=(("lferrari", 3),),
        state_total=11,
    ),
    Task(
        task_id="t5_cancellazione_multipla",
        prompt="Cancella tutte le iscrizioni dell'utente mrossi.",
        rationale=(
            "Il numero di invocazioni non e' deducibile dal testo della "
            "richiesta: si scopre eseguendo la prima. ``mrossi`` risulta "
            "iscritto a tre eventi, quindi la via piu' breve e' una lettura "
            "e tre cancellazioni, ma un agente puo' legittimamente "
            "rileggere per verificare. E' il compito con la maggiore "
            "escursione sul numero di iterazioni, ed e' l'unico che eserciti "
            "``delete_registration``."
        ),
        expected="le tre iscrizioni di mrossi (eventi 3, 4, 5) rimosse",
        must_contain=(("cancellat", "eliminat", "rimoss"),),
        min_tool_calls=4,
        mutates_state=True,
        state_absent=(("mrossi", None),),
        state_total=8,
    ),
    Task(
        task_id="t6_iscrizione_condizionale",
        prompt=(
            "Iscrivi lferrari a tutti gli eventi di ottobre a cui non e' "
            "gia' iscritta."
        ),
        rationale=(
            "Il compito piu' esigente dell'insieme, e l'unico che unisca "
            "ragionamento e scrittura. Richiede due letture di natura "
            "diversa — quali eventi cadono in ottobre, a quali e' gia' "
            "iscritta — e poi la loro differenza insiemistica, che il "
            "modello deve calcolare da se': nessuno strumento la "
            "restituisce. Quante iscrizioni servano non e' deducibile dal "
            "testo, e ``lferrari`` e' scelta apposta perche' e' gia' "
            "iscritta a uno dei tre eventi di ottobre: con qualunque altro "
            "utente la risposta sarebbe 'tutti e tre' e l'esclusione non "
            "verrebbe mai esercitata."
            "\n\n"
            "In analisi va controllato se il modello tenti l'iscrizione "
            "all'evento 3. Sarebbe un errore suo di ragionamento, non del "
            "compito, ma i due bracci vi reagiscono in modo opposto — si "
            "veda ``harness/error_paths.py`` — e questo puo' toccare i "
            "conteggi."
        ),
        expected="lferrari iscritta agli eventi 4 e 5",
        must_contain=(("iscritt", "aggiunt", "registrat"),),
        min_tool_calls=4,
        mutates_state=True,
        state_present=(("lferrari", 4), ("lferrari", 5)),
        state_total=13,
    ),
    Task(
        task_id="t7_creazione",
        prompt=(
            "Crea un evento intitolato 'Seminario di Basi di Dati', che si "
            "tiene a Quartu Sant'Elena il 15 giugno 2026 alle 18:00, con "
            "descrizione 'Introduzione ai database relazionali'. Poi "
            "iscrivici Andrea Conti, username aconti, email "
            "andrea.conti@example.it."
        ),
        rationale=(
            "Due cose che nessun altro compito richiede. L'identificativo da "
            "passare alla seconda chiamata non viene **letto** ma "
            "**prodotto**: e' l'evento appena creato, e non esiste prima che "
            "l'agente lo crei — in ``t3`` l'identificativo si scopre "
            "interrogando, qui si ottiene da una scrittura. E gli argomenti "
            "non sono ne' copiati dalla richiesta ne' letti da un risultato: "
            "la data va **sintetizzata** nel formato che lo strumento "
            "dichiara, partendo da 'il 15 giugno 2026 alle 18:00'."
            "\n\n"
            "E' quindi anche l'unico compito in cui la qualita' della "
            "descrizione dello schema incide direttamente sull'esito, il che "
            "lo collega alla misura di ``harness/schema_gate.py``. Unico a "
            "esercitare ``create_event``."
        ),
        expected="evento 9 creato, aconti iscritto",
        must_contain=(("creat",), ("iscritt", "registrat", "aggiunt")),
        min_tool_calls=2,
        mutates_state=True,
        state_present=(("aconti", 9),),
        state_total=12,
    ),
)

TASKS_BY_ID: dict[str, Task] = {t.task_id: t for t in TASKS}


def verify_state(task: Task, registrations: list[dict[str, Any]]) -> bool | None:
    """Verifica l'effetto sullo stato per i compiti che lo modificano.

    Per i compiti di sola lettura restituisce ``None``: non c'e' nulla da
    controllare. Il controllo sullo stato e' piu' affidabile di quello
    sul testo, perche' non dipende da come il modello ha formulato la
    risposta.
    """
    if not task.mutates_state:
        return None

    presenti = {(r.get("username"), r.get("event_id")) for r in registrations}

    if task.state_total is not None and len(registrations) != task.state_total:
        return False
    if any(coppia not in presenti for coppia in task.state_present):
        return False
    for username, event_id in task.state_absent:
        if event_id is None:
            if any(u == username for u, _ in presenti):
                return False
        elif (username, event_id) in presenti:
            return False
    return True
