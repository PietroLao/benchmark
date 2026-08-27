"""Prompt di sistema e compiti, definiti una sola volta per entrambi i bracci.

Terzo punto di verità unica del banco di prova, accanto a
``tools_spec`` (cosa il modello vede) e ``operations`` (cosa lo strumento
fa): qui vive **cosa viene chiesto al modello**.

Il prompt di sistema è condiviso deliberatamente. Nulla in MCP impone un
certo prompt e nulla in LangChain lo vieta: è una scelta dello
sviluppatore, non una proprietà dei due approcci. Lasciarlo diverso fra i
bracci introdurrebbe nei conteggi di iterazioni una differenza arbitraria,
indistinguibile da un effetto del protocollo. Il framework, di suo, non ne
aggiunge alcuno — verificato: senza ``system_prompt`` esplicito
``create_agent`` invia al modello il solo messaggio dell'utente.

I compiti sono sei: la valutazione è deliberatamente piccola, e serve a
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

L'escursione viene da tre fonti distinte: una soluzione ottenibile per
vie di lunghezza diversa (``t3``), un numero di operazioni che si scopre
solo eseguendo (``t5``), e un'operazione che fallisce e va riconosciuta
come tale (``t6``).

La risposta attesa di ciascun compito è ricavabile dai dati di prova
(``server/fixture.py``) ed è unica per costruzione: un solo evento a
Cagliari, esattamente tre eventi in ottobre, ``svitale`` iscritta agli
eventi 1 e 7, ``mrossi`` a tre eventi, ``lferrari`` già iscritta
all'evento 3.
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
        task_id="t2_filtro_temporale",
        prompt="Quali eventi si tengono nel mese di ottobre? Elencane i titoli.",
        rationale=(
            "Una sola lettura, ma tre risultati attesi invece di uno e un "
            "filtro su una data: verifica che il modello non si fermi al "
            "primo elemento utile."
        ),
        expected=(
            "Hackathon Open Source, Seminario di Reti Neurali, "
            "Corso di Cloud Computing"
        ),
        must_contain=(
            ("Hackathon",),
            ("Reti Neurali",),
            ("Cloud Computing",),
        ),
        min_tool_calls=1,
    ),
    Task(
        task_id="t3_join_titoli",
        prompt=(
            "Quali sono i titoli degli eventi a cui e' iscritta l'utente "
            "con username svitale?"
        ),
        rationale=(
            "Mette davvero in relazione due entita': ``list_registrations`` "
            "restituisce identificativi di evento, non titoli, e i titoli "
            "stanno solo fra gli eventi. La lunghezza della soluzione non e' "
            "fissata: i due titoli si possono ottenere con una sola "
            "``list_events`` oppure con due ``get_event``, e quale via il "
            "modello scelga e' una decisione sua. E' il primo compito in cui "
            "il numero di invocazioni ha escursione."
        ),
        expected=(
            "Conferenza sull'Intelligenza Artificiale, "
            "Convegno su Robotica Industriale"
        ),
        must_contain=(("Intelligenza Artificiale",), ("Robotica",)),
        min_tool_calls=2,
    ),
    Task(
        task_id="t4_catena_scrittura",
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
        task_id="t6_conflitto",
        prompt=(
            "Iscrivi Laura Ferrari, username lferrari, "
            "email laura.ferrari@example.it, all'Hackathon Open Source."
        ),
        rationale=(
            "L'iscrizione richiesta esiste gia', quindi lo strumento "
            "fallisce e il compito si risolve riconoscendo il fallimento e "
            "riferendolo, non riprovando. E' il solo compito in cui i due "
            "bracci differiscono per meccanismo e non per involucro: MCP "
            "segnala l'errore con ``isError`` sul risultato, mentre in "
            "LangChain ``ToolNode`` intercetta l'eccezione e la consegna al "
            "modello come contenuto di un messaggio di strumento. Il modello "
            "vede quindi due cose diverse, e come reagisce e' precisamente "
            "cio' che questo compito misura."
        ),
        expected="fallimento: lferrari e' gia' iscritta all'evento 3",
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
