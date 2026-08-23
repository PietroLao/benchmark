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

I compiti sono cinque, come concordato: la valutazione è deliberatamente
piccola, e serve a osservare *come* i due bracci lavorano, non a coprire
un dominio.

La risposta attesa di ciascun compito è ricavabile dai dati di prova
(``server/fixture.py``) ed è unica per costruzione: un solo evento a
Cagliari, esattamente tre eventi in ottobre, ``mrossi`` iscritto
esattamente a quei tre, tre iscritti all'evento 1.
"""

from __future__ import annotations

import re
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
MAX_ITERATIONS = 8


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


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
            "gli altri."
        ),
        expected="8",
        must_contain=(("8", "otto"),),
        min_tool_calls=1,
    ),
    Task(
        task_id="t2_disambiguazione",
        prompt="Qual e' il titolo dell'evento che si tiene a Cagliari?",
        rationale=(
            "Richiede di filtrare la lista per un attributo. Lo strumento "
            "non accetta filtri, quindi il modello deve recuperare tutti gli "
            "eventi ed esaminarli: mette alla prova la descrizione dello "
            "strumento, che e' esattamente cio' che cambia fra schema "
            "esplicito e schema dedotto."
        ),
        expected="Conferenza sull'Intelligenza Artificiale (evento 1)",
        must_contain=(("Intelligenza Artificiale",),),
        min_tool_calls=1,
    ),
    Task(
        task_id="t3_filtro_temporale",
        prompt="Quali eventi si tengono nel mese di ottobre? Elencane i titoli.",
        rationale=(
            "Filtro su una data anziche' su una stringa, con tre risultati "
            "attesi invece di uno: verifica che il modello non si fermi al "
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
        task_id="t4_join",
        prompt="A quanti eventi e' iscritto l'utente con username mrossi?",
        rationale=(
            "Richiede di mettere in relazione due entita' distinte, "
            "iscrizioni e utenti, che nessuno strumento restituisce gia' "
            "collegate."
        ),
        expected="3",
        must_contain=(("3", "tre"),),
        min_tool_calls=1,
    ),
    Task(
        task_id="t5_catena_scrittura",
        prompt=(
            "Iscrivi Paolo Greco, username pgreco, email paolo.greco@example.it, "
            "all'evento che si tiene a Cagliari."
        ),
        rationale=(
            "Unico compito che modifica lo stato, e unico a richiedere due "
            "invocazioni in sequenza dipendente: l'identificativo da passare "
            "alla seconda si conosce solo dopo la prima. E' il caso in cui un "
            "ciclo agentico serve davvero."
        ),
        expected="iscrizione di pgreco all'evento 1",
        must_contain=(("iscri",),),
        min_tool_calls=2,
        mutates_state=True,
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
    if task.task_id == "t5_catena_scrittura":
        return any(
            r.get("username") == "pgreco" and r.get("event_id") == 1
            for r in registrations
        )
    return None
