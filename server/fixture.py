"""Dataset deterministico e leggibile per gli esperimenti.

Sostituisce il seeding basato su Faker del progetto d'esame. Faker con
seme fisso e' gia' riproducibile, ma produce titoli generati
automaticamente (``catch_phrase``) e localita' casuali: i task
risulterebbero illeggibili nella tesi e non ci sarebbe garanzia che
esista un evento in una data citta' o in un dato mese.

I dati qui sotto sono scelti in modo che ogni task abbia una risposta
attesa verificabile:

* un solo evento a Cagliari (id 1), per i task di disambiguazione;
* tre eventi in ottobre (id 3, 4, 5), per i task a ventaglio;
* ``mrossi`` iscritto esattamente ai tre eventi di ottobre;
* l'evento 1 ha esattamente tre iscritti.

Il modulo va importato dopo che ``server.wrapper`` ha inserito la radice
del progetto Event Manager in ``sys.path``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import Session

from app.models.event import Event
from app.models.registration import Registration
from app.models.user import User

EVENTS: tuple[dict[str, Any], ...] = (
    {
        "id": 1,
        "title": "Conferenza sull'Intelligenza Artificiale",
        "description": "Giornata di interventi su modelli linguistici e sistemi agentici.",
        "date": datetime(2026, 9, 15, 9, 0),
        "location": "Cagliari",
    },
    {
        "id": 2,
        "title": "Workshop di Cybersecurity",
        "description": "Laboratorio pratico su analisi delle vulnerabilita' e difesa.",
        "date": datetime(2026, 9, 28, 14, 0),
        "location": "Milano",
    },
    {
        "id": 3,
        "title": "Hackathon Open Source",
        "description": "Maratona di programmazione su progetti a sorgente aperto.",
        "date": datetime(2026, 10, 5, 10, 0),
        "location": "Torino",
    },
    {
        "id": 4,
        "title": "Seminario di Reti Neurali",
        "description": "Introduzione teorica e pratica alle architetture neurali.",
        "date": datetime(2026, 10, 12, 15, 0),
        "location": "Sassari",
    },
    {
        "id": 5,
        "title": "Corso di Cloud Computing",
        "description": "Percorso introduttivo su infrastrutture e servizi cloud.",
        "date": datetime(2026, 10, 20, 9, 30),
        "location": "Roma",
    },
    {
        "id": 6,
        "title": "Meetup Sviluppatori Python",
        "description": "Incontro informale della comunita' Python locale.",
        "date": datetime(2026, 11, 3, 18, 0),
        "location": "Bologna",
    },
    {
        "id": 7,
        "title": "Convegno su Robotica Industriale",
        "description": "Stato dell'arte dell'automazione nella manifattura.",
        "date": datetime(2026, 11, 18, 9, 0),
        "location": "Napoli",
    },
    {
        "id": 8,
        "title": "Forum sulla Trasformazione Digitale",
        "description": "Tavola rotonda su digitalizzazione dei processi aziendali.",
        "date": datetime(2026, 12, 2, 11, 0),
        "location": "Firenze",
    },
)

USERS: tuple[dict[str, str], ...] = (
    {"username": "mrossi", "name": "Marco Rossi", "email": "marco.rossi@example.it"},
    {"username": "lferrari", "name": "Laura Ferrari", "email": "laura.ferrari@example.it"},
    {"username": "gbianchi", "name": "Giulia Bianchi", "email": "giulia.bianchi@example.it"},
    {"username": "aconti", "name": "Andrea Conti", "email": "andrea.conti@example.it"},
    {"username": "svitale", "name": "Sara Vitale", "email": "sara.vitale@example.it"},
    {"username": "pgreco", "name": "Paolo Greco", "email": "paolo.greco@example.it"},
)

REGISTRATIONS: tuple[tuple[str, int], ...] = (
    ("mrossi", 3),
    ("mrossi", 4),
    ("mrossi", 5),
    ("lferrari", 1),
    ("lferrari", 3),
    ("gbianchi", 1),
    ("aconti", 2),
    ("aconti", 6),
    ("svitale", 1),
    ("svitale", 7),
    ("pgreco", 8),
)


def seed_fixture(session: Session) -> dict[str, int]:
    """Popola il database con il dataset del benchmark.

    Gli id degli eventi sono assegnati esplicitamente anziche' lasciati
    all'autoincremento, cosi' che i task possano riferirsi a un id fisso
    e la risposta attesa resti valida a ogni reset.
    """
    session.add_all(Event(**data) for data in EVENTS)
    session.add_all(User(**data) for data in USERS)
    session.add_all(
        Registration(username=username, event_id=event_id)
        for username, event_id in REGISTRATIONS
    )
    session.commit()
    return {
        "events": len(EVENTS),
        "users": len(USERS),
        "registrations": len(REGISTRATIONS),
    }
