"""Caricamento della configurazione locale da file ``.env``.

Importare questo modulo carica il file ``.env`` posto nella radice del
banco di prova, se esiste, senza sovrascrivere le variabili gia' presenti
nell'ambiente: una variabile esportata a mano nel terminale ha percio'
sempre la precedenza sul file, che serve da comodita' e non da vincolo.

Il file contiene la chiave API e **non deve mai entrare nel
repository**: e' escluso dal controllo di versione tramite
``.gitignore``. Il modello da copiare e' ``.env.example``, che invece e'
versionato perche' contiene solo i nomi delle variabili.

Il percorso e' calcolato a partire da questo file e non dalla directory
corrente, cosi' che gli script funzionino da qualunque punto li si
lanci.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

#: ``override=False``: chi esporta la variabile nel terminale vince sul
#: file. E' l'ordine che serve quando si vuole eseguire una campagna con
#: una chiave diversa da quella abituale senza modificare il file.
_loaded = load_dotenv(ENV_PATH, override=False)


def env_file_found() -> bool:
    """Indica se un file ``.env`` e' stato effettivamente trovato e letto."""
    return bool(_loaded)
