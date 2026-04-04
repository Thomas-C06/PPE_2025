"""Gestionnaire d'alertes pour GeoQuant AI.

Detecte les changements de signal (LONG -> CASH ou CASH -> LONG)
et envoie une notification par email via SMTP.

Le dernier signal connu est persiste dans data/processed/last_signal.json
pour survivre aux redemarrages du dashboard.

Utilisation (depuis le dashboard) :
    from alert_manager import AlertManager

    mgr = AlertManager(base_dir=RACINE)
    if mgr.signal_changed(ticker="^GSPC", new_signal=1):
        mgr.send_email(
            smtp_server="smtp.gmail.com",
            smtp_port=465,
            email_from="ton@email.com",
            email_to="destinataire@email.com",
            password="mot_de_passe_app",
            subject="GeoQuant : signal LONG sur ^GSPC",
            body="...",
        )
        mgr.save_signal(ticker="^GSPC", signal=1)
"""

from __future__ import annotations

import json
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional


LAST_SIGNAL_FILE = "last_signal.json"


class AlertManager:
    """
    Gere la detection de changement de signal et l'envoi d'alertes email.

    Attributes:
        base_dir: Racine du projet GeoQuant AI.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir  = base_dir
        self._path     = base_dir / "data" / "processed" / LAST_SIGNAL_FILE

    # ── Persistance du dernier signal ─────────────────────────────────────

    def load_last_signals(self) -> dict:
        """Charge le dictionnaire {ticker: signal} depuis le fichier JSON."""
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_signal(self, ticker: str, signal: int) -> None:
        """Sauvegarde le signal courant pour un ticker."""
        data = self.load_last_signals()
        data[ticker] = signal
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def signal_changed(self, ticker: str, new_signal: int) -> bool:
        """Retourne True si le signal a change depuis la derniere sauvegarde."""
        last = self.load_last_signals()
        return last.get(ticker) != new_signal

    # ── Envoi email ───────────────────────────────────────────────────────

    @staticmethod
    def send_email(
        smtp_server: str,
        smtp_port:   int,
        email_from:  str,
        email_to:    str,
        password:    str,
        subject:     str,
        body:        str,
    ) -> tuple[bool, str]:
        """
        Envoie un email via SMTP SSL.

        Supporte Gmail (smtp.gmail.com:465) et Outlook (smtp-mail.outlook.com:587).

        Returns:
            (succes: bool, message_erreur: str)
        """
        try:
            msg              = MIMEText(body, "plain", "utf-8")
            msg["Subject"]   = subject
            msg["From"]      = email_from
            msg["To"]        = email_to

            if smtp_port == 587:
                with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
                    server.starttls()
                    server.login(email_from, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15) as server:
                    server.login(email_from, password)
                    server.send_message(msg)

            return True, ""
        except Exception as exc:
            return False, str(exc)

    # ── Corps d'email pre-formate ─────────────────────────────────────────

    @staticmethod
    def build_signal_body(
        ticker:      str,
        signal:      int,
        prix:        float,
        geo_score:   float,
        ma50:        float,
        ma200:       float,
        golden_cross: bool,
        confidence:  float,
        timestamp:   str,
    ) -> tuple[str, str]:
        """
        Construit le sujet et le corps d'un email d'alerte de signal.

        Returns:
            (subject, body)
        """
        signal_label = "LONG (investir)" if signal == 1 else "CASH (sortie du marche)"
        emoji        = "🟢" if signal == 1 else "🔴"

        subject = f"GeoQuant AI {emoji} Signal {signal_label} sur {ticker}"

        body = f"""GeoQuant AI -- Alerte de signal
========================================
Timestamp    : {timestamp}
Ticker       : {ticker}
Signal       : {signal_label}
Confiance    : {confidence:.0%}

-- Marche --
Prix actuel  : {prix:,.2f}
MA 50        : {ma50:,.2f}
MA 200       : {ma200:,.2f}
Golden Cross : {"Oui" if golden_cross else "Non"}

-- Sentiment --
Geo-Score    : {geo_score:+.3f}

========================================
Ce message est genere automatiquement par GeoQuant AI.
Il ne constitue pas un conseil en investissement.
"""
        return subject, body
