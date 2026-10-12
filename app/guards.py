"""Hard-boundary pre-check — surgical, unambiguous hard-boundary intent only.

Ported from error-inbox PR #209 (lib/bro/harness.ts). Design: multi-signal
matchers (ACTION + CRIMINAL OBJECT). Bare keywords NEVER match — "memory
dump", "checkout validation", "audio mixing", "mock login", "cold storage
move", "bath bomb", "nuke the test db" all pass. This is a backstop only;
the model (Venice) handles its own refusals. Each guard carries a plain
1-line reason + the nearest legitimate alternative, no lecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryHit:
    blocked: bool
    reason: str
    alternative: str


@dataclass(frozen=True)
class Guard:
    id: str
    match: object  # Callable[[str], bool]
    reason: str
    alternative: str


_ACTIONS = {
    "acquire": re.compile(r"\b(buy|purchase|sell|get|find|where can i|need|looking for|source)\b"),
    "useCard": re.compile(r"\b(validate|validating|validation|check|checking|use|using|run|test|testing)\b"),
    "build": re.compile(r"\b(build|make|making|create|creating|assemble|write|deploy|set up|set-up|give me|send me|craft)\b"),
    "steal": re.compile(r"\b(steal|stealing|drain|draining|empty|emptying|siphon|rug|rob)\b"),
    "attack": re.compile(r"\b(hack|hacking|break into|breaking into|crack|compromise)\b"),
}

_CRIMINAL_TARGET = re.compile(r"\b(someone|somebody|else's|their|his|her|victim|stranger|other people)\b")

# Explicit authorized-security-research context — these always pass.
_AUTHORIZED_CTX = re.compile(
    r"\b(my own|mine|authorized|authorization|permission|consent|pentest|pen test|ctf\b|"
    r"bounty|hackerone|bugcrowd|red team|redteam|lab|test (account|server|machine|environment|database|db)|"
    r"security awareness|training)\b"
)


def _stolen_cards(t: str) -> bool:
    if re.search(r"\bfullz\b|\bdumps?\s*\+?\s*pin\b|carding\s+(forum|shop|market|marketplace)\b", t):
        return True
    return bool(
        re.search(r"stolen|carding|fraudulent", t)
        and _ACTIONS["useCard"].search(t)
        and re.search(r"\b(cc|cvvs?|card|cards|credit card|debit card)\b", t)
    )


def _fraud_phishing(t: str) -> bool:
    if re.search(r"phishing\s+(kit|kits|page|pages|site|sites|landing page)\b", t):
        return True
    if re.search(r"credential\s+(harvest|harvesting|steal|stealing|theft)", t):
        return True
    return bool(
        re.search(r"fake\s+(login|bank|invoice|payment|check|id|passport|document)", t)
        and re.search(r"\b(trick|scam|fool|deceive|deception|victim|steal)\b", t)
    )


def _theft_wallets(t: str) -> bool:
    if (
        _ACTIONS["steal"].search(t)
        and _CRIMINAL_TARGET.search(t)
        and not _AUTHORIZED_CTX.search(t)
        and re.search(r"\b(wallet|crypto|btc|bitcoin|eth|funds|money|account|balance|keys?)\b", t)
    ):
        return True
    return bool(
        re.search(r"\b(mix|mixer|tumbler|tumbling)\b", t)
        and re.search(r"\b(crypto|btc|bitcoin|eth|coins?|funds)\b", t)
        and re.search(r"\b(mix|hide|obscure|launder|clean|anonymize)\b", t)
    )


def _unauthorized_hacking(t: str) -> bool:
    return bool(
        not _AUTHORIZED_CTX.search(t)
        and _ACTIONS["attack"].search(t)
        and _CRIMINAL_TARGET.search(t)
        and re.search(r"\b(account|email|phone|computer|server|network|system|password|inbox)\b", t)
    )


def _weapons_mass_harm(t: str) -> bool:
    # "bath bomb", "bomb calorimeter", and "nuke the test db" are NOT here.
    return bool(
        _ACTIONS["build"].search(t)
        and re.search(
            r"(?<!bath\s)\bbomb\b(?!\s*calorimeter)|pipe bomb|bioweapon|bio-weapon|"
            r"chemical weapon|dirty bomb|molotov|nerve agent|sarin|anthrax|wmd\b",
            t,
        )
    )


def _csam(t: str) -> bool:
    if re.search(r"\bcsam\b", t):
        return True
    return bool(
        re.search(r"\b(child|children|minor|minors|kid|kids|teen|underage)\b", t)
        and re.search(r"\b(sexual|porn|pornography|nude|explicit|erotic)\b", t)
    )


def _doxxing(t: str) -> bool:
    if re.search(r"\bdox(x)?(ing|ed|es)?\b", t):
        return True
    # Public business contact info ("business phone number") passes.
    return bool(
        re.search(r"\b(find|get|give|locate|lookup|look up|pull|dig up)\b", t)
        and re.search(
            r"\b(home address|personal (phone|cell|mobile)|cell (phone )?number|social security|ssn\b)", t
        )
    )


GUARDS: list[Guard] = [
    Guard(
        id="stolen-cards",
        match=_stolen_cards,
        reason="Can't help with stolen-card stuff.",
        alternative="If it's your own card or a legit payment integration (your BTC checkout, a processor), tell me which and I'll build that instead.",
    ),
    Guard(
        id="fraud-phishing",
        match=_fraud_phishing,
        reason="Can't help with fraud — phishing kits or fake pages meant to trick people.",
        alternative="If you need a legit login or billing page for one of your businesses, I'll build that.",
    ),
    Guard(
        id="theft-wallets",
        match=_theft_wallets,
        reason="Can't help with theft — draining wallets, stealing funds, or laundering them.",
        alternative="If the wallet is yours and you lost access, I can walk you through legit recovery options instead.",
    ),
    Guard(
        id="unauthorized-hacking",
        match=_unauthorized_hacking,
        reason="Can't hack targets you didn't authorize — someone else's accounts or systems.",
        alternative="Your own repos/domains/apps, CTFs, and bounty programs you name are fair game — point me at one and I'll dig in.",
    ),
    Guard(
        id="weapons-mass-harm",
        match=_weapons_mass_harm,
        reason="Can't help with weapons of mass harm.",
        alternative="Anything else — legit chemistry, your safety certs, coursework — I'm in.",
    ),
    Guard(
        id="csam",
        match=_csam,
        reason="Absolute no on anything sexual involving minors.",
        alternative="No alternative here.",
    ),
    Guard(
        id="doxxing",
        match=_doxxing,
        reason="Can't pull private info (home addresses, personal numbers) on people.",
        alternative="Public business contact info is fine — give me the business name and I'll look that up.",
    ),
]


def check_hard_boundary(text: str) -> BoundaryHit | None:
    """Pre-check: returns a BoundaryHit only for unambiguous hard-boundary intent, else None."""
    t = f" {str(text or '').lower()} "
    for g in GUARDS:
        if g.match(t):  # type: ignore[operator]
            return BoundaryHit(blocked=True, reason=g.reason, alternative=g.alternative)
    return None
