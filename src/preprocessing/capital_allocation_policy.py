"""Pure conversion from a market regime to an equity-exposure recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class AllocationConfig:
    version: str = "capital-v1"
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "Konservatif": (0.70, 0.40),
        "Moderat": (0.80, 0.50),
        "Agresif": (0.90, 0.60),
    })


CONFIG = AllocationConfig()


def recommend_fraction(status: str, severity: float | None, risk_profile: str,
                       config: AllocationConfig = CONFIG) -> float | None:
    if risk_profile not in config.bounds:
        raise ValueError(f"Profil risiko tidak dikenal: {risk_profile}")
    if status == "unknown":
        return None
    if status == "normal":
        return 1.0
    if status != "bear":
        raise ValueError(f"Status pasar tidak dikenal: {status}")
    if severity is None or not math.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("Severity bear harus berada di 0..1.")
    base, minimum = config.bounds[risk_profile]
    if not 0 < minimum <= base <= 1:
        raise ValueError("Batas alokasi tidak valid.")
    return round(base - severity * (base - minimum), 6)


def effective_fraction(assessment: dict, risk_profile: str, choice: str) -> float:
    """Resolve explicit user choice; unknown never silently becomes full."""
    if choice == "full":
        return 1.0
    if choice != "recommended":
        raise ValueError("Pilihan modal harus 'recommended' atau 'full'.")
    recommendation = recommend_fraction(
        assessment["status"], assessment["severity"], risk_profile
    )
    if recommendation is None:
        raise ValueError("Kondisi pasar unknown; pilih full modal secara eksplisit atau batalkan.")
    return recommendation
