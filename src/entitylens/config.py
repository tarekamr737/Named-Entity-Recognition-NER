"""Central project paths and runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Locations used for reproducible local runs."""

    root: Path = PROJECT_ROOT

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.artifacts / "metrics"

    @property
    def plots(self) -> Path:
        return self.artifacts / "plots"

    @property
    def screenshots(self) -> Path:
        return self.artifacts / "screenshots"

    @property
    def vocabularies(self) -> Path:
        return self.artifacts / "vocabularies"

    def ensure(self) -> None:
        """Create the writable project locations if they do not yet exist."""
        for path in (
            self.data_raw,
            self.data_processed,
            self.checkpoints,
            self.metrics,
            self.plots,
            self.screenshots,
            self.vocabularies,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainingConfig:
    """Shared defaults, intentionally used by every architecture."""

    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 8
    max_length: int = 128
    embedding_dim: int = 100
    hidden_dim: int = 256
    dropout: float = 0.2
