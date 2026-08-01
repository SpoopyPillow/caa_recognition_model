from dataclasses import dataclass, field

EPS = 1e-10  # A very small value used for numerical stability
INF_PROXY = 10  # A value used to provide very large but finite bounds for mvn integration


@dataclass(frozen=True)
class CAAConfig:
    delta_t: float = 0.025  # Size of discrete time increment (sec)
    nr_ssteps: int = 8192  # Number of steps along the spatial axis
    nr_samples: int = 10000  # Number of trials to use for MC likelihood computation

    max_t: float = 8.0  # The ceiling of the time axis
    quantiles: list = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    nr_tsteps: int = field(init=False)

    def __post_init__(self):
        """Runs automatically upon creation."""
        nr_tsteps = int(self.max_t / self.delta_t)
        object.__setattr__(self, "nr_tsteps", nr_tsteps)


CAA_CFG = CAAConfig()
