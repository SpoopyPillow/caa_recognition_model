from dataclasses import dataclass

@dataclass
class ModelEvaluation:
    method: str
    n_trials: int
    n_params: int
    nll: float
    aic: float
    bic: float

    def _repr_markdown_(self) -> str:
        return f"""
### Model Evaluation Summary

| Metric | Value |
| :--- | :--- |
| **Method** | `{self.method.upper()}` |
| **Trials ($N$)** | {self.n_trials:,} |
| **Parameters ($k$)** | {self.n_params} |
| **Negative Log-Likelihood** | {self.nll:.2f} |
| **AIC** | **{self.aic:.2f}** |
| **BIC** | **{self.bic:.2f}** |
"""
