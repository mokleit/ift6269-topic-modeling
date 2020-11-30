from dataclasses import dataclass


@dataclass
class Document:
    words: []
    counts: []
    length: int
    total: int


@dataclass
class LdaModel:
    alpha: float
    log_prob_w: []
    num_topics: int
    num_terms: int


@dataclass
class Settings:
    VAR_MAX_ITER: int
    VAR_CONVERGED: float
    EM_MAX_ITER: int
    ESTIMATE_ALPHA: int



