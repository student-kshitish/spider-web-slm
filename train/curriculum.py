import math
from dataclasses import dataclass
from config import get_config

cfg = get_config()


@dataclass
class Phase:
    name: str
    steps: int
    # dataset_label is a documentary label only. The dataloader is hardcoded to
    # data/raw/ and is NOT switched between phases. All phases train on the single
    # TinyStories file (data/raw/tinystories.txt).
    dataset_label: str
    hard_routing: bool
    entropy_weight: float
    description: str
    # aux_weight removed: loss_aux is hardcoded to 0.0 in loss.py and was never
    # applied to the total loss, so the weight had no effect on training.


# Entropy-annealing schedule over 30K steps on TinyStories.
# P4 is the only architecturally distinct phase: Gumbel-Softmax → hard STE routing.
CURRICULUM = [
    Phase(
        name="P1_Atomic_Logic",
        steps=3000,
        dataset_label="tinystories",
        hard_routing=False,
        entropy_weight=0.1,
        description="Soft routing, high entropy encouragement (exploration).",
    ),
    Phase(
        name="P2_Structural_Syntax",
        steps=6000,
        dataset_label="tinystories",
        hard_routing=False,
        entropy_weight=0.05,
        description="Soft routing, moderate entropy encouragement.",
    ),
    Phase(
        name="P3_Stabilization",
        steps=6000,
        dataset_label="tinystories",
        hard_routing=False,
        entropy_weight=0.02,
        description="Soft routing, low entropy encouragement.",
    ),
    Phase(
        name="P4_Logical_Scaling",
        steps=15000,
        dataset_label="tinystories",
        hard_routing=True,
        entropy_weight=0.01,
        description="Hard (straight-through) routing, minimal entropy term.",
    ),
]


class CurriculumManager:
    """
    Schedules entropy_weight annealing and the soft→hard routing transition.
    Dataset switching is not implemented; all phases use the same dataloader.
    """

    def __init__(self):
        self.curriculum = CURRICULUM

    def get_phase(self, step: int) -> Phase:
        cumulative = 0
        for phase in self.curriculum:
            cumulative += phase.steps
            if step < cumulative:
                return phase
        return self.curriculum[-1]

    def apply_to_model(self, step: int) -> dict:
        phase = self.get_phase(step)
        cfg.train.entropy_weight = phase.entropy_weight
        return {
            "name":           phase.name,
            "hard_routing":   phase.hard_routing,
            "entropy_weight": phase.entropy_weight,
        }
