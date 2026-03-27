"""
Agents d'imitation (Imitation Learning).

Ces agents apprennent à partir de démonstrations d'experts.
"""

from deeprl.agents.imitation.expert_apprentice import (
    ExpertApprenticeAgent,
    ExpertPolicy,
    MCTSExpert,
    BehaviorCloning,
    DAgger
)

__all__ = [
    "ExpertApprenticeAgent",
    "ExpertPolicy",
    "MCTSExpert",
    "BehaviorCloning",
    "DAgger",
]
