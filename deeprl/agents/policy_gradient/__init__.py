"""
Agents policy gradient (gradient de politique).

Ces agents optimisent directement la politique pi_theta(a|s) par gradient
ascendant sur le retour esperé, sans passer par une fonction de valeur Q.

Progression pedagogique :
1. REINFORCE                    - gradient Monte-Carlo pur, haute variance
2. REINFORCEWithMeanBaseline    - soustrait la moyenne pour reduire la variance
3. REINFORCEWithCriticBaseline  - critique apprend V(s), avantage A = G - V(s)
4. PPO                          - clipping du ratio pour des MAJ stables

References :
- Sutton & Barto (2018), Ch. 13 — Policy Gradient Methods
- Williams (1992) "Simple Statistical Gradient-Following Algorithms" (REINFORCE)
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms" (PPO)
"""

from deeprl.agents.policy_gradient.reinforce import (
    REINFORCE,
    REINFORCEWithMeanBaseline,
    REINFORCEWithCriticBaseline,
    PPO,
)

__all__ = [
    "REINFORCE",
    "REINFORCEWithMeanBaseline",
    "REINFORCEWithCriticBaseline",
    "PPO",
]
