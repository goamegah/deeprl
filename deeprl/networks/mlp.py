"""
Multi-Layer Perceptron (MLP) - Réseau de neurones simple.

Le MLP est l'architecture de base pour les algorithmes Deep RL.
Il prend un état en entrée et produit:
- Des valeurs Q pour chaque action (DQN)
- Une distribution de probabilités sur les actions (Policy Gradient)
- Une valeur d'état (Critic)

Architecture typique:
    Input (state_dim) → FC1 → ReLU → FC2 → ReLU → FC3 → Output (n_actions)

Concepts clés:
- Fully Connected (FC): chaque neurone est connecté à tous les neurones précédents
- ReLU (Rectified Linear Unit): fonction d'activation max(0, x)
- Dropout: régularisation pour éviter le surapprentissage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron pour l'approximation de fonctions en RL.
    
    Peut être utilisé pour:
    - DQN: prédire Q(s, a) pour toutes les actions
    - Policy Gradient: prédire π(a|s)
    - Value function: prédire V(s)
    
    Attributes:
        layers: Liste des couches fully-connected
        
    Exemple d'utilisation:
        >>> net = MLP(state_dim=4, output_dim=2, hidden_dims=[64, 64])
        >>> state = torch.randn(1, 4)  # batch_size=1
        >>> q_values = net(state)  # shape: (1, 2)
    """
    
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        dropout: float = 0.0,
        output_activation: Optional[str] = None
    ):
        """
        Initialise le MLP.
        
        Args:
            state_dim: Dimension de l'entrée (taille de l'état)
            output_dim: Dimension de la sortie (nombre d'actions ou 1 pour valeur)
            hidden_dims: Liste des dimensions des couches cachées
                Ex: [64, 64] pour 2 couches de 64 neurones
            activation: Fonction d'activation ("relu", "tanh", "elu")
            dropout: Taux de dropout (0 = pas de dropout)
            output_activation: Activation de sortie (None, "softmax", "tanh")
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Choisir la fonction d'activation
        self.activation_fn = self._get_activation(activation)
        self.output_activation = output_activation
        
        # Construire les couches
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            # Couche fully-connected
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Activation
            layers.append(self.activation_fn)
            # Dropout optionnel
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Créer le réseau séquentiel
        self.network = nn.Sequential(*layers)
        
        # Initialisation des poids (important pour la stabilité)
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Retourne la fonction d'activation correspondante."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "silu": nn.SiLU(),  # Swish
        }
        if name not in activations:
            raise ValueError(f"Activation inconnue: {name}. Choix: {list(activations.keys())}")
        return activations[name]
    
    def _init_weights(self):
        """
        Initialise les poids du réseau.
        
        Utilise l'initialisation Xavier/Glorot pour les couches linéaires.
        C'est important pour:
        - Éviter les gradients qui explosent ou s'évanouissent
        - Accélérer la convergence
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialisation Xavier pour les poids
                nn.init.xavier_uniform_(module.weight)
                # Biais à 0
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du réseau.
        
        Args:
            state: Tenseur d'état de forme (batch_size, state_dim)
        
        Returns:
            Tenseur de sortie de forme (batch_size, output_dim)
        """
        x = self.network(state)
        
        # Appliquer l'activation de sortie si spécifiée
        if self.output_activation == "softmax":
            x = F.softmax(x, dim=-1)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)
        elif self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Retourne le nombre total de paramètres."""
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"MLP(input={self.state_dim}, hidden={self.hidden_dims}, "
            f"output={self.output_dim}, params={self.get_num_params():,})"
        )


class DuelingMLP(nn.Module):
    """
    Dueling DQN Network Architecture.
    
    Sépare l'estimation de la valeur d'état V(s) et de l'avantage A(s, a):
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
    
    Avantages:
    - Meilleure généralisation car V(s) est partagé
    - Apprend plus vite quand les actions ont des effets similaires
    - Plus stable que DQN standard
    
    Architecture:
        State → Shared Layers → ┬→ Value Stream  → V(s)
                                └→ Advantage Stream → A(s, a)
                                
        Q(s, a) = V(s) + (A(s, a) - mean(A))
    
    Référence:
        "Dueling Network Architectures for Deep RL" (Wang et al., 2016)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu"
    ):
        """
        Initialise le réseau Dueling DQN.
        
        Args:
            state_dim: Dimension de l'état
            n_actions: Nombre d'actions
            hidden_dims: Dimensions des couches cachées partagées
            activation: Fonction d'activation
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Couches partagées (feature extraction)
        self.shared = MLP(
            state_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation
        )
        
        # Stream de valeur: produit V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)  # Une seule valeur
        )
        
        # Stream d'avantage: produit A(s, a) pour chaque action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calcule les valeurs Q pour toutes les actions.
        
        Args:
            state: Tenseur d'état (batch_size, state_dim)
        
        Returns:
            Valeurs Q (batch_size, n_actions)
        """
        # Extraction de features partagées
        features = self.shared(state)
        
        # Calculer V(s) et A(s, a)
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)
        
        # Combiner: Q = V + (A - mean(A))
        # Soustraire la moyenne pour identifier l'action
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Retourne uniquement la valeur d'état V(s)."""
        features = self.shared(state)
        return self.value_stream(features)
    
    def get_advantage(self, state: torch.Tensor) -> torch.Tensor:
        """Retourne les avantages A(s, a)."""
        features = self.shared(state)
        return self.advantage_stream(features)


class ActorCriticMLP(nn.Module):
    """
    Réseau Actor-Critic pour les algorithmes Policy Gradient.
    
    Combine deux "têtes":
    - Actor (π): produit une distribution de probabilités sur les actions
    - Critic (V): estime la valeur de l'état
    
    Architecture:
        State → Shared Layers → ┬→ Actor Head  → π(a|s)
                                └→ Critic Head → V(s)
    
    Utilisé par:
    - A2C (Advantage Actor-Critic)
    - PPO (Proximal Policy Optimization)
    - A3C (Asynchronous A3C)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        shared: bool = True
    ):
        """
        Initialise le réseau Actor-Critic.
        
        Args:
            state_dim: Dimension de l'état
            n_actions: Nombre d'actions
            hidden_dims: Dimensions des couches cachées
            activation: Fonction d'activation
            shared: Si True, partage les couches entre Actor et Critic
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.shared = shared
        
        if shared:
            # Couches partagées
            self.shared_layers = MLP(
                state_dim=state_dim,
                output_dim=hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                activation=activation
            )
            
            # Têtes séparées
            self.actor_head = nn.Linear(hidden_dims[-1], n_actions)
            self.critic_head = nn.Linear(hidden_dims[-1], 1)
        else:
            # Réseaux complètement séparés
            self.actor = MLP(
                state_dim=state_dim,
                output_dim=n_actions,
                hidden_dims=hidden_dims,
                activation=activation
            )
            self.critic = MLP(
                state_dim=state_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                activation=activation
            )
    
    def forward(self, state: torch.Tensor):
        """
        Retourne à la fois les logits de la politique et la valeur.
        
        Args:
            state: Tenseur d'état
        
        Returns:
            Tuple (policy_logits, value)
        """
        if self.shared:
            features = self.shared_layers(state)
            policy_logits = self.actor_head(features)
            value = self.critic_head(features)
        else:
            policy_logits = self.actor(state)
            value = self.critic(state)
        
        return policy_logits, value
    
    def get_policy(self, state: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités d'action (softmax des logits)."""
        policy_logits, _ = self.forward(state)
        return F.softmax(policy_logits, dim=-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Retourne la valeur d'état."""
        _, value = self.forward(state)
        return value
    
    def get_action_and_value(self, state: torch.Tensor):
        """
        Échantillonne une action et retourne sa log-probabilité et la valeur.
        
        Utile pour l'entraînement des algorithmes policy gradient.
        
        Returns:
            Tuple (action, log_prob, value)
        """
        policy_logits, value = self.forward(state)
        
        # Créer une distribution catégorique
        dist = torch.distributions.Categorical(logits=policy_logits)
        
        # Échantillonner une action
        action = dist.sample()
        
        # Log-probabilité de l'action
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)


# Tests
if __name__ == "__main__":
    print("=== Test des réseaux de neurones ===\n")
    
    # Configuration
    state_dim = 10
    n_actions = 4
    batch_size = 32
    
    # Test MLP simple
    print("1. Test MLP:")
    mlp = MLP(state_dim=state_dim, output_dim=n_actions, hidden_dims=[64, 64])
    print(f"   {mlp}")
    
    state = torch.randn(batch_size, state_dim)
    output = mlp(state)
    print(f"   Input shape: {state.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test Dueling MLP
    print("\n2. Test Dueling MLP:")
    dueling = DuelingMLP(state_dim=state_dim, n_actions=n_actions, hidden_dims=[64, 64])
    q_values = dueling(state)
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   V(s) shape: {dueling.get_value(state).shape}")
    print(f"   A(s,a) shape: {dueling.get_advantage(state).shape}")
    
    # Test Actor-Critic
    print("\n3. Test Actor-Critic:")
    ac = ActorCriticMLP(state_dim=state_dim, n_actions=n_actions, hidden_dims=[64, 64])
    policy_logits, value = ac(state)
    print(f"   Policy logits shape: {policy_logits.shape}")
    print(f"   Value shape: {value.shape}")
    
    action, log_prob, val = ac.get_action_and_value(state)
    print(f"   Sampled action shape: {action.shape}")
    print(f"   Log prob shape: {log_prob.shape}")
    
    # Test gradients
    print("\n4. Test des gradients:")
    loss = output.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in mlp.parameters())
    print(f"   Gradients calculés: {has_grad}")
    
    print("\n[OK] Tests passes!")
