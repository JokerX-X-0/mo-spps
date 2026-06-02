from dataclasses import dataclass, field


@dataclass(frozen=True)
class Component:
    """Immutable representation of a single component in the universe V = {1, ..., M}.

    Each component is the basic unit from which candidate solutions are constructed.
    Corresponds to component_j = (id_j, attr_j, rho_j, Q_j) in Section 3.1.
    """

    id: int
    attributes: dict = field(default_factory=dict)
    base_weight: float = 1.0
    pool_capacity: int = 1

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Component):
            return False
        return self.id == other.id
