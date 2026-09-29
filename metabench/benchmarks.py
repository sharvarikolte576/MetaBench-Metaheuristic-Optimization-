"""
Benchmark Topologies Module for MetaBench.

Defines 20 continuous-domain benchmark functions spanning diverse landscape
geometries (unimodal, multimodal, valley-shaped, plate-shaped, separable, non-separable),
supporting arbitrary dimensionality up to 100-D where scalable, as well as fixed 2D benchmarks.

All functions support both single-vector (D,) and batch (N, D) evaluations.
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict, List
import numpy as np


@dataclass
class BenchmarkFunction:
    """Represents a continuous optimization benchmark topology."""
    name: str
    dimension: int
    is_scalable: bool
    bounds: Tuple[float, float] | np.ndarray  # (lower, upper) or (2, D)
    global_minimum_val: float
    global_optimum_coords: Optional[np.ndarray]
    landscape_type: str  # 'unimodal', 'multimodal', 'valley', 'plate'
    separability: str    # 'separable', 'non-separable'
    func: Callable[[np.ndarray], np.ndarray]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate objective function.
        Args:
            x: shape (D,) or (N, D)
        Returns:
            scalar or shape (N,) float values
        """
        x_arr = np.asarray(x, dtype=np.float64)
        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_arr = x_arr.reshape(1, -1)
        
        vals = self.func(x_arr)
        
        if is_1d:
            return float(vals[0])
        return vals

    def get_bounds(self, dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower_bounds, upper_bounds) as 1D arrays of length dim."""
        if not self.is_scalable:
            d = self.dimension
        else:
            d = dim if dim is not None else self.dimension
        if isinstance(self.bounds, tuple):
            low, high = self.bounds
            return np.full(d, low, dtype=np.float64), np.full(d, high, dtype=np.float64)
        else:
            bounds_arr = np.asarray(self.bounds, dtype=np.float64)
            if bounds_arr.shape == (2,):
                return np.full(d, bounds_arr[0]), np.full(d, bounds_arr[1])
            return bounds_arr[0, :d], bounds_arr[1, :d]


# -------------------------------------------------------------------------
# Mathematical Benchmark Implementations (Vectorized for (N, D) input)
# -------------------------------------------------------------------------

def _sphere(x: np.ndarray) -> np.ndarray:
    """f(x) = sum(x_i^2)"""
    return np.sum(x ** 2, axis=-1)


def _sum_of_squares(x: np.ndarray) -> np.ndarray:
    """f(x) = sum(i * x_i^2)"""
    d = x.shape[-1]
    weights = np.arange(1, d + 1, dtype=np.float64)
    return np.sum(weights * (x ** 2), axis=-1)


def _zakharov(x: np.ndarray) -> np.ndarray:
    """f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4"""
    d = x.shape[-1]
    weights = 0.5 * np.arange(1, d + 1, dtype=np.float64)
    term1 = np.sum(x ** 2, axis=-1)
    term2 = np.sum(weights * x, axis=-1)
    return term1 + (term2 ** 2) + (term2 ** 4)


def _dixon_price(x: np.ndarray) -> np.ndarray:
    """f(x) = (x_1 - 1)^2 + sum_{i=2}^D i * (2*x_i^2 - x_{i-1})^2"""
    d = x.shape[-1]
    term1 = (x[:, 0] - 1.0) ** 2
    if d == 1:
        return term1
    i_vals = np.arange(2, d + 1, dtype=np.float64)
    diff = 2.0 * (x[:, 1:] ** 2) - x[:, :-1]
    term2 = np.sum(i_vals * (diff ** 2), axis=-1)
    return term1 + term2


def _rosenbrock(x: np.ndarray) -> np.ndarray:
    """f(x) = sum_{i=1}^{D-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]"""
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=-1)


def _beale(x: np.ndarray) -> np.ndarray:
    """Beale function (2D)."""
    x1, x2 = x[:, 0], x[:, 1]
    t1 = (1.5 - x1 + x1 * x2) ** 2
    t2 = (2.25 - x1 + x1 * (x2 ** 2)) ** 2
    t3 = (2.625 - x1 + x1 * (x2 ** 3)) ** 2
    return t1 + t2 + t3


def _bukin_n6(x: np.ndarray) -> np.ndarray:
    """Bukin function N.6 (2D)."""
    x1, x2 = x[:, 0], x[:, 1]
    return 100.0 * np.sqrt(np.abs(x2 - 0.01 * (x1 ** 2))) + 0.01 * np.abs(x1 + 10.0)


def _rastrigin(x: np.ndarray) -> np.ndarray:
    """f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))"""
    d = x.shape[-1]
    return 10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def _ackley(x: np.ndarray) -> np.ndarray:
    """Ackley function."""
    d = x.shape[-1]
    sum_sq = np.sum(x ** 2, axis=-1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x), axis=-1)
    t1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
    t2 = -np.exp(sum_cos / d)
    return t1 + t2 + 20.0 + np.e


def _griewank(x: np.ndarray) -> np.ndarray:
    """Griewank function."""
    d = x.shape[-1]
    i_vals = np.sqrt(np.arange(1, d + 1, dtype=np.float64))
    sum_term = np.sum(x ** 2, axis=-1) / 4000.0
    prod_term = np.prod(np.cos(x / i_vals), axis=-1)
    return sum_term - prod_term + 1.0


def _schwefel(x: np.ndarray) -> np.ndarray:
    """Schwefel function. Global minimum at x_i = 420.9687, f(x*) = 0."""
    d = x.shape[-1]
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)


def _levy(x: np.ndarray) -> np.ndarray:
    """Levy function."""
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term2 = np.sum((w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2), axis=-1)
    term3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    return term1 + term2 + term3


def _michalewicz(x: np.ndarray, m: int = 10) -> np.ndarray:
    """Michalewicz function."""
    d = x.shape[-1]
    i_vals = np.arange(1, d + 1, dtype=np.float64)
    sin_term = np.sin(i_vals * (x ** 2) / np.pi) ** (2 * m)
    return -np.sum(np.sin(x) * sin_term, axis=-1)


def _booth(x: np.ndarray) -> np.ndarray:
    """Booth function (2D)."""
    x1, x2 = x[:, 0], x[:, 1]
    return (x1 + 2.0 * x2 - 7.0) ** 2 + (2.0 * x1 + x2 - 5.0) ** 2


def _matyas(x: np.ndarray) -> np.ndarray:
    """Matyas function (2D)."""
    x1, x2 = x[:, 0], x[:, 1]
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2


def _cross_in_tray(x: np.ndarray) -> np.ndarray:
    """Cross-in-Tray function (2D). Min val: -2.06261"""
    x1, x2 = x[:, 0], x[:, 1]
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(np.abs(100.0 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))
    return -0.0001 * (np.abs(fact1 * fact2) + 1.0) ** 0.1


def _holder_table(x: np.ndarray) -> np.ndarray:
    """Holder Table function (2D). Min val: -19.2085"""
    x1, x2 = x[:, 0], x[:, 1]
    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(np.abs(1.0 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))
    return -np.abs(fact1 * fact2)


def _schaffer_n2(x: np.ndarray) -> np.ndarray:
    """Schaffer function N.2 (2D)."""
    x1, x2 = x[:, 0], x[:, 1]
    num = np.sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5
    den = (1.0 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2
    return 0.5 + num / den


def _shubert(x: np.ndarray) -> np.ndarray:
    """Shubert function (2D). Min val: -186.7309"""
    x1, x2 = x[:, 0], x[:, 1]
    j_vals = np.arange(1, 6, dtype=np.float64)
    # Shape of outer product: (N, 5)
    t1 = np.sum([i * np.cos((i + 1) * x1 + i) for i in j_vals], axis=0)
    t2 = np.sum([i * np.cos((i + 1) * x2 + i) for i in j_vals], axis=0)
    return t1 * t2


def _trid(x: np.ndarray) -> np.ndarray:
    """Trid function. Global minimum f(x*) = -D(D+4)(D-1)/6."""
    d = x.shape[-1]
    t1 = np.sum((x - 1.0) ** 2, axis=-1)
    t2 = np.sum(x[:, 1:] * x[:, :-1], axis=-1)
    return t1 - t2


# -------------------------------------------------------------------------
# Factory / Registry for the 20 Benchmark Functions
# -------------------------------------------------------------------------

def create_benchmark(name: str, dim: int = 10) -> BenchmarkFunction:
    """
    Instantiate a BenchmarkFunction by name and target dimension.
    Supported 20 topologies:
    1. Sphere            (Unimodal, Scalable)
    2. Sum of Squares    (Unimodal, Scalable)
    3. Zakharov          (Unimodal, Scalable)
    4. Dixon-Price       (Unimodal, Scalable)
    5. Rosenbrock        (Valley-shaped, Scalable)
    6. Beale             (Valley-shaped, 2D)
    7. Bukin N.6         (Valley-shaped, 2D)
    8. Rastrigin         (Multimodal, Scalable)
    9. Ackley            (Multimodal, Scalable)
    10. Griewank         (Multimodal, Scalable)
    11. Schwefel         (Multimodal, Scalable)
    12. Levy             (Multimodal, Scalable)
    13. Michalewicz      (Multimodal, Scalable)
    14. Booth            (Plate-shaped, 2D)
    15. Matyas           (Plate-shaped, 2D)
    16. Cross-in-Tray    (Multimodal, 2D)
    17. Holder Table     (Multimodal, 2D)
    18. Schaffer N.2     (Multimodal, 2D)
    19. Shubert          (Multimodal, 2D)
    20. Trid             (Valley/Multimodal, Scalable)
    """
    name_norm = name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")

    benchmarks_registry: Dict[str, Dict] = {
        "sphere": {
            "name": "Sphere",
            "is_scalable": True,
            "bounds": (-5.12, 5.12),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.zeros(d),
            "landscape": "unimodal",
            "separability": "separable",
            "func": _sphere
        },
        "sum_of_squares": {
            "name": "Sum of Squares",
            "is_scalable": True,
            "bounds": (-10.0, 10.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.zeros(d),
            "landscape": "unimodal",
            "separability": "separable",
            "func": _sum_of_squares
        },
        "zakharov": {
            "name": "Zakharov",
            "is_scalable": True,
            "bounds": (-5.0, 10.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.zeros(d),
            "landscape": "unimodal",
            "separability": "non-separable",
            "func": _zakharov
        },
        "dixon_price": {
            "name": "Dixon-Price",
            "is_scalable": True,
            "bounds": (-10.0, 10.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.array([2.0 ** (-(2.0**i - 2.0) / (2.0**i)) for i in range(1, d + 1)]),
            "landscape": "unimodal",
            "separability": "non-separable",
            "func": _dixon_price
        },
        "rosenbrock": {
            "name": "Rosenbrock",
            "is_scalable": True,
            "bounds": (-5.0, 10.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.ones(d),
            "landscape": "valley",
            "separability": "non-separable",
            "func": _rosenbrock
        },
        "beale": {
            "name": "Beale",
            "is_scalable": False,
            "bounds": (-4.5, 4.5),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.array([3.0, 0.5]),
            "landscape": "valley",
            "separability": "non-separable",
            "func": _beale
        },
        "bukin_n6": {
            "name": "Bukin N.6",
            "is_scalable": False,
            "bounds": np.array([[-15.0, -3.0], [-5.0, 3.0]]),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.array([-10.0, 1.0]),
            "landscape": "valley",
            "separability": "non-separable",
            "func": _bukin_n6
        },
        "rastrigin": {
            "name": "Rastrigin",
            "is_scalable": True,
            "bounds": (-5.12, 5.12),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.zeros(d),
            "landscape": "multimodal",
            "separability": "separable",
            "func": _rastrigin
        },
        "ackley": {
            "name": "Ackley",
            "is_scalable": True,
            "bounds": (-32.768, 32.768),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.zeros(d),
            "landscape": "multimodal",
            "separability": "non-separable",
            "func": _ackley
        },
        "griewank": {
            "name": "Griewank",
            "is_scalable": True,
            "bounds": (-600.0, 600.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.zeros(d),
            "landscape": "multimodal",
            "separability": "non-separable",
            "func": _griewank
        },
        "schwefel": {
            "name": "Schwefel",
            "is_scalable": True,
            "bounds": (-500.0, 500.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.full(d, 420.9687),
            "landscape": "multimodal",
            "separability": "separable",
            "func": _schwefel
        },
        "levy": {
            "name": "Levy",
            "is_scalable": True,
            "bounds": (-10.0, 10.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.ones(d),
            "landscape": "multimodal",
            "separability": "non-separable",
            "func": _levy
        },
        "michalewicz": {
            "name": "Michalewicz",
            "is_scalable": True,
            "bounds": (0.0, np.pi),
            "opt_val": -9.66015 if dim == 10 else (-1.8013 if dim == 2 else -0.966 * dim),
            "opt_coords": lambda d: None,
            "landscape": "multimodal",
            "separability": "separable",
            "func": _michalewicz
        },
        "booth": {
            "name": "Booth",
            "is_scalable": False,
            "bounds": (-10.0, 10.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.array([1.0, 3.0]),
            "landscape": "plate",
            "separability": "non-separable",
            "func": _booth
        },
        "matyas": {
            "name": "Matyas",
            "is_scalable": False,
            "bounds": (-10.0, 10.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.array([0.0, 0.0]),
            "landscape": "plate",
            "separability": "non-separable",
            "func": _matyas
        },
        "cross_in_tray": {
            "name": "Cross-in-Tray",
            "is_scalable": False,
            "bounds": (-10.0, 10.0),
            "opt_val": -2.06261,
            "opt_coords": lambda d: np.array([1.34941, -1.34941]),
            "landscape": "multimodal",
            "separability": "non-separable",
            "func": _cross_in_tray
        },
        "holder_table": {
            "name": "Holder Table",
            "is_scalable": False,
            "bounds": (-10.0, 10.0),
            "opt_val": -19.2085,
            "opt_coords": lambda d: np.array([8.05502, 9.66459]),
            "landscape": "multimodal",
            "separability": "non-separable",
            "func": _holder_table
        },
        "schaffer_n2": {
            "name": "Schaffer N.2",
            "is_scalable": False,
            "bounds": (-100.0, 100.0),
            "opt_val": 0.0,
            "opt_coords": lambda d: np.array([0.0, 0.0]),
            "landscape": "multimodal",
            "separability": "non-separable",
            "func": _schaffer_n2
        },
        "shubert": {
            "name": "Shubert",
            "is_scalable": False,
            "bounds": (-10.0, 10.0),
            "opt_val": -186.7309,
            "opt_coords": lambda d: None,
            "landscape": "multimodal",
            "separability": "non-separable",
            "func": _shubert
        },
        "trid": {
            "name": "Trid",
            "is_scalable": True,
            "bounds": (-dim**2, dim**2) if dim > 1 else (-10.0, 10.0),
            "opt_val": -float(dim * (dim + 4) * (dim - 1)) / 6.0,
            "opt_coords": lambda d: np.array([i * (d + 1 - i) for i in range(1, d + 1)], dtype=np.float64),
            "landscape": "valley",
            "separability": "non-separable",
            "func": _trid
        }
    }

    if name_norm not in benchmarks_registry:
        available = ", ".join(benchmarks_registry.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")

    spec = benchmarks_registry[name_norm]
    actual_dim = dim if spec["is_scalable"] else 2
    opt_coords = spec["opt_coords"](actual_dim) if callable(spec["opt_coords"]) else None

    bounds = spec["bounds"]
    if spec["name"] == "Trid":
        bounds = (-float(actual_dim**2), float(actual_dim**2))

    return BenchmarkFunction(
        name=spec["name"],
        dimension=actual_dim,
        is_scalable=spec["is_scalable"],
        bounds=bounds,
        global_minimum_val=spec["opt_val"],
        global_optimum_coords=opt_coords,
        landscape_type=spec["landscape"],
        separability=spec["separability"],
        func=spec["func"]
    )


def get_all_benchmark_names() -> List[str]:
    """Returns list of the 20 benchmark topologies."""
    return [
        "Sphere",
        "Sum of Squares",
        "Zakharov",
        "Dixon-Price",
        "Rosenbrock",
        "Beale",
        "Bukin N.6",
        "Rastrigin",
        "Ackley",
        "Griewank",
        "Schwefel",
        "Levy",
        "Michalewicz",
        "Booth",
        "Matyas",
        "Cross-in-Tray",
        "Holder Table",
        "Schaffer N.2",
        "Shubert",
        "Trid"
    ]


def get_benchmark_suite(dim: int = 10) -> List[BenchmarkFunction]:
    """Instantiate all 20 benchmark topologies."""
    return [create_benchmark(name, dim=dim) for name in get_all_benchmark_names()]
