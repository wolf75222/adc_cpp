"""User-facing symbolic operation constructors for the ADC IR (Spec 4).

Moved verbatim from :mod:`adc.dsl` and :mod:`adc.math`. NO C++ emission: each
function builds a symbolic node from :mod:`adc.ir.expr` / :mod:`adc.ir.values`.

  - flux-DSL ops (from :mod:`adc.dsl`): :func:`sqrt`, :func:`abs_`, :func:`sign`,
    :func:`eig_max_im`, :func:`eig_lmin`, :func:`eig_lmax`, :func:`eig_all_real`,
    :func:`left`, :func:`right`;
  - board ops (from :mod:`adc.math`): :func:`grad`, :func:`dx`, :func:`dy`,
    :func:`laplacian`, :func:`div`, :func:`ddt`, :func:`rate`, :func:`unknown`,
    :func:`integral`, :func:`board_sqrt`.

sqrt-clash resolution -- :mod:`adc.dsl` and :mod:`adc.math` both export a
``sqrt``. The flux-DSL ``sqrt`` is the symbolic primitive (``Sqrt(_wrap(x))``)
and is the canonical :func:`sqrt` here. The board ``sqrt`` (numeric-or-symbolic,
with a host-numeric fallback) is preserved verbatim as :func:`board_sqrt`; its
body still delegates to :func:`adc.dsl.sqrt` lazily, exactly as in
:mod:`adc.math`. The package re-exports the flux-DSL :func:`sqrt` as ``sqrt``.
"""
from .expr import (
    Abs,
    Divergence,
    Gradient,
    Integral,
    Laplacian,
    Partial,
    Sign,
    Sqrt,
    TimeDerivative,
    Unknown,
    _wrap,
)
from .values import EigWitness, StateRef


# ===========================================================================
# Flux-DSL operations (from adc.dsl)
# ===========================================================================


def sqrt(x):
    """Symbolic square root."""
    return Sqrt(_wrap(x))


def abs_(x):
    """Symbolic absolute value (equivalent of abs(expr); suffixed name so as not to shadow abs)."""
    return Abs(_wrap(x))


def sign(x):
    """Signe symbolique (-1 / 0 / 1) : selections par masques sans branche par cellule."""
    return Sign(_wrap(x))


def eig_max_im(rows):
    """Temoin de VALEURS PROPRES COMPLEXES d'une petite matrice dense @p rows (liste de @c k lignes de
    @c k Expr) : valeur scalaire = max des |Im(lambda)| (0 = spectre reel, donc hyperbolique), via
    ``adc::real_eig_minmax`` (ADC-289). Sert les selections branchless de m.projection (p.ex. ``si
    max_im > tol, corriger``, ecrit en max/min/sign). Le caller assemble la matrice (bloc 3x3 du
    jacobien, compagnon...) a partir d'expressions de moments ; aucune physique n'est dans le coeur."""
    return EigWitness(rows, "max_im")


def eig_lmin(rows):
    """Plus petite PARTIE REELLE du spectre de la matrice dense @p rows (cf. eig_max_im), via
    ``adc::real_eig_minmax`` -- valeur scalaire DSL (extreme de borne de vitesse / spectre reel)."""
    return EigWitness(rows, "lmin")


def eig_lmax(rows):
    """Plus grande PARTIE REELLE du spectre de la matrice dense @p rows (cf. eig_max_im), via
    ``adc::real_eig_minmax`` -- valeur scalaire DSL (extreme de borne de vitesse / spectre reel)."""
    return EigWitness(rows, "lmax")


def eig_all_real(rows, im_tol=1e-5):
    """PREDICAT reel/complexe du spectre de la PETITE matrice dense @p rows (liste de @c k lignes de
    @c k Expr) : valeur scalaire DSL = 1.0 si le spectre est REEL et le bloc a CONVERGE, 0.0 sinon
    (paire complexe OU non-convergence). Surface DSL d'``adc::EigBounds::all_real`` (dense_eig.hpp,
    ADC-276) ; @p im_tol est le seuil RELATIF d'|Im| (defaut 1e-5, mis a l'echelle par
    max(|lmin|,|lmax|,1) -- cf. EigBounds::all_real pour le contrat de tolerance, la couverture de
    multiplicite et l'asymetrie relative).

    A PREFERER a ``eig_max_im(rows) <= tol`` pour decider "spectre reel ?" : ``all_real`` est verrouille
    sur ``converged``, donc un repli Gershgorin (non-convergence) rend 0.0 (= PAS reel), JAMAIS 1.0 ;
    le ``max_im`` brut vaut 0 sous repli PAR CONVENTION et serait lu a tort comme un spectre reel.
    Compose sans branche dans m.projection : ``complexe = 1 - eig_all_real(rows)``, puis un melange
    max/min sans if (p.ex. "si paire complexe, corriger").

    Codegen device-clean : foncteur nomme (methode statique ADC_HD) qui remplit M[k][k] et renvoie
    ``adc::Real(adc::real_eig_minmax(M).all_real(im_tol))`` (cf. EigWitness ; pas de lambda cross-TU).
    eval(env) : miroir hote via numpy (LAPACK converge toujours -> jamais de kUnknown cote hote ;
    coincide avec la brique sur matrices saines, le cas vise)."""
    return EigWitness(rows, "all_real", im_tol=im_tol)


def left(expr):
    """Marks @p expr as evaluated on the LEFT state UL (Roe dissipation, m.roe_dissipation)."""
    return StateRef("L", expr)


def right(expr):
    """Marks @p expr as evaluated on the RIGHT state UR (Roe dissipation, m.roe_dissipation)."""
    return StateRef("R", expr)


# ===========================================================================
# Board operations (from adc.math)
# ===========================================================================


# --- public constructors -----------------------------------------------------
def grad(field):
    """The gradient of a scalar field; use ``grad(phi).x`` / ``.y`` for components."""
    return Gradient(field)


def dx(field):
    """The x partial derivative of a field (``grad(field).x``)."""
    return Partial(field, 0)


def dy(field):
    """The y partial derivative of a field (``grad(field).y``)."""
    return Partial(field, 1)


def laplacian(field):
    """The Laplacian of a field -- the elliptic operator of a Poisson solve."""
    return Laplacian(field)


def div(flux):
    """The divergence of a flux; write ``-div(F)`` for the hyperbolic ``-div F``."""
    return Divergence(flux)


def ddt(state):
    """The time derivative of a state -- the left-hand side of a rate equation."""
    return TimeDerivative(state)


def rate(state):
    """The rate (time derivative) of a state; synonym of :func:`ddt` for time programs."""
    return TimeDerivative(state)


def unknown(name):
    """A named solve unknown, e.g. ``unknown("U*")``."""
    return Unknown(name)


def integral(expr, over=None):
    """The spatial integral of ``expr`` -- the value of a generic invariant."""
    return Integral(expr, over=over)


def board_sqrt(x):
    """Square root, lifted to the board: delegates to :func:`adc.dsl.sqrt` on an
    expression / parameter, falls back to :func:`math.sqrt` on a plain number."""
    try:
        from .. import dsl as _dsl
        return _dsl.sqrt(x)
    except Exception:
        import math as _pymath
        return _pymath.sqrt(x)
