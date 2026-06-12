"""Generateur GENERIQUE de modeles de moments 2D (hierarchies de Vlasov / methodes QMOM).

Pour un systeme de moments d'ordre <= N en 2D, TOUTE la chaine M -> C -> S -> fermeture ->
C' -> M' est de l'algebre binomiale systematique SAUF la fermeture (la physique). Ce module
derive cette algebre EN BOUCLES sur l'AST de la DSL (adc.dsl) : l'equivalent du role d'une
toolbox symbolique (MATLAB Symbolic / SymPy) generant les flux hors ligne, mais rejoue a
CHAQUE construction du modele depuis la source unique -- la fermeture change, les flux, la
jacobienne (autodiff) et les vitesses d'onde suivent, sans etape de re-generation manuelle
ni risque de desynchronisation.

L'utilisateur ne fournit que :
  - l'ordre N des moments transportes (N=2 -> 6 variables, N=4 -> 15 variables) ;
  - la fermeture : un callable S -> dict des moments STANDARDISES d'ordre N+1
    (cles 'S{p}{q}'), seule physique du modele. `gaussian_closure(order)` est fournie
    (fermeture de Levermore : cumulants gaussiens, generique en l'ordre) ;
  - optionnellement des sources (`lorentz_sources` : hierarchie Vlasov-Lorentz, generique).

Convention d'ordre des variables (IDENTIQUE au cas hyqmom15 d'adc_cases pour N=4) :
q externe croissant puis p croissant, soit [M00..M40, M01..M31, M02..M22, M03 M13, M04].

Aucune physique nommee ici : fermetures et parametres restent cote utilisateur (doctrine
du coeur generique). Validation croisee chez l'utilisateur de reference (adc_cases
hyqmom15) : flux generes == goldens MATLAB (Flux_closure15_2D.m) a 7.7e-13 et == modele
ecrit a la main a 2.6e-13 sur les 10 etats golden.
"""
from math import comb

from . import dsl


def moment_indices(order):
    """Liste canonique des (p, q) avec p + q <= order : q externe, p interne croissants."""
    if order < 1:
        raise ValueError("moments : order >= 1 requis (ordre %r)" % (order,))
    return [(p, q) for q in range(order + 1) for p in range(order + 1 - q)]


def moment_names(order):
    """Noms canoniques 'M{p}{q}' alignes sur moment_indices(order)."""
    return ["M%d%d" % pq for pq in moment_indices(order)]


def _pow(e, k):
    """e**k par multiplications repetees (k >= 0 ; e Expr DSL ou nombre)."""
    if k == 0:
        return 1.0
    r = e
    for _ in range(k - 1):
        r = r * e
    return r


def _is_zero(e):
    # Zero NUMERIQUE (int/float) ou zero SYMBOLIQUE (dsl.Const(0.0)) : une fermeture peut rendre
    # l'un ou l'autre ; les deux suppriment le terme du flux genere (primitive morte non emise).
    if isinstance(e, (int, float)):
        return float(e) == 0.0
    return isinstance(e, dsl.Const) and e.value == 0.0


def gaussian_closure(order):
    """Fermeture gaussienne (Levermore) generique : les moments standardises d'ordre
    order+1 sont ceux d'une gaussienne standardisee de correlation s11 = S['S11'].

    Ordre+1 IMPAIR : tous nuls (moments centres impairs d'une gaussienne). Ordre+1 PAIR :
    recurrence de Stein standardisee m_pq = (p-1) m_{p-2,q} + q s11 m_{p-1,q-1}.
    @return callable S -> dict 'S{p}{q}' (p+q = order+1)."""
    top = order + 1

    def closure(S):
        if top % 2 == 1:
            return {"S%d%d" % (p, top - p): 0.0 for p in range(top + 1)}
        s11 = S["S11"]
        memo = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0}

        def m(p, q):
            if p < 0 or q < 0:
                return 0.0
            if (p, q) not in memo:
                if p >= 1:
                    memo[(p, q)] = (p - 1) * m(p - 2, q) + (q * s11 * m(p - 1, q - 1)
                                                            if q >= 1 else 0.0)
                else:
                    memo[(p, q)] = (q - 1) * m(p, q - 2)
            return memo[(p, q)]

        return {"S%d%d" % (p, top - p): m(p, top - p) for p in range(top + 1)}

    return closure


def lorentz_sources(M, ex, ey, q_over_m, omega_c):
    """Sources de la hierarchie de moments sous force de Lorentz (Vlasov), generiques en
    l'ordre et INDEPENDANTES de la fermeture (aucun moment d'ordre superieur reference :
    le terme electrique ABAISSE l'ordre, le terme magnetique le CONSERVE) :

        S[M_pq] = q_over_m (p ex M_{p-1,q} + q ey M_{p,q-1}) + omega_c (p M_{p-1,q+1} - q M_{p+1,q-1})

    @p M : dict (p, q) -> Expr/valeur des moments transportes (cles = moment_indices).
    @p ex, ey : champ electrique (Expr aux ou valeurs). @p q_over_m, omega_c : Expr param
    ou valeurs. @return liste alignee sur moment_indices(order). Accepte des nombres purs
    partout (utilisable comme oracle numerique)."""
    order = max(p + q for (p, q) in M)
    out = []
    for (p, q) in moment_indices(order):
        expr = None
        if p >= 1:
            t = q_over_m * (float(p) * ex * M[(p - 1, q)])
            expr = t if expr is None else expr + t
            t = omega_c * (float(p) * M[(p - 1, q + 1)])
            expr = expr + t
        if q >= 1:
            t = q_over_m * (float(q) * ey * M[(p, q - 1)])
            expr = t if expr is None else expr + t
            t = omega_c * (-float(q) * M[(p + 1, q - 1)])
            expr = expr + t
        out.append(0.0 if expr is None else expr)
    return out


def build_moment_model(name, order, closure, blocks=None, exact_speeds=True,
                       robust=False, eps_m00=1e-12, eps_cov=1e-12, sources=None):
    """Modele de moments 2D a fermeture arbitraire : flux et intermediaires GENERES.

    @p order : ordre max des moments transportes (order=2 -> 6 variables, order=4 -> 15).
    @p closure : callable S -> dict 'S{p}{q}' des moments standardises d'ordre order+1
       (TOUTES les cles p+q = order+1 requises ; valeurs Expr DSL ou nombres -- un zero
       numerique supprime le terme du flux genere). S contient les standardises let-bound
       pour 2 <= p+q <= order, avec S20 = S02 = 1.0 exacts (identites de standardisation).
    @p blocks : structure par blocs de la jacobienne pour l'eig (pass-through vers
       m.wave_speeds_from_jacobian ; defaut matrice pleine). Ignore si exact_speeds=False.
    @p exact_speeds : True = vitesses d'onde exactes par autodiff du flux + valeurs propres
       numeriques par cellule (riemann='hll' fidele). False = l'appelant posera lui-meme
       m.eigenvalues / m.wave_speeds (p.ex. borne de bring-up).
    @p robust : True = planchers lisses max(x, eps) = ((x+eps)+|x-eps|)/2 sur M00 (division)
       et C20/C02 (sqrt) -- derivables (diff(Abs)), donc compatibles exact_speeds. False =
       chemin nu, fidele aux references sans gardes (peut produire NaN sur etat degenere).
    @p sources : callable (m, M) -> liste d'Expr (alignee sur moment_indices), cable via
       m.source ; M = dict (p, q) -> variable conservative. Voir lorentz_sources.
    @return adc.dsl.Model pret a compiler (l'appelant peut encore ajouter elliptic_rhs,
       params, aux... avant m.compile)."""
    if order < 2:
        raise ValueError("build_moment_model : order >= 2 requis (la standardisation "
                         "s'appuie sur C20/C02 ; ordre %r)" % (order,))
    idx = moment_indices(order)
    m = dsl.Model(name)
    cons = m.conservative_vars(*moment_names(order))
    M = dict(zip(idx, cons))

    def floor(nm, x, eps):
        # max(x, eps) = ((x + eps) + |x - eps|) / 2 : plancher lisse, exprimable dans l'AST.
        return m.primitive(nm, ((x + eps) + dsl.abs_(x - eps)) / 2.0)

    M00 = floor("M00f", M[(0, 0)], eps_m00) if robust else M[(0, 0)]
    u = m.primitive("u", M[(1, 0)] / M00)
    v = m.primitive("v", M[(0, 1)] / M00)

    # moments bruts normalises m_pq = M_pq / M00 (pas de let : reutilises une fois chacun)
    mn = {pq: (1.0 if pq == (0, 0) else M[pq] / M00) for pq in idx}

    # --- moments centres : transformation binomiale, derivee en boucle ---
    # C_pq = sum_{i<=p, j<=q} comb(p,i) comb(q,j) (-u)^(p-i) (-v)^(q-j) m_ij
    C = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0}
    for s in range(2, order + 1):
        for q in range(s + 1):
            p = s - q
            expr = None
            for i in range(p + 1):
                for j in range(q + 1):
                    coef = float(comb(p, i) * comb(q, j) * (-1) ** (p - i + q - j))
                    t = coef * _pow(u, p - i) * _pow(v, q - j)
                    if (i, j) != (0, 0):
                        t = t * mn[(i, j)]
                    expr = t if expr is None else expr + t
            C[(p, q)] = m.primitive("C%d%d" % (p, q), expr)

    # --- standardisation : S_pq = C_pq / (sx^p sy^q) ; S20 = S02 = 1 par construction ---
    C20 = floor("C20f", C[(2, 0)], eps_cov) if robust else C[(2, 0)]
    C02 = floor("C02f", C[(0, 2)], eps_cov) if robust else C[(0, 2)]
    sx = m.primitive("sx", dsl.sqrt(C20))
    sy = m.primitive("sy", dsl.sqrt(C02))
    S = {"S20": 1.0, "S02": 1.0}
    for (p, q), c in C.items():
        if p + q >= 2 and (p, q) not in ((2, 0), (0, 2)):
            S["S%d%d" % (p, q)] = m.primitive("S%d%d" % (p, q),
                                              c / (_pow(sx, p) * _pow(sy, q)))

    # --- fermeture (l'UNIQUE physique) puis de-standardisation C'_pq = S'_pq sx^p sy^q ---
    top = closure(S)
    want = {"S%d%d" % (p, order + 1 - p) for p in range(order + 2)}
    if set(top) != want:
        raise ValueError("moments : la fermeture doit rendre exactement les cles %s "
                         "(recu %s)" % (sorted(want), sorted(top)))
    Call = dict(C)
    for key, e in top.items():
        p, q = int(key[1]), int(key[2])
        Call[(p, q)] = (0.0 if _is_zero(e)
                        else m.primitive("C%d%d" % (p, q), e * _pow(sx, p) * _pow(sy, q)))

    # --- reconstruction des moments bruts d'ordre order+1 : binomiale inverse ---
    # m_pq = sum_{i<=p, j<=q} comb(p,i) comb(q,j) u^(p-i) v^(q-j) C_ij
    Mtop = {}
    for q in range(order + 2):
        p = order + 1 - q
        expr = None
        for i in range(p + 1):
            for j in range(q + 1):
                cij = Call.get((i, j))
                if cij is None or _is_zero(cij):
                    continue
                t = float(comb(p, i) * comb(q, j)) * _pow(u, p - i) * _pow(v, q - j)
                if not (isinstance(cij, float) and cij == 1.0):
                    t = t * cij
                expr = t if expr is None else expr + t
        Mtop[(p, q)] = m.primitive("M%d%d" % (p, q), M00 * expr)

    # --- flux : decalage d'ordre F_x[M_pq] = M_{p+1,q}, F_y[M_pq] = M_{p,q+1} ---
    def raw(pq):
        return M[pq] if pq in M else Mtop[pq]

    m.flux(x=[raw((p + 1, q)) for (p, q) in idx],
           y=[raw((p, q + 1)) for (p, q) in idx])

    if exact_speeds:
        m.wave_speeds_from_jacobian(blocks=blocks)
    if sources is not None:
        m.source(sources(m, M))
    m.primitive_vars(*cons)
    m.conservative_from(list(cons))
    return m
