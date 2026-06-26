"""SSPRK3 (Shu-Osher, ordre 3) sur AMR avec REFLUX PAR ETAGE (ADC-64).

L'AMR sous-cycle (Berger-Oliger) avec reflux conservatif aux interfaces grossier-fin. SSPRK3 y est
cable comme METHODE TEMPORELLE optionnelle (pops.Explicit(ssprk3=True) -> time.kind == "ssprk3"),
mono-bloc (coupleur AmrCouplerMP) ET multi-blocs (moteur AmrRuntime). Le reflux enregistre le FLUX
EFFECTIF du pas SSP, Feff = 1/6 F(U0) + 1/6 F(U1) + 2/3 F(U2) : la correction grossier-fin reste
exactement conservative pour le pas d'ordre 3 (cf. subcycle_level_mp / ssprk3_advance_level).

On verifie cote Python :
  (a) AMR ssprk3 MONO-BLOC et MULTI-BLOCS tournent (etat fini) AVEC patchs fins actifs, et la MASSE
      de chaque bloc est conservee a ~machine (reflux + average_down par etage) ;
  (b) DEFAUT bit-identique : add_block sans ssprk3 (Explicit() == Euler avant) reste deterministe
      (deux runs euler explicit -> dmax == 0), verrou du threading time_method ;
  (c) ORDRE : l'erreur L1 (temporelle, isolee par reference Richardson ssprk3 a dt/8) du pas ssprk3
      est NETTEMENT plus petite que celle du pas euler (explicit) au MEME dt -- ssprk3 ordre 3 >> 1 ;
  (d) imex + ssprk3 : combinaison REJETEE explicitement (ssprk3 = transport explicite, exclusif du
      traitement IMEX de la source ; selecteur time.kind unique, parite avec System) ;
  (e) loader .so + ssprk3 : REJETE explicitement (l'ABI plate du loader natif AMR ne transporte pas
      la methode temporelle -> jamais un repli kEuler silencieux).

Test PUR Python (aucune compilation .so) : ne gate sur rien, toujours execute.
"""
import numpy as np

import pops


def _bump(n, amp):
    """Densite lisse, OFFSET MOYEN NUL (Sum q n solvable en Poisson periodique)."""
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    r = 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    return r + (1.0 - r.mean())


def _scalar_charge(q, B0=1.0):
    return pops.Model(pops.Scalar(), pops.ExB(B0=B0), pops.NoSource(), pops.ChargeDensity(charge=q))


# --- (a) mono-bloc + multi-blocs ssprk3 : fini + masse conservee, patchs fins actifs ---
def _check_mono(n=32):
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=4)
    sim.add_block("ne", _scalar_charge(+1.0),
                  spatial=pops.Spatial(limiter="minmod", flux="rusanov"),
                  time=pops.Explicit(ssprk3=True))  # SSPRK3 mono-bloc (chemin AmrCouplerMP)
    sim.set_poisson(bc="periodic")
    sim.set_refinement(1.05)  # seuil bas -> le bump tague et raffine (patchs fins actifs)
    sim.set_density("ne", _bump(n, 0.40))
    m0 = sim.mass()
    sim.advance(0.002, 12)
    d1 = np.asarray(sim.density())
    m1 = sim.mass()
    assert np.isfinite(d1).all(), "mono-bloc ssprk3 : etat non fini"
    assert sim.n_patches() >= 1, "mono-bloc ssprk3 : aucun patch fin actif"
    drift = abs(m1 - m0) / (abs(m0) + 1.0)
    assert drift < 1e-9, "mono-bloc ssprk3 : masse non conservee (drift relatif=%.2e)" % drift
    print("OK  (a) mono-bloc ssprk3 : fini, %d patch(s) fin(s), masse conservee (drift=%.2e)"
          % (sim.n_patches(), drift))


def _check_multi(n=32):
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=4)
    sim.add_block("ions", _scalar_charge(+1.0),
                  spatial=pops.Spatial(limiter="none", flux="rusanov"),
                  time=pops.Explicit(ssprk3=True))     # SSPRK3 multi-blocs (moteur AmrRuntime)
    sim.add_block("electrons", _scalar_charge(-1.0),
                  spatial=pops.Spatial(limiter="minmod", flux="rusanov"),
                  time=pops.Explicit(ssprk3=True))     # 2e bloc ssprk3, SCHEMA SPATIAL DIFFERENT
    sim.set_poisson(bc="periodic")
    sim.set_refinement(1.05)  # union des tags -> patchs fins actifs
    sim.set_density("ions", _bump(n, 0.40))
    sim.set_density("electrons", _bump(n, 0.20))
    m0i, m0e = sim.mass("ions"), sim.mass("electrons")
    sim.advance(0.002, 12)
    d1i = np.asarray(sim.density("ions"))
    d1e = np.asarray(sim.density("electrons"))
    m1i, m1e = sim.mass("ions"), sim.mass("electrons")
    assert np.isfinite(d1i).all() and np.isfinite(d1e).all(), "multi-blocs ssprk3 : etat non fini"
    assert sim.n_patches() >= 1, "multi-blocs ssprk3 : aucun patch fin actif"
    di = abs(m1i - m0i) / (abs(m0i) + 1.0)
    de = abs(m1e - m0e) / (abs(m0e) + 1.0)
    assert di < 1e-9, "multi-blocs ssprk3 : masse ions non conservee (drift=%.2e)" % di
    assert de < 1e-9, "multi-blocs ssprk3 : masse electrons non conservee (drift=%.2e)" % de
    print("OK  (a) multi-blocs ssprk3 : 2 blocs schemas differents, fini, %d patch(s), masse par "
          "bloc conservee (di=%.2e, de=%.2e)" % (sim.n_patches(), di, de))


# --- (b) defaut bit-identique : euler explicit deterministe (verrou du threading time_method) ---
def _check_default_bit_identical(n=32):
    def run_euler():
        s = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
        s.add_block("ne", _scalar_charge(+1.0),
                    spatial=pops.Spatial(limiter="minmod", flux="rusanov"))  # time defaut = Explicit() euler
        s.set_poisson(bc="periodic")
        s.set_density("ne", _bump(n, 0.40))
        s.advance(0.002, 10)
        return np.asarray(s.density())

    a = run_euler()
    b = run_euler()
    dmax = float(np.abs(a - b).max())
    assert dmax == 0.0, "defaut euler non bit-identique (dmax=%.3e) : threading time_method casse" % dmax
    print("OK  (b) defaut (add_block sans ssprk3 = euler avant) bit-identique : dmax == 0")


# --- (c) ordre : erreur L1 temporelle ssprk3 << euler au meme dt (reference Richardson ssprk3 dt/8) --
def _build_advect(n, kind):
    """AMR mono-bloc, hierarchie FIGEE (regrid_every=0, patch seed central) : SEULE la methode
    temporelle (time) change entre les runs -> l'erreur mesuree est purement TEMPORELLE."""
    s = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
    s.add_block("ne", _scalar_charge(+1.0),
                spatial=pops.Spatial(limiter="none", flux="rusanov"),  # MEME schema spatial pour tous
                time=pops.Explicit(ssprk3=True) if kind == "ssprk3" else pops.Explicit())
    s.set_poisson(bc="periodic")
    s.set_density("ne", _bump(n, 0.40))
    return s


def _check_order(n=32):
    # dt FIXE, sonde a CFL ~ 0.5 : FE + Rusanov reste stable (TVD jusqu'a CFL ~ 1) MAIS son erreur de
    # transport (O(dt^2) locale, ordre 1 globale) est alors GRANDE, de sorte qu'elle DOMINE l'erreur de
    # SPLITTING champ/transport COMMUNE aux deux integrateurs (le coupleur resout Poisson une fois par
    # pas puis gele le champ pendant les etages SSP : ce splitting de Lie est ordre 1 et borne l'ordre
    # GLOBAL des deux schemas). En regime ou l'erreur de transport domine, l'integrateur d'ordre 3 reduit
    # NETTEMENT l'erreur totale (l'erreur de transport SSPRK3 est negligeable devant celle d'Euler). La
    # reference Richardson (SSPRK3 a dt/k) capture la solution temps-convergee, isolant l'ecart restant.
    probe = _build_advect(n, "euler")
    probe.step_cfl(0.5)
    dt = float(probe.time())
    assert dt > 0.0 and np.isfinite(dt), "sonde CFL : dt invalide (%r)" % dt
    nsteps, k = 10, 8

    se = _build_advect(n, "euler");  se.advance(dt, nsteps)
    ss = _build_advect(n, "ssprk3"); ss.advance(dt, nsteps)
    sr = _build_advect(n, "ssprk3"); sr.advance(dt / k, nsteps * k)  # reference temps-convergee

    ref = np.asarray(sr.density())
    err_euler = float(np.abs(np.asarray(se.density()) - ref).mean())   # L1 (moyenne) vs reference
    err_ssprk3 = float(np.abs(np.asarray(ss.density()) - ref).mean())
    assert np.isfinite(err_euler) and np.isfinite(err_ssprk3), "erreurs L1 non finies"
    # erreur euler non triviale (sinon comparaison de bruit) ET ssprk3 plus precise (inegalite nette).
    assert err_euler > 1e-12, "erreur euler trop petite (%.2e) : dt insuffisant pour isoler le temps" % err_euler
    assert err_ssprk3 < err_euler, (
        "ssprk3 pas plus precis qu'euler (L1 ssprk3=%.3e, euler=%.3e, ratio=%.2f)"
        % (err_ssprk3, err_euler, err_ssprk3 / max(err_euler, 1e-300)))
    print("OK  (c) ordre : L1 temporelle ssprk3=%.3e < euler=%.3e (dt=%.3e, ratio=%.1fx)"
          % (err_ssprk3, err_euler, dt, err_euler / max(err_ssprk3, 1e-300)))


# --- (d) imex + ssprk3 : combinaison rejetee explicitement ---
def _check_imex_ssprk3_rejected(n=16):
    # ssprk3 (time.kind) et imex (time.kind) sont des selecteurs MUTUELLEMENT EXCLUSIFS (parite avec
    # System) : la facade standard ne peut PAS les combiner. On le prouve via le binding bas niveau --
    # demander le traitement IMEX de la source (masque implicite implicit_vars, OU rapport Newton
    # newton_diagnostics) AVEC time='ssprk3' est REJETE : ssprk3 = transport explicite, exclusif de
    # l'IMEX. (Le moteur subcycle_level_mp porte en plus une garde de defense en profondeur ssprk3+imex.)
    model = _scalar_charge(+1.0)

    def add(time, **kw):
        s = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
        kwargs = dict(implicit_vars=[], implicit_roles=[], newton_max_iters=2, newton_rel_tol=0.0,
                      newton_abs_tol=0.0, newton_fd_eps=1e-7, newton_damping=1.0,
                      newton_fail_policy="none", newton_diagnostics=False)
        kwargs.update(kw)
        s._s.add_block("b", model, "minmod", "rusanov", "conservative", time, 1, 1,
                       kwargs["implicit_vars"], kwargs["implicit_roles"], kwargs["newton_max_iters"],
                       kwargs["newton_rel_tol"], kwargs["newton_abs_tol"], kwargs["newton_fd_eps"],
                       kwargs["newton_damping"], kwargs["newton_fail_policy"],
                       kwargs["newton_diagnostics"])
        return s

    # ssprk3 SEUL accepte (transport explicite), imex SEUL accepte (source implicite) : chacun valide.
    add("ssprk3")
    add("imex")
    # ssprk3 + masque implicite (IMEX partiel) -> rejet.
    for kw in (dict(implicit_vars=["rho"]), dict(newton_diagnostics=True)):
        try:
            add("ssprk3", **kw)
        except Exception:
            pass
        else:
            raise AssertionError("imex+ssprk3 NON rejete (add_block ssprk3 + %r accepte)" % kw)
    print("OK  (d) imex + ssprk3 rejete : ssprk3 (transport explicite) exclusif du traitement IMEX")


# --- (e) loader .so + ssprk3 : rejet explicite (ABI plate ne transporte pas la methode) ---
def _check_native_loader_rejects_ssprk3(n=16):
    s = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
    # add_native_block valide time AVANT le dlopen : aucun .so reel requis pour observer le rejet.
    try:
        s._s.add_native_block("b", "/tmp/_adc_ssprk3_inexistant.so", "minmod", "rusanov",
                              "conservative", "ssprk3", 1.4, 1)
    except Exception as e:
        assert "ssprk3" in str(e), "rejet .so present mais message inattendu : %s" % e
    else:
        raise AssertionError("loader .so + ssprk3 NON rejete (option silencieusement ignoree)")
    # explicit reste accepte au stade validation (echoue plus loin faute de .so reel) : le rejet est
    # SPECIFIQUE a ssprk3, pas un refus generique de add_native_block. NB : chemin SANS 'ssprk3'
    # dans le nom -- dlopen echoie le chemin dans son message, ce qui piegerait l'assertion.
    try:
        s._s.add_native_block("b", "/tmp/_adc_loader_inexistant.so", "minmod", "rusanov",
                              "conservative", "explicit", 1.4, 1)
    except Exception as e:
        assert "ssprk3" not in str(e), "explicit rejete pour cause de ssprk3 (rejet trop large)"
    print("OK  (e) loader .so + ssprk3 rejete explicitement (ABI plate sans methode temporelle)")


def main():
    _check_mono()
    _check_multi()
    _check_default_bit_identical()
    _check_order()
    _check_imex_ssprk3_rejected()
    _check_native_loader_rejects_ssprk3()
    print("OK test_amr_ssprk3")


if __name__ == "__main__":
    main()
