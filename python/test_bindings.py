"""Teste le module Python `adc` (binding de la facade libadc).

Verifie qu'on pilote les solveurs concrets depuis Python sans rien savoir des
templates C++ : construction par config, pas de temps, invariants physiques
(masse conservee, quantite de mouvement nulle pour la gravite), et champ rendu
en tableau numpy de la bonne forme. PYTHONPATH pointe sur le dossier du .so.
"""
import sys
import numpy as np
import adc

fails = 0


def chk(cond, what):
    global fails
    if not cond:
        print("FAIL", what)
        fails += 1


# --- DiocotronSolver ---
cfg = adc.DiocotronConfig()
cfg.n = 64
ds = adc.DiocotronSolver(cfg)
m0 = ds.mass()
for _ in range(5):
    ds.step(0.01)
rho = ds.density()
print(f"DiocotronSolver : shape={rho.shape} masse {m0:.6e} -> {ds.mass():.6e}")
chk(isinstance(rho, np.ndarray) and rho.shape == (64, 64), "diocotron_density_numpy")
chk(abs(ds.mass() - m0) < 1e-9, "diocotron_masse_conservee")

# --- EulerPoissonSolver, backend FFT ---
ec = adc.EulerPoissonConfig()
ec.n = 64
ec.use_fft = True
es = adc.EulerPoissonSolver(ec)
em0 = es.mass()
for _ in range(5):
    es.step(0.004)
print(f"EulerPoissonSolver(FFT) : masse={es.mass():.6e} "
      f"p=({es.total_momentum(0):.2e}, {es.total_momentum(1):.2e})")
chk(abs(es.mass() - em0) < 1e-9, "ep_masse_conservee")
chk(abs(es.total_momentum(0)) < 1e-9, "ep_qte_mouvement_nulle")
chk(es.density().shape == (64, 64), "ep_density_numpy")

# --- DiocotronSolver, CI bande + pas auto CFL ---
bc = adc.DiocotronConfig()
bc.n = 48
bc.ic = adc.DiocotronIC.Band
db = adc.DiocotronSolver(bc)
bm0 = db.mass()
for _ in range(10):
    db.step_cfl(0.4)  # pas stable choisi par la facade (derive E x B)
print(f"DiocotronSolver(Band) : v_derive={db.max_drift_speed():.3e} "
      f"phi.shape={db.potential().shape} dmasse={abs(db.mass() - bm0):.2e}")
chk(db.potential().shape == (48, 48), "diocotron_potential_numpy")
chk(abs(db.mass() - bm0) < 1e-9, "diocotron_band_masse_conservee")

# --- TwoFluidAPSolver, regime raide (AP) ---
tc = adc.TwoFluidAPConfig()
tc.n = 64
tc.omega_pe = 1e3
tc.omega_pi = 20.0
ts = adc.TwoFluidAPSolver(tc)
tm0 = ts.mass_e()
ts.advance(5.0 / 1e3, 200)  # dt*omega_pe = 5 : explicite exploserait
print(f"TwoFluidAPSolver(raide) : max|dne|={ts.max_dev():.3e} "
      f"max|charge|={ts.max_charge():.3e} dmasse_e={abs(ts.mass_e() - tm0):.2e}")
chk(ts.density_e().shape == (64, 64), "tfap_density_numpy")
chk(ts.max_dev() < 0.1, "tfap_AP_borne")
chk(ts.max_charge() < 0.1, "tfap_quasi_neutre")
chk(abs(ts.mass_e() - tm0) < 1e-7, "tfap_masse_conservee")

if fails == 0:
    print("OK test_bindings")
sys.exit(0 if fails == 0 else 1)
