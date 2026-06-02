#pragma once

#include <memory>
#include <string>
#include <vector>

// Composition MULTI-ESPECES a l'EXECUTION : le « systeme / coupleur » du tuteur.
//
// On COMPOSE un systeme bloc par bloc depuis Python : chaque bloc (= une espece /
// un U) porte SON modele physique, SON schema spatial (limiteur + flux), SON
// traitement temporel (explicite / IMEX) et SON nombre de sous-pas. Tous les blocs
// partagent un meme Poisson : le second membre elliptique est la SOMME des
// contributions de chaque modele, f = Sum_s elliptic_rhs_s(u_s) -- c'est la ou les
// especes interagissent (RHS de la partie elliptique), exactement comme decrit. La
// source S (force electrostatique, couplages) agit en plus, par bloc.
//
// Python dit QUOI assembler ; tout le calcul cellule par cellule (assemble_rhs<L,F>,
// Newton local de la source implicite, multigrille / FFT de Poisson) reste en C++
// compile et fige a l'ajout du bloc. La LISTE de blocs est dynamique, jamais le
// noyau : aucun callback Python dans le hot path -- SAUF si l'utilisateur fournit
// LUI-MEME un integrateur temporel en Python, via les primitives eval_rhs / get_state
// / set_state (Python par PAS, le C++ reste par CELLULE).
//
// Le second membre de Poisson est la SOMME des contributions de chaque modele,
// f = Sum_s elliptic_rhs_s(u_s) : c'est LA ou les especes interagissent (RHS de la
// partie elliptique). Selon le modele, elliptic_rhs vaut q*n (fluides charges),
// alpha*(n - n_i0) (diocotron) ou s*4piG*(rho - rho0) (Euler-Poisson).
//
// Modeles (tag) :
//   "diocotron"       : derive E x B, 1 variable, fond neutralisant n_i0 ;
//   "electron_euler"  : Euler complet + force electrostatique, 4 variables ;
//   "ion_isothermal"  : Euler isotherme + force electrostatique, 3 variables ;
//   "euler_poisson"   : Euler complet + champ self-consistent (charge = signe du
//                       couplage : +1 auto-gravite, -1 electrostatique/Langmuir),
//                       4 variables, fond rho0, intensite four_pi_G.

namespace adc {

struct SystemConfig {
  int n = 64;             // cellules par direction
  double L = 1.0;         // taille du domaine [0,L]^2
  double B0 = 1.0;        // champ magnetique (derive E x B, modele "diocotron")
  double n_i0 = 0.0;      // fond ionique neutralisant (modele "diocotron")
  double alpha = 1.0;     // constante de couplage Poisson (modele "diocotron")
  double gamma = 1.4;     // adiabatique (modeles "electron_euler" / "euler_poisson")
  double cs2 = 0.5;       // vitesse du son^2 isotherme (modele "ion_isothermal")
  double four_pi_G = 1.0; // intensite du couplage (modele "euler_poisson")
  double rho0 = 1.0;      // fond neutralisant (modele "euler_poisson", solvabilite)
  bool periodic = true;
};

class System {
 public:
  explicit System(const SystemConfig& cfg);
  ~System();
  System(System&&) noexcept;
  System& operator=(System&&) noexcept;

  // --- Composition : un BLOC d'equation (= une espece) ---------------------
  //   model    : "diocotron" | "electron_euler" | "ion_isothermal"
  //   charge   : signe pour le couplage (q dans elliptic_rhs des fluides charges)
  //   limiter  : reconstruction spatiale "none" | "minmod" | "vanleer"
  //   flux     : flux numerique "rusanov" | "hllc" (hllc exige un modele Euler complet)
  //   time     : "explicit" (SSPRK2) | "imex" (transport explicite + source implicite)
  //   substeps : sous-pas par bloc (ex. 10 electrons : 1 ion). Chaque bloc son schema.
  void add_block(const std::string& name, const std::string& model, double charge,
                 const std::string& limiter = "minmod",
                 const std::string& flux = "rusanov",
                 const std::string& time = "explicit", int substeps = 1);
  // Raccourci historique : minmod + rusanov + explicite + 1 sous-pas.
  void add_species(const std::string& name, const std::string& model, double charge);

  // --- Poisson de systeme : solveur + conditions aux limites + paroi -------
  //   solver : "geometric_mg" (tout cas, paroi comprise) | "fft" (periodique, n=2^k)
  //   bc     : "auto" (periodique si cfg.periodic sinon Dirichlet) | "periodic"
  //            | "dirichlet" (phi=0 au bord) | "neumann" (Foextrap)
  //   wall   : "none" | "circle" -> paroi conductrice circulaire (Dirichlet sur le
  //            cercle, masque embedded), centre (L/2,L/2), rayon wall_radius.
  //   rhs    : "charge_density" -> f = Sum_s q_s n_s (densite de charge ; pour le
  //            diocotron a fond nul c'est exactement alpha*n du modele). Seul mode.
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  // Densite (composante 0) d'une espece, tableau n*n row-major ; les autres
  // composantes (qte de mvt, energie) posees a l'equilibre au repos.
  void set_density(const std::string& name, const std::vector<double>& rho);

  // --- Avancee en temps ----------------------------------------------------
  void solve_fields();                 // resout Poisson (Sum elliptic_rhs) + aux = grad phi
  void step(double dt);                // solve_fields puis avance chaque bloc (son schema)
  void advance(double dt, int nsteps);
  double step_cfl(double cfl);         // dt = cfl * h / w_max(systeme), puis step ; rend dt

  // --- Primitives pour un integrateur temporel CUSTOM en Python ------------
  // Permettent d'ecrire son propre take_step cote Python (par PAS), le calcul
  // cellule par cellule (residu, Poisson) restant en C++ :
  //   solve_fields() ; R = eval_rhs(name) ; U = get_state(name) ; ... ; set_state(name, U)
  std::vector<double> eval_rhs(const std::string& name);   // -div F + S, taille ncomp*n*n
  std::vector<double> get_state(const std::string& name);  // U, taille ncomp*n*n (comp-major)
  void set_state(const std::string& name, const std::vector<double>& u);
  int n_vars(const std::string& name) const;

  // --- Diagnostics ---------------------------------------------------------
  int nx() const;
  double time() const;
  int n_species() const;
  double mass(const std::string& name) const;
  std::vector<double> density(const std::string& name) const;  // n*n row-major
  std::vector<double> potential();  // phi, n*n row-major (construit Poisson au besoin)

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
