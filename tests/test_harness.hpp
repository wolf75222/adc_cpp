#pragma once

/// @file
/// @brief Helpers partages des tests autonomes du coeur (assertions, tolerances, sommes de controle).
///
/// Couche : `tests/` (hors `include/adc`, reserve aux executables de test ; non installe).
/// Role : factoriser les briques copiees a l'identique dans des dizaines de tests -- le compteur
///   d'echecs + la fonction d'assertion `chk`, le predicat d'exception `raises`, la comparaison
///   relative `close_rel`, la somme de controle `checksum`, la constante `kPi`.
/// Contrat : header-only, zero dependance externe. ON NE REMPLACE PAS la philosophie "chaque test =
///   un main() autonome qui renvoie 0/1" : ce n'est PAS un framework (ni gtest ni catch). Chaque
///   main() garde sa structure (un compteur local + des appels d'assertion + un retour final) ;
///   ce header ne fournit que les briques, sans macro magique ni enregistrement global.
///
/// Invariants :
/// - aucune variable globale, aucun etat de processus : tout passe par le compteur fourni par
///   l'appelant (`pops::test::Checker` encapsule le compteur ou l'appelant garde son `int fails`) ;
/// - les fonctions n'impriment que sur stdout/stderr et ne touchent ni MPI ni Kokkos (un test MPI
///   garde son propre `long fails` reduit par all_reduce a la main, cf. tests MPI) ;
/// - `close_rel` et `checksum` sont des copies bit-a-bit des versions locales remplacees, pour que
///   le comportement (donc la sortie et le code retour) des tests migres reste STRICTEMENT identique.
///
/// Adoption INCREMENTALE : ce header est introduit avec une premiere vague de tests migres
///   (test_box2d, test_dense_eig, test_elliptic_operator, test_amr_hierarchy, test_amr_system_contract,
///   test_amr_regrid_mpi_parity). Le reste du dossier `tests/` garde encore ses copies locales de
///   `chk` / `close_rel` / `raises` / `checksum` : c'est de la dette CONNUE, a resorber au fil des
///   prochaines passes en remplacant ces lambdas par les briques d'ici. Aucun changement de
///   comportement attendu a chaque migration (memes assertions, meme sortie, meme code retour).

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace pops::test {

/// Pi en double precision (copie de la constante `kPi` / `pi` dupliquee dans les tests).
inline constexpr double kPi = 3.14159265358979323846;

/// Compteur d'echecs encapsule + assertion.
///
/// Usage : `Checker chk;` puis `chk(cond, "libelle");` dans le corps du main(), et
///   `return chk.failed();` a la fin. Remplace le couple idiomatique `int fails = 0;` + lambda
///   `auto chk = [&](bool c, const char* w){...}` recopie dans presque chaque test.
/// Contrat : deux styles d'impression, choisis a la construction, pour reproduire EXACTEMENT la
///   sortie des deux familles de tests existantes :
///   - `Style::Terse`   (defaut) : n'imprime QUE les echecs, `FAIL <libelle>\n` (cf. test_box2d) ;
///   - `Style::Verbose` : imprime chaque ligne `  [OK ] <libelle>` / `  [XX ] <libelle>`
///     (cf. test_dense_eig, test_amr_system_contract).
/// Contraintes : non copiable par valeur n'est pas requis ; on le capture par reference dans une
///   lambda `chk` si le main() prefere garder cette forme (cf. exemple ci-dessous).
class Checker {
 public:
  enum class Style { Terse, Verbose };

  explicit Checker(Style style = Style::Terse) : style_(style) {}

  /// Verifie @p cond ; incremente le compteur et imprime un diagnostic si @p cond est faux.
  /// @return @p cond (permet `if (!chk(...)) {...}` si l'appelant veut court-circuiter).
  bool operator()(bool cond, const char* label) {
    if (style_ == Style::Verbose) {
      std::printf("  [%s] %s\n", cond ? "OK " : "XX ", label);
    } else if (!cond) {
      std::printf("FAIL %s\n", label);
    }
    if (!cond)
      ++fails_;
    return cond;
  }

  /// Nombre d'echecs accumules.
  int fails() const { return fails_; }

  /// Code retour conventionnel : 0 si aucun echec, 1 sinon (a renvoyer depuis main()).
  int failed() const { return fails_ == 0 ? 0 : 1; }

 private:
  Style style_;
  int fails_ = 0;
};

/// Renvoie true si l'appel @p f leve un `std::runtime_error` (le refus attendu d'un contrat).
///
/// Copie de la lambda `raises` dupliquee dans les tests de contrat (test_amr_system_contract, ...).
/// Une exception d'un autre type, ou aucune exception, renvoie false.
template <class F>
bool raises(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

/// Comparaison a tolerance relative + absolue : |a - b| <= rtol * max(|a|, |b|) + atol.
///
/// Copie bit-a-bit de la version locale de test_dense_eig (atol par defaut 1e-12). @p T est le type
/// flottant (`pops::Real` ou `double`) ; deduit a l'appel.
template <class T>
bool close_rel(T a, T b, T rtol, T atol = T(1e-12)) {
  const T d = std::fabs(a - b);
  const T s = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
  return d <= rtol * s + atol;
}

/// Somme de controle = somme des carres des elements (signature deterministe d'un champ).
///
/// Copie de la lambda `checksum` dupliquee dans les tests de parite MPI/AMR (test_amr_regrid_mpi_parity,
/// test_mpi_amr_*_parity). On somme x*x : invariant au signe, sensible a toute divergence numerique.
inline double checksum(const std::vector<double>& v) {
  double s = 0;
  for (double x : v)
    s += x * x;
  return s;
}

}  // namespace pops::test
