#pragma once

/// @file
/// @brief Helpers partages des harnais de bench/profilage (chronometrage, percentiles, argparse).
///
/// Couche : `bench/` (hors `include/adc` ; harnais de MESURE, hors du build par defaut, jamais en CI).
/// Role : factoriser les briques copiees a l'identique dans profile_step / frontend_cpp /
///   profile_transport_mbox / scaling_step / scaling_amr -- le chronometre `timed` (avec fences
///   device autour de la phase), le `percentile` interpole, l'accumulateur `PhaseTimers` par phase,
///   et la lambda d'argparse `eat`.
/// Contrat : header-only, zero dependance externe HORMIS `adc/core/kokkos_env.hpp` pour `device_fence()`
///   (deja tire transitivement par tous les harnais : c'est le seam qui porte le backend Kokkos/serie).
///   Les harnais restent des main() autonomes ; ce header ne fournit que les briques de mesure.
///
/// Invariants :
/// - `timed` encadre la phase de `device_fence()` AVANT et APRES : sous Kokkos Cuda les kernels sont
///   async, sans fence on ne mesurerait que le temps de SOUMISSION ; en serie/OpenMP c'est un no-op ;
/// - aucune mesure ici n'est une collective MPI : l'agregation cross-rang (all_reduce_max sur les
///   temps, percentiles reduits) reste a la charge de chaque harnais, hors de ce header ;
/// - copies bit-a-bit des versions locales remplacees -> le comportement (sortie chronometree, JSON)
///   des harnais migres reste identique.
///
/// Adoption INCREMENTALE : ce header est introduit avec une premiere vague de harnais migres
///   (profile_step, frontend_cpp). Les autres harnais de `bench/` (scaling_step, scaling_amr,
///   profile_transport_mbox) gardent encore leurs copies locales de `timed` / `percentile` / `eat` :
///   dette CONNUE, a resorber aux prochaines passes en pointant vers les briques d'ici. Aucun
///   changement de comportement attendu a chaque migration (meme sortie chronometree, meme JSON).

#include <pops/core/foundation/kokkos_env.hpp>  // device_fence (seam backend : no-op serie/OpenMP, fence sous Cuda)

#include <algorithm>
#include <chrono>
#include <cstdlib>  // std::atof, std::atoi (conversion des arguments par eat)
#include <cstring>  // std::strcmp (comparaison de cle par eat)
#include <string>   // std::string (cas chaine de eat)
#include <type_traits>
#include <vector>

namespace pops::bench {

/// Horloge monotone des harnais (insensible aux ajustements d'horloge murale).
using Clock = std::chrono::steady_clock;

/// Chronometre une phase et renvoie sa duree en SECONDES.
///
/// device_fence() AVANT et APRES @p f pour capturer l'execution device reelle (cf. invariant du
/// fichier). Copie de la fonction `timed` dupliquee dans profile_step / frontend_cpp /
/// profile_transport_mbox. @p F est un appelable sans argument.
template <class F>
double timed(F&& f) {
  device_fence();
  const auto t0 = Clock::now();
  f();
  device_fence();
  const auto t1 = Clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

/// Percentile interpole (lineaire) du quantile @p q in [0,1] de @p v.
///
/// Copie bit-a-bit de la version dupliquee dans frontend_cpp / scaling_step / scaling_amr : trie une
/// COPIE de @p v (passee par valeur a dessein), interpole entre les deux rangs encadrants. Renvoie 0
/// sur un echantillon vide. @p q n'est pas borne ici (les harnais passent 0.10 / 0.5 / 0.90).
inline double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = q * (v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

/// Accumulateur de temps (secondes) par phase d'un pas instrumente.
///
/// Phases du pas representatif de System::step telles que mesurees par profile_step / frontend_cpp :
/// poisson, derivation aux, halos, transport, reduction, fence isolee, (re)allocation des temporaires.
/// `add` agrege les pas chronometres ; `total` somme toutes les phases (utile au rapport par phase).
/// Copie de la struct `PhaseTimers` partagee par profile_step (avec total()) et frontend_cpp.
struct PhaseTimers {
  double poisson = 0, aux_derive = 0, halos = 0, transport = 0, reduction = 0, fence = 0,
         alloc_tmp = 0;
  void add(const PhaseTimers& o) {
    poisson += o.poisson;
    aux_derive += o.aux_derive;
    halos += o.halos;
    transport += o.transport;
    reduction += o.reduction;
    fence += o.fence;
    alloc_tmp += o.alloc_tmp;
  }
  double total() const {
    return poisson + aux_derive + halos + transport + reduction + fence + alloc_tmp;
  }
};

/// Consomme un argument `--cle valeur` de la ligne de commande.
///
/// A appeler dans la boucle `for (int a = 1; a < argc; ++a)` du main() : si `argv[@p a]` vaut @p key
/// et qu'une valeur suit, ecrit la valeur convertie dans @p out (selon son type : std::string brute,
/// double via atof, sinon int via atoi), avance @p a au-dela de la valeur, et renvoie true.
/// Copie de la lambda `eat` dupliquee dans profile_step / frontend_cpp / scaling_step / scaling_amr,
/// transformee en fonction libre prenant l'indice @p a par reference (la lambda capturait a/argc/argv).
/// @return true si @p key a ete consommee (l'appelant fait alors `continue`).
template <class T>
bool eat(int argc, char** argv, int& a, const char* key, T& out) {
  if (std::strcmp(argv[a], key) == 0 && a + 1 < argc) {
    if constexpr (std::is_same_v<T, std::string>)
      out = argv[++a];
    else if constexpr (std::is_same_v<T, double>)
      out = std::atof(argv[++a]);
    else
      out = std::atoi(argv[++a]);
    return true;
  }
  return false;
}

}  // namespace pops::bench
