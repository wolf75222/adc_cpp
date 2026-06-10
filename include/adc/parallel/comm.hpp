#pragma once

// Le seam parallele. Sans ADC_HAS_MPI : rang unique (serie). Avec ADC_HAS_MPI :
// MPI_Comm_rank/size + collectives sur MPI_COMM_WORLD. Tout le reste du code
// passe par my_rank() / n_ranks() / all_reduce_* et ignore le backend.
//
// Robustesse : meme compile avec MPI, si MPI n'est pas initialise (ex. un test
// serie linke contre adc), my_rank() rend 0 et n_ranks() rend 1. Il faut donc
// appeler comm_init() au debut de main() pour les runs reellement distribues.

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

namespace adc {

#ifdef ADC_HAS_MPI

inline bool comm_active() {
  int inited = 0, fin = 0;
  MPI_Initialized(&inited);
  MPI_Finalized(&fin);
  return inited && !fin;
}

inline void comm_init(int* argc = nullptr, char*** argv = nullptr) {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) MPI_Init(argc, argv);
}

inline void comm_finalize() {
  int fin = 0;
  MPI_Finalized(&fin);
  if (!fin) MPI_Finalize();
}

inline int my_rank() {
  if (!comm_active()) return 0;
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

inline int n_ranks() {
  if (!comm_active()) return 1;
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

inline void barrier() {
  if (comm_active()) MPI_Barrier(MPI_COMM_WORLD);
}

inline double all_reduce_sum(double x) {
  if (!comm_active()) return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return r;
}

inline double all_reduce_max(double x) {
  if (!comm_active()) return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return r;
}

// Min global (pendant de all_reduce_max). Brique des bornes GLOBALES de pas de temps
// (System::add_dt_bound) : la callback hote est evaluee PAR RANG, le min global garantit un dt
// IDENTIQUE sur tous les rangs (sans quoi les collectifs du pas -- Krylov, fill_boundary --
// divergeraient -> deadlock). En serie : identite.
inline double all_reduce_min(double x) {
  if (!comm_active()) return x;
  double r = x;
  MPI_Allreduce(&x, &r, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return r;
}

// Somme element par element d'un tableau, en place, sur tous les rangs. Brique de
// base du reflux AMR multi-patch distribue : chaque rang remplit les contributions de
// ses patchs locaux (0 ailleurs), all-reduce -> chaque rang a le registre complet.
inline void all_reduce_sum_inplace(double* buf, int n) {
  if (!comm_active() || n <= 0) return;
  MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

inline long all_reduce_sum(long x) {
  if (!comm_active()) return x;
  long r = x;
  MPI_Allreduce(&x, &r, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return r;
}

// OU logique element par element d'un tableau de marqueurs (0/1), en place, sur tous les rangs.
// Brique du regrid AMR a GROSSIER REPARTI : chaque rang ne tague QUE ses boites grossieres
// locales (tag_cells iterant sur local_size()), donc personne ne voit la grille de tags complete ;
// l'OU global rassemble les tags sur chaque rang avant le clustering Berger-Rigoutsos, qui produit
// alors des patchs fins IDENTIQUES partout (sinon la BoxArray fine differerait par rang -> MPI
// desynchronise). Cf. la note de tag_box.hpp ("les tags repartis seront rassembles avant clustering").
inline void all_reduce_or_inplace(char* buf, int n) {
  if (!comm_active() || n <= 0) return;
  MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_CHAR, MPI_BOR, MPI_COMM_WORLD);
}

#else  // ----- serie -----

inline bool comm_active() { return false; }
inline void comm_init(int* = nullptr, char*** = nullptr) {}
inline void comm_finalize() {}
inline int my_rank() { return 0; }
inline int n_ranks() { return 1; }
inline void barrier() {}
inline double all_reduce_sum(double x) { return x; }
inline double all_reduce_max(double x) { return x; }
inline double all_reduce_min(double x) { return x; }
inline long all_reduce_sum(long x) { return x; }
inline void all_reduce_sum_inplace(double*, int) {}  // serie : identite
inline void all_reduce_or_inplace(char*, int) {}     // serie : identite

#endif

}  // namespace adc
