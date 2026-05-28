// Preuve de portabilite GPU : un pas d'advection E x B (flux de Rusanov,
// vitesse constante) execute dans un kernel CUDA, en reutilisant TELS QUELS les
// vues Array4 / ConstArray4 de la couche maillage. C'est le pendant Grace
// Hopper du seam for_each_cell : meme layout (SoA composante-lente), meme
// indexation a(i, j, c), annotee ADC_HD (host+device). On valide le resultat
// GPU contre une reference CPU (meme formule), erreur ~ arrondi.
//
// Compilation (sur un noeud armgpu de ROMEO, cf. scripts/romeo_gpu.sbatch) :
//   nvcc -O2 -std=c++17 -I include examples/gpu/advect_cuda.cu -o advect_cuda
// Le seul changement cote bibliotheque : Array4::operator() porte ADC_HD.

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

using namespace adc;

// flux de Rusanov (= upwind) pour l'advection lineaire a vitesse v.
ADC_HD inline double rus(double uL, double uR, double v) {
  return 0.5 * v * (uL + uR) - 0.5 * fabs(v) * (uR - uL);
}

// un pas explicite par cellule valide : un = u - dt div F.
__global__ void advect_kernel(ConstArray4 u, Array4 un, int ilo, int ihi,
                              int jlo, int jhi, double vx, double vy, double dx,
                              double dy, double dt) {
  const int i = ilo + blockIdx.x * blockDim.x + threadIdx.x;
  const int j = jlo + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > ihi || j > jhi) return;
  const double fxL = rus(u(i - 1, j), u(i, j), vx);
  const double fxR = rus(u(i, j), u(i + 1, j), vx);
  const double fyB = rus(u(i, j - 1), u(i, j), vy);
  const double fyT = rus(u(i, j), u(i, j + 1), vy);
  un(i, j) = u(i, j) - dt * ((fxR - fxL) / dx + (fyT - fyB) / dy);
}

static void ck(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::printf("CUDA FAIL %s : %s\n", w, cudaGetErrorString(e));
    std::exit(2);
  }
}

int main() {
  const int N = 256, ng = 1;
  const double dx = 1.0 / N, dy = 1.0 / N, dt = 0.3 * dx;
  const double vx = 1.0, vy = 0.3;  // derive E x B constante
  Box2D dom = Box2D::from_extents(N, N);

  // champ initial + ghosts periodiques (sur l'hote).
  Fab2D U(dom, 1, ng);
  auto u0 = [&](int i, int j) {
    constexpr double kPi = 3.14159265358979323846;
    return 1.0 + 0.5 * std::sin(2 * kPi * (i + 0.5) / N) *
                     std::sin(2 * kPi * (j + 0.5) / N);
  };
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) U(i, j) = u0(i, j);
  auto wrap = [&](int x) { return (x % N + N) % N; };
  for (int j = -ng; j < N + ng; ++j)
    for (int i = -ng; i < N + ng; ++i)
      if (i < 0 || i >= N || j < 0 || j >= N) U(i, j) = u0(wrap(i), wrap(j));

  // reference CPU (meme formule que le kernel).
  std::vector<double> ref(static_cast<std::size_t>(N) * N);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) {
      const double fxL = rus(U(i - 1, j), U(i, j), vx);
      const double fxR = rus(U(i, j), U(i + 1, j), vx);
      const double fyB = rus(U(i, j - 1), U(i, j), vy);
      const double fyT = rus(U(i, j), U(i, j + 1), vy);
      ref[j * N + i] = U(i, j) - dt * ((fxR - fxL) / dx + (fyT - fyB) / dy);
    }

  // copie H2D : meme buffer (layout SoA composante-lente identique).
  const ConstArray4 hv = U.const_array();
  const long nbytes = U.size() * static_cast<long>(sizeof(Real));
  Real *d_u = nullptr, *d_un = nullptr;
  ck(cudaMalloc(&d_u, nbytes), "malloc u");
  ck(cudaMalloc(&d_un, nbytes), "malloc un");
  ck(cudaMemcpy(d_u, U.data(), nbytes, cudaMemcpyHostToDevice), "H2D");
  ck(cudaMemset(d_un, 0, nbytes), "memset");

  // vues device : memes strides/offsets que l'hote, pointeur device.
  ConstArray4 u_dev{d_u, hv.nx_tot, hv.comp_stride, hv.ig0, hv.jg0};
  Array4 un_dev{d_un, hv.nx_tot, hv.comp_stride, hv.ig0, hv.jg0};

  dim3 blk(16, 16);
  dim3 grd((N + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y);
  advect_kernel<<<grd, blk>>>(u_dev, un_dev, 0, N - 1, 0, N - 1, vx, vy, dx, dy,
                              dt);
  ck(cudaGetLastError(), "launch");
  ck(cudaDeviceSynchronize(), "sync");

  // D2H + comparaison.
  std::vector<Real> host(U.size());
  ck(cudaMemcpy(host.data(), d_un, nbytes, cudaMemcpyDeviceToHost), "D2H");
  Array4 hview{host.data(), hv.nx_tot, hv.comp_stride, hv.ig0, hv.jg0};
  double maxdiff = 0;
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(hview(i, j) - ref[j * N + i]));

  cudaFree(d_u);
  cudaFree(d_un);

  int dev = 0;
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  std::printf("GPU=%s  N=%d  maxdiff(GPU vs CPU)=%.3e\n", prop.name, N, maxdiff);
  if (maxdiff < 1e-12) {
    std::printf("OK advect_cuda\n");
    return 0;
  }
  std::printf("FAIL advect_cuda\n");
  return 1;
}
