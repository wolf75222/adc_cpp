"""Genere un harnais CUDA : la brique EulerGen GENEREE tourne dans un kernel __global__ sur le GPU,
et on compare son flux a pops::Euler calcule sur l'hote. Placeholder __BRICK__ (pas de % pour ne pas
heurter les printf)."""
import sys
sys.path.insert(0, "python/tests")
from test_dsl_brick import build_euler_brick

brick = build_euler_brick().emit_cpp_brick(name="EulerGen")  # CSE active par defaut

HARNESS = r"""// harnais CUDA : brique generee sur GPU (H100) vs pops::Euler sur hote.
#include <pops/physics/fluids/euler.hpp>
__BRICK__
#include <cstdio>
#include <cmath>

__global__ void kflux(const double* U, double* Fx, double* Fy, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  pops::StateVec<4> u{};
  for (int i = 0; i < 4; ++i) u[i] = U[t * 4 + i];
  pops::Aux a{};
  pops_generated::EulerGen gen;          // brique GENEREE, executee sur le device
  auto fx = gen.flux(u, a, 0);
  auto fy = gen.flux(u, a, 1);
  for (int i = 0; i < 4; ++i) { Fx[t * 4 + i] = fx[i]; Fy[t * 4 + i] = fy[i]; }
}

int main() {
  const int n = 4;
  double hU[n * 4] = {1.0, 0.2, -0.1, 2.5,  2.0, 0.5, 0.3, 6.0,
                      0.5, -0.2, 0.1, 1.8,  1.5, 0.0, 0.0, 3.0};
  double *dU, *dFx, *dFy;
  cudaMalloc(&dU, sizeof(hU)); cudaMalloc(&dFx, sizeof(hU)); cudaMalloc(&dFy, sizeof(hU));
  cudaMemcpy(dU, hU, sizeof(hU), cudaMemcpyHostToDevice);
  kflux<<<1, n>>>(dU, dFx, dFy, n);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA err: %s\n", cudaGetErrorString(e)); return 2; }
  double hFx[n * 4], hFy[n * 4];
  cudaMemcpy(hFx, dFx, sizeof(hU), cudaMemcpyDeviceToHost);
  cudaMemcpy(hFy, dFy, sizeof(hU), cudaMemcpyDeviceToHost);

  pops::Euler ref; ref.gamma = 1.4; pops::Aux a{};
  double md = 0.0;
  for (int t = 0; t < n; ++t) {
    pops::StateVec<4> u{};
    for (int i = 0; i < 4; ++i) u[i] = hU[t * 4 + i];
    auto rx = ref.flux(u, a, 0); auto ry = ref.flux(u, a, 1);
    for (int i = 0; i < 4; ++i) {
      md = fmax(md, fabs(rx[i] - hFx[t * 4 + i]));
      md = fmax(md, fabs(ry[i] - hFy[t * 4 + i]));
    }
  }
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("device=%s  maxdiff(GPU EulerGen vs hote pops::Euler)=%.3e\n", p.name, md);
  return md < 1e-12 ? 0 : 1;
}
"""

open("/tmp/euler_gpu.cu", "w").write(HARNESS.replace("__BRICK__", brick))
print("euler_gpu.cu genere :", len(HARNESS) + len(brick), "octets")
