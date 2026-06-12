// Compare deux snap_*.raw du driver diocotron (header 3 doubles {n,t,k}, puis
// 15*n*n doubles d'etat + n*n doubles de potentiel). Rend max |a-b|, max rel,
// nombre de doubles bit-identiques, et l'ecart d'horloge t.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
static std::vector<double> load(const char* p){
  std::ifstream f(p, std::ios::binary);
  if(!f){ std::fprintf(stderr,"introuvable: %s\n",p); std::exit(2); }
  f.seekg(0,std::ios::end); auto bytes=f.tellg(); f.seekg(0);
  std::vector<double> v(bytes/sizeof(double));
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  return v;
}
int main(int argc,char**argv){
  if(argc<3){ std::fprintf(stderr,"usage: %s a.raw b.raw\n",argv[0]); return 1; }
  auto a=load(argv[1]), b=load(argv[2]);
  if(a.size()!=b.size()){ std::fprintf(stderr,"tailles differentes: %zu vs %zu\n",a.size(),b.size()); return 3; }
  // header
  std::printf("n_a=%.0f t_a=%.17g k_a=%.0f\n", a[0],a[1],a[2]);
  std::printf("n_b=%.0f t_b=%.17g k_b=%.0f\n", b[0],b[1],b[2]);
  std::printf("dt_clock (|t_a-t_b|) = %.3e\n", std::fabs(a[1]-b[1]));
  double maxabs=0, maxrel=0; long nident=0, ntot=0; long iabs=-1;
  for(size_t i=3;i<a.size();++i){
    ++ntot;
    double d=std::fabs(a[i]-b[i]);
    if(a[i]==b[i]) ++nident;
    if(d>maxabs){ maxabs=d; iabs=(long)i; }
    double den=std::fmax(std::fabs(a[i]),std::fabs(b[i]));
    if(den>0){ double r=d/den; if(r>maxrel) maxrel=r; }
  }
  std::printf("payload doubles      = %ld\n", ntot);
  std::printf("bit-identiques       = %ld (%.4f%%)\n", nident, 100.0*nident/ntot);
  std::printf("max |a-b|            = %.6e (index %ld)\n", maxabs, iabs);
  std::printf("max rel              = %.6e\n", maxrel);
  std::printf(maxabs==0.0 ? "VERDICT: BIT-IDENTIQUE\n" : "VERDICT: ECART (voir chiffres)\n");
  return 0;
}
