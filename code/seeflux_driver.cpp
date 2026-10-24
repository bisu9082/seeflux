// seeflux driver: integral neutron flux from PARMA4 (Sato 2015/2016)
// input lines (stdin): tag mode a b c d e f g
//  mode D: year month day lat lon alt_km g      (s,r,d from PARMA tables)
//  mode S: W-index rc(GV) depth(g/cm2) g          (direct)
//  mode A: W-index rc(GV) alt_km g               (depth from US Std Atm 1976, glat=100)
// output: tag W-index(s) rc depth g F_gt10 F_gt1 F_1to10 F_total  [n/cm2/s], log-trapezoid integration
// optional trailing groups "WB sigma_sat E_th W s" (4-parameter Weibull cross section, cm2/device, MeV):
//   each appends R = integral phi(E) sigma(E) dE over E_th..1e5 MeV  [upsets/s/device]
// optional "WC sigma_sat E_th W s E_cap": same, with sigma held constant at sigma(E_cap) above E_cap (sensitivity)
// optional "WBP ..." and "WCP ...": as WB and WC, folded with the PARMA proton spectrum (particle index 1) instead
//   of neutrons; the neutron Weibull cross section is applied to protons (Step 8 round 6)
// optional "PR": appends the proton flux above 10 MeV (PARMA particle index 1) [/cm2/s]
// optional "WR sigma_sat E_th W s E1 E2 ip": Weibull fold over [max(E_th,E1), E2] for particle ip (0 n, 1 p);
//   with sigma_sat = 0 the flux in [E1, E2] is returned (Step 8 round 7: energy bands, truncated test spectra)
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib>
using namespace std;
double getHPcpp(int,int,int);
double getrcpp(double,double);
double getdcpp(double,double);
double getSpecCpp(int,double,double,double,double,double);
double integ(double s,double r,double d,double g,double e1,double e2,int ip=0){
 static int n=0; if(!n){const char* v=getenv("SEEFLUX_NINT"); n=v?atoi(v):4000;} double l1=log(e1),l2=log(e2),h=(l2-l1)/n,sum=0;
 for(int i=0;i<=n;i++){double e=exp(l1+i*h); double f=getSpecCpp(ip,s,r,d,e,g)*e; sum+=(i==0||i==n)?0.5*f:f;}
 return sum*h;}
double fold(double s,double r,double d,double g,double sig,double eth,double w,double k,double ecap=1.0e5,int ip=0,double emin=0.0,double emax=1.0e5){
 const int n=4000; double lo=(emin>eth?emin:eth*(1.0+1e-9)); if(lo>=emax) return 0.0; double l1=log(lo),l2=log(emax),h=(l2-l1)/n,sum=0;
 for(int i=0;i<=n;i++){double e=exp(l1+i*h); double ex=(e<ecap?e:ecap); double xs=(sig>0)?sig*(1.0-exp(-pow((ex-eth)/w,k))):1.0;
  double f=getSpecCpp(ip,s,r,d,e,g)*xs*e; sum+=(i==0||i==n)?0.5*f:f;}
 return sum*h;}
int main(){ string line; cout.precision(6);
 while(getline(cin,line)){ if(line.empty()||line[0]=='#')continue; istringstream is(line); string tag,mode; is>>tag>>mode;
  double s,r,d,g;
  if(mode=="D"){int y,m,dd; double lat,lon,alt; is>>y>>m>>dd>>lat>>lon>>alt>>g; s=getHPcpp(y,m,dd); r=getrcpp(lat,lon); d=getdcpp(alt,lat);}
  else if(mode=="A"){double alt; is>>s>>r>>alt>>g; d=getdcpp(alt,100.0);}
  else {is>>s>>r>>d>>g;}
  const double EMAX=1.0e5;
  double f10=integ(s,r,d,g,10.0,EMAX), f1=integ(s,r,d,g,1.0,EMAX), ft=integ(s,r,d,g,1.0e-8,EMAX);
  cout<<tag<<" "<<s<<" "<<r<<" "<<d<<" "<<g<<" "<<scientific<<f10<<" "<<f1<<" "<<(f1-f10)<<" "<<ft;
  string kw; while(is>>kw){
   if(kw=="WB"){double sig,eth,w,k; is>>sig>>eth>>w>>k; cout<<" "<<fold(s,r,d,g,sig,eth,w,k);}
   else if(kw=="WC"){double sig,eth,w,k,ec; is>>sig>>eth>>w>>k>>ec; cout<<" "<<fold(s,r,d,g,sig,eth,w,k,ec);}
   else if(kw=="WBP"){double sig,eth,w,k; is>>sig>>eth>>w>>k; cout<<" "<<fold(s,r,d,g,sig,eth,w,k,1.0e5,1);}
   else if(kw=="WCP"){double sig,eth,w,k,ec; is>>sig>>eth>>w>>k>>ec; cout<<" "<<fold(s,r,d,g,sig,eth,w,k,ec,1);}
   else if(kw=="WR"){double sig,eth,w,k,e1,e2; int ip; is>>sig>>eth>>w>>k>>e1>>e2>>ip; cout<<" "<<fold(s,r,d,g,sig,eth,w,k,1.0e5,ip,e1,e2);}
   else if(kw=="PR"){cout<<" "<<integ(s,r,d,g,10.0,EMAX,1);}
   else break;}
  cout<<defaultfloat<<"\n"; }
}
