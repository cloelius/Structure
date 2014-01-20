import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import special
from itertools import product
from scipy.optimize import fsolve, curve_fit
def diffeq(L,Energy):
    "This function returns a function that will be the differential equation to be solved based on E and L"
    def deriv_chi(f, r):
            chi, chiprime = f
            #return [chiprime, -2.9233295/(1.0+np.exp((r-2.58532)/.65))*chi-.047845*Energy*chi+L*(L+1)*chi/r**2 ]
            return [chiprime, (-2.9233295/(1.0+np.exp((r-2.58532)/.65))*chi-.047845*Energy*chi+L*(L+1)*chi/r**2) ]

            #return [chiprime, 2.92/(1.0+np.exp((r-2.59)/.65))*chi-.047845*Energy*chi-.01*L*(L+1)*chi/r**2 ]
    return deriv_chi
sin=np.sin
def HenkelPluslim(rho,L):
    "Hankel function asymptotic using exp(i*r)"
    return (1j)**(-L)*np.exp(1j*rho)
def HenkelMinuslim(rho,L):
    return (1j)**(L)*np.exp(-1j*rho)
def HenkelPlusPrimelim(rho,L,k):
    return k*(1j)**(-L+1)*np.exp(1j*rho)
def HenkelMinusPrimelim(rho,L,k):
    return -k*(1j)**(L+1)*np.exp(-1j*rho)
def HenkelPlus(rho,L):
    return np.exp(1j*(rho-L*np.pi/2))
def HenkelMinus(rho,L):
    return np.exp(-1j*(rho-L*np.pi/2))
def HenkelPlusPrime(rho,L,k):
    return k*1j*np.exp(1j*(rho-L*np.pi/2))
def HenkelMinusPrime(rho,L,k):
    return k*(-1j)*np.exp(-1j*(rho-L*np.pi/2))



class chi:
    "At some level a general second order differential equation solver, which anticipates the problem. It uses runge-kutta in SolveDifEq, after using SetDiffEq to make the equation to be solved. It keeps the result, as well as all input used in an object for easy calculation and plotting"
    diff=0#Differential equation to be solved, set via SetDiffEq
    init=[]#array of initial values in the form [y(0),y'(0)]
    rvalues=[]#array containing all r values to be sampled over 
    Energy=0#Energy of system in Schrodinger Eq.
    L=0#Angular Momentum Quantum number in Schro. Eq.
    chivalues=[]#Array containing chi(x) values, each element in the array corresponds as such: chivalues[i]=chi(rvalues[i])
    chiprimevalues=[]#Similar to chivalues but for chi prime
    delta=0#Phase shift.
    Rmat=0#Logarithmic Derivative divided by a
    SMat=0#Represents admixture of Hankel plus function(which is to say "outgoing spherical wave")
    sindelta=0#sin of delta
    Mass=931.49272#Where this comes from taking .0478 from the notes, dividing by 2, multipling by hbar, and then dividing by (1 fm^2*1MeV)
    crosssection=0#Cross Section in arb units.
    aindex=0#index of chivalues, chiprimevalues to be used to generate R matrix(i.e. chivalues[aindex]/chivaluesprime[aindex])
    def SetDiffEq(self):
        "Choose Differential Equation to be solved"
        self.diff=diffeq(self.L,self.Energy)
    def SolveDifeq(self):
        "Solve Differential Equation"
        z = integrate.odeint(self.diff, self.init, self.rvalues,full_output=0,mxstep=10000)
        self.chivalues, self.chiprimevalues=z.T
       # print(self.chivalues[-1])
        #print(self.chiprimevalues[-1])
    def GetK(self):
        "Calculate k in fm"
        k=np.sqrt(.047845*self.Energy)#hbar^2k^2/2m=E
        return k
    def SetRMatrix(self):
        "Find Logarithmic Derivative"
        self.Rmat=self.chivalues[self.aindex]/self.chiprimevalues[self.aindex]/self.rvalues[self.aindex]
    def SetSMatrix(self):
        "Calculate S Matrix from RMatrix using Hankel Functions"
        k=self.GetK()
        a=self.rvalues[self.aindex]
        #a=self.chivalues[-1]
        num=HenkelMinus(k*a,self.L)-a*self.Rmat*HenkelMinusPrime(k*a,self.L,k)
        denom=HenkelPlus(k*a,self.L)-a*self.Rmat*HenkelPlusPrime(k*a,self.L,k)
        #num=special.hankel2(self.L,k*self.rvalues[-1])-1/2*k*self.rvalues[-1]*self.Rmat*(special.hankel2(self.L-1,k*self.rvalues[-1])-special.hankel2(self.L+1,k*self.rvalues[-1]))
       # denom=special.hankel1(self.L,k*self.rvalues[-1])-self.rvalues[-1]*k*self.Rmat*(self.L*special.hankel1(self.L,k*self.rvalues[-1])/(k*self.rvalues[-1])-special.hankel1(self.L+1,k*self.rvalues[-1]))
        #num=(np.sqrt(self.rvalues[-1])-1/(2*np.sqrt(self.rvalues[-1])))*special.hankel2(self.L,k*self.rvalues[-1])-1/2*k*np.sqrt(self.rvalues[-1])*self.rvalues[-1]*self.Rmat*(special.hankel2(self.L-1,k*self.rvalues[-1])-special.hankel2(self.L+1,k*self.rvalues[-1]))
        #denom=(np.sqrt(self.rvalues[-1])-1/(2*np.sqrt(self.rvalues[-1])))*special.hankel1(self.L,k*self.rvalues[-1])-np.sqrt(self.rvalues[-1])*self.rvalues[-1]*k*self.Rmat*(self.L*special.hankel1(self.L,k*self.rvalues[-1])/(k*self.rvalues[-1])-special.hankel1(self.L+1,k*self.rvalues[-1]))
        #num=HenkelMinuslim(k*a,self.L)-a*self.Rmat*HenkelMinusPrimelim(k*a,self.L,k)
        #denom=HenkelPluslim(k*a,self.L)-a*self.Rmat*HenkelPlusPrimelim(k*a,self.L,k)
        self.SMat=num/denom
  #      print(self.SMat)
    def SetDelta(self):
        "Calculate Delta From Logarithm"
        self.delta=1/(2j)*np.log(self.SMat)
      #  print(self.delta)
    def SetSinDelta(self):
        "Calculate Sin Delta, save in chi"
        self.sindelta=(sin(np.real(self.delta)))
    def SetCrossSection(self):
        "Calculate Cross Section in units of inverse k squared"
        self.crosssection=4*np.pi*(2*self.L+1)/(self.GetK()**2)*(self.sindelta)**2
    def __init__(self, rvalues,Energy,L,init,aindex):
        "Initialization, requires range, Energy, L, initial conditions and the index to be sampled at."
        self.rvalues=rvalues
        self.Energy=Energy
        self.L=L
        self.init=init
        self.aindex=aindex
def twodplot(x,y,title,xaxis,yaxis):
    fig=plt.figure()
    ax2=fig.add_subplot(111)
    ax2.plot(x,y)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
init=[.001,.001]
r=np.linspace(.0001,10000,200)
Energies=[.1,10]   
Ls=[0,1,2]
mychi=[]
pychi=[]
radwaves=[]
ain=-10
for vars in product(Energies,Ls):
    radwav=chi(r,vars[0],vars[1],init,ain)
    radwav.SetDiffEq()
    radwav.SolveDifeq()
    radwav.SetRMatrix()
    radwav.SetSMatrix()
    radwav.SetDelta()
    radwav.SetSinDelta()
    twodplot(radwav.rvalues,radwav.chivalues,"chi vs r at Energy "+str(vars[0])+" and L "+str(vars[1]),"r(fm)", "chi")
    #twodplot(radwav.rvalues,radwav.chiprimevalues,"chiprime vs r at Energy "+str(vars[0])+" and L "+str(vars[1]),"r(fm)", "chi")

    radwaves.append(radwav)
    
for xx in r:
    mychi.append( 1j/2*(HenkelMinus(radwaves[0].GetK()*xx,radwaves[0].L)-radwaves[0].SMat*HenkelPlus(radwaves[0].GetK()*xx,radwaves[0].L)))
    pychi.append( np.sqrt(xx)*1j/2*(special.hankel2(radwaves[0].L,radwaves[0].GetK()*xx)-radwaves[0].SMat*special.hankel2(radwaves[0].L,radwaves[0].GetK()*xx)))
rs=np.linspace(100,10000,100)
print(mychi)
print(len(mychi))
deltas=[]
sindeltas=[]
Ss=[]
CrossSections=[]

for a in rs:
    Energy=10
    L=0
    rr=np.linspace(.00001,a,1000)
    radwav=chi(rr,Energy,L,init,ain)
    radwav.SetDiffEq()
    radwav.SolveDifeq()
    radwav.SetRMatrix()
    radwav.SetSMatrix()
    radwav.SetDelta()
    radwav.SetSinDelta()
    radwav.SetCrossSection()
    deltas.append(radwav.delta)
    sindeltas.append(radwav.sindelta)
    Ss.append(radwav.SMat)
    CrossSections.append(radwav.crosssection)
twodplot(rs,deltas,"delta vs a at Energy .1 MeV and L 1","r(fm)", "delta")
twodplot(rs,sindeltas,"sin delta vs a at Energy .1 MeV and L 1","r(fm)", "sin delta")
#twodplot(rs,Ss,"S vs a at Energy .1 MeV and L 1","r(fm)", "S")
#twodplot(r,mychi,"mychi","x","chi")
#twodplot(r,pychi,"pychi","x","chi")
twodplot(rs,CrossSections,"Cross sections vs a", "a(fm)","Cross Section")
EnergiesDelta=np.linspace(.1,4,8)
for L in Ls:
    deltasen=[]
    sindeltasen=[]
    Smats=[]
    rmats=[]
    rmatsfromsmats=[]
    crosssections=[]
    print(len(EnergiesDelta))
    i=0
    for En in EnergiesDelta:
        i=i+1
        print(i)
        radwav=chi(r,En,L,init,ain)
        radwav.SetDiffEq()
        radwav.SolveDifeq()
        radwav.SetRMatrix()
        radwav.SetSMatrix()
        radwav.SetDelta()
        radwav.SetSinDelta()
        radwav.SetCrossSection()
        deltasen.append(radwav.delta)
        sindeltasen.append(radwav.sindelta)
        Smats.append(radwav.SMat)
        rmats.append(np.abs(radwav.Rmat))
        crosssections.append(radwav.crosssection)
        rmatsfromsmats.append(1/radwav.rvalues[ain]*(HenkelMinus(radwav.GetK()*radwav.rvalues[ain],radwav.L)-radwav.SMat*HenkelPlus(radwav.GetK()*radwav.rvalues[-1],radwav.L))/((HenkelMinusPrime(radwav.GetK()*radwav.rvalues[-1],radwav.L,radwav.GetK())-radwav.SMat*HenkelPlusPrime(radwav.GetK()*radwav.rvalues[-1],radwav.L,radwav.GetK()))))
    twodplot(EnergiesDelta,deltasen," delta vs Energy for  L "+str(L),"E(MeV)", "delta")
    #twodplot(EnergiesDelta,sindeltasen," sin delta vs Energy for  L "+str(L),"E(MeV)", " sin delta")
    #twodplot(EnergiesDelta,Smats," SMat vs Energy for  L "+str(L),"E(MeV)", " SMat")
    #twodplot(EnergiesDelta,rmats," RMat vs Energy for  L "+str(L),"E(MeV)", " RMat")
   # twodplot(EnergiesDelta,rmatsfromsmats," RMat from Smat vs Energy for  L "+str(L),"E(MeV)", " RMat")
    twodplot(EnergiesDelta,crosssections," Cross section vs Energy for  L "+str(L),"E(MeV)", " Cross Section")
PrimeCrossSections=[]
primes=np.linspace(.001,10000,1000)
for prime in primes:
    Energy=10
    L=0
    inits=[0,prime]
    rr=np.linspace(.00001,100,1000)
    radwav=chi(rr,Energy,L,inits,ain)
    radwav.SetDiffEq()
    radwav.SolveDifeq()
    radwav.SetRMatrix()
    radwav.SetSMatrix()
    radwav.SetDelta()
    radwav.SetSinDelta()
    radwav.SetCrossSection()
    #deltas.append(radwav.delta)
    #sindeltas.append(radwav.sindelta)
    #Ss.append(radwav.SMat)
    PrimeCrossSections.append(radwav.crosssection)
twodplot(primes,PrimeCrossSections,"Cross Section as Value of Initial Condition","chi'(0)", "Cross Section(arb units)")
plt.show()

