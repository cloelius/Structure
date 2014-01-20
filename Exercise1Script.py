import numpy
import matplotlib.pyplot as plt
from matplotlib  import cm
def MakeZNPlot(Z,N,X,TITLE):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_title(TITLE,fontsize=24)
    ax.set_xlabel("N",fontsize=16)
    ax.set_ylabel("Z",fontsize=16)
    ax.set_ylim([0,120])
    ax.set_xlim([0,160])

    ax.grid(True,linestyle='-',color='0.75')
    

    # scatter with colormap mapping to z value
    a=ax.scatter(N,Z,s=15,c=X, marker = 'o', cmap = cm.jet );
    plt.colorbar(a)
    plt.show()
    plt.savefig(TITLE+".png")
def MakeAPlot(A,X,TITLE,YLAB):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_title(TITLE,fontsize=14)
    ax.set_xlabel("A",fontsize=12)
    ax.set_ylabel(YLAB,fontsize=12)
    ax.grid(True,linestyle='-',color='0.75')
    ax.set_xlim([0,300])


    # scatter with colormap mapping to z value
    ax.scatter(A,X,s=5, marker = 'o' );
    plt.savefig(TITLE+".png")
    plt.show()
def MakeAPlots(A,X,TITLE,YLAB,names):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_xlim([0,300])
    ax.set_title(TITLE,fontsize=14)
    ax.set_xlabel("A",fontsize=12)
    ax.set_ylabel(YLAB,fontsize=12)
    ax.grid(True)
    for i,x in enumerate(X):
        plt.plot(A,x,label=names[i])
    plt.legend(loc=4)
    plt.savefig(TITLE+".png")
    plt.show()
def WBBE(A,Z,a1,a2,a3,a4):
        N=A-Z
        return a1*A-a2*A**(2.0/3.0)-a3*Z**2/(A**(1/3))-a4*(N-Z)**2/A
class Nucleus:
        Z=0
        N=0
        A=0
        Sn=0
        Sp=0
        SnWB=0
        SpWB=0
        BE=0
        BEEMP=0#BE is the binding energy from Weisaker-Bette
        alpha1=15.49#Energy in MeV
        alpha2=17.23
        alpha3=.697
        alpha4= 22.6
#         def __init__(self,Z,A,BE):
#             self.Z=Z
#             self.A=A
#             self.N=A-Z
#             self.BEEMP=WBBE(self.A,self.Z,self.alpha1,self.alpha2,self.alpha3,self.alpha4)
#             self.BE=BE
        def __init__(self,Z,A,BE,alpha1=15.49,alpha2=17.23,alpha3=.697,alpha4=22.6):
            self.Z=Z
            self.A=A
            self.N=A-Z
            self.alpha1=alpha1
            self.alpha2=alpha2
            self.alpha3=alpha3
            self.alpha4=alpha4
            self.BEEMP=WBBE(self.A,self.Z,self.alpha1,self.alpha2,self.alpha3,self.alpha4)
            self.BE=BE
            
        def  GetSeparationEnergies(self,l):
            if(A==2):
                self.Sp=self.BE
                self.Sn=self.BE
            else:
                for nuc in l:
                    if(nuc.N==self.N):
                        if(nuc.Z==int(self.Z-1)):
                            self.Sp=self.BE-nuc.BE
                            self.SpWB=self.BEEMP-nuc.BEEMP
                    if(nuc.Z==self.Z):
                            if(nuc.N==int(self.N)-1):
                                    self.Sn=self.BE-nuc.BE
                                    self.SnWB=self.BEEMP-nuc.BEEMP
fileform=numpy.loadtxt("C:\\Users\\Charles\\Downloads\\mass\\mass\\aud11.dat",skiprows=2)
nuclei=[]
nucleialpha1=[]
nucleialpha12=[]
nucleialpha123=[]
nucleialpha1234=[]

for line in fileform:#Read in files here
        Z=line[0]
        A=line[1]
        BE=line[2]
        nuclei.append(Nucleus(Z,A,BE))
        nucleialpha1.append(Nucleus(Z,A,BE,15.49,0,0,0))
        nucleialpha12.append(Nucleus(Z,A,BE,15.49,17.23,0,0))
        nucleialpha123.append(Nucleus(Z,A,BE,15.49,17.23,.697,0))
        nucleialpha1234.append(Nucleus(Z,A,BE,15.49,17.23,.697,22.6))

for n in nuclei:#perform first calculations
    n.GetSeparationEnergies(nuclei)
Ns=[]
Zs=[]
BEs=[]
WBBEs=[]
Sns=[]
Sps=[]
WBSns=[]
WBSps=[]
As=[]
BEPN=[]
BEREL=[]
WBBEsalpha1=[]
WBBEsalpha12=[]
WBBEsalpha123=[]
WBBEsalpha1234=[]

for n in nuclei:#Prepare for plotting
    Ns.append(n.N)
    Zs.append(n.Z)
    BEs.append(n.BE)
    WBBEs.append(n.BEEMP)
    As.append(int(n.N+n.Z))
    Sns.append(n.Sn)
    Sps.append(n.Sp)
    BEPN.append(n.BE/n.A)
    BEREL.append((n.BE-n.BEEMP)/n.BE)
    WBSns.append(n.SnWB)
    WBSps.append(n.SpWB)
for i in range(len(nuclei)):
    WBBEsalpha1.append(nucleialpha1[i].BEEMP/nucleialpha1[i].A)
    WBBEsalpha12.append(nucleialpha12[i].BEEMP/nucleialpha12[i].A)
    WBBEsalpha123.append(nucleialpha123[i].BEEMP/nucleialpha123[i].A)
    WBBEsalpha1234.append(nucleialpha1234[i].BEEMP/nucleialpha1234[i].A)

MakeZNPlot(Zs,Ns,Sps,"Proton Separation Energy(MeV) vs N, Z")
#MakeZNPlot(Zs,Ns,As)
MakeZNPlot(Zs,Ns,Sns,"Neutron Separation Energy(MeV) vs N, Z")
MakeZNPlot(Zs,Ns,BEs,"Binding Energy(MeV) vs N, Z")
MakeZNPlot(Zs,Ns,BEs,"Liquid Drop Model Binding Energy(MeV) vs N, Z")
MakeZNPlot(Zs,Ns,WBSns,"Liquid Drop Model Sn (MeV) vs N, Z")
MakeZNPlot(Zs,Ns,WBSps,"Liquid Drop Model Sp (MeV) vs N, Z")
MakeZNPlot(Zs,Ns,BEREL,"Binding Energy Minus Liquid Drop Model Binding Energy over Binding Energy(MeV) vs N, Z")
MakeAPlot(As,BEs,"Binding Energy(MeV) vs A","Binding Energy")   
WBs=[]
#WBs.append(WBBEs)
WBs.append(WBBEsalpha1)
WBs.append(WBBEsalpha12)
WBs.append(WBBEsalpha123)
WBs.append(WBBEsalpha1234)
names=["Volume Term", "Volume and Surface Term", "Volume, Surface, and Coulomb term", "Volume, Surface, Coulomb and Pairing term"]
MakeAPlots(As,WBs,"Different Terms in Liquid Drop","Binding Energy per Nucleon(MeV)",names)
#print(nuclei[20].Sp)
#print(nuclei[20].Sn)
