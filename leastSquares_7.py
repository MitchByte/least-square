#Autorin: Fischer Michele
#serie 4

import numpy as np
import scipy 
from scipy import linalg
from numpy import loadtxt
import pandas



#Loesung des Problems Ax=b#
###########################
 
def DK(n):
    k = range(0, n, 1)
    k = [float(i) for i in k]
    

    #Erstellen der dk-Exponenten 

    dk = []
    for i in k:
        if n > 1:
            d = 0.1
            d = (d + ((0.4*i)/(float(n)-1.0)))
            dk.append(d)
        else:
            d = 0.1
            dk.append(d)
    
    return dk
    
        
def qr(xi,yi,dk):
    z1k = []
    z2k = []
    r1k = []
    N = 3 #anzahl der gesuchten parameter

    for d in dk:
    #erstellen der Matrix A
        xi = [float(i) for i in xi]
        A = []
            
        for x in xi:
            m = float(x)*d
            ai = [np.exp(m),1.0/np.exp(m),1]
            A.append(ai)
        A = np.array(A)
            
            
        #QR-Zerlegung
        Q, R = np.linalg.qr(A)
        #z=(Q^t)*b
        z = (np.array(np.transpose(Q))).dot(np.array(yi))
        
        #Zerlegung von z
        Z1 = z[:N]
        Z2 = z[N:]
        #Zerlegung von R
        R1 = R[:N]
        z1k.append(Z1)
        z2k.append(Z2)
        r1k.append(R1)
            
    z1k = np.array(z1k)
    z2k = np.array(z2k)
    r1k = np.array(r1k)  
    return z1k, z2k, r1k
    
    
#Rueckwartseinsetzen
        
def solve(z1k,z2k,r1k):
	m = len(z1k)
	xk = []
	for l in range(m):
		parameter = []
        r1 = r1k[l]
        z1 = z1k[l]
		
        for i in range(1, m+1):
			rn = r1[-i]  #i-te zeile, von unten nach oben
			rnn = rn[-i] #i-ter eintrag der i-ten zeile
			rnm = rn[1:] #i-te zeile ohne 1. eintrag
			print "Die",i,"-te Zeile der ",l+1,"-ten Matrix R1", rn
			print "rnm", rnm
			print "Eintrag durch den geteilt wird", rnn
			print "Die",l,"-te Matrix R1", r1
			try:
				t = np.array(parameter)*np.array(rnm) #Erstellen des Terms der zu subtrahieren ist
			except ValueError:
				t=0
			print "t:" , t
			xn = (z1[-i]-float(t))/ float(rnn)
			print "xn" , xn
			parameter.append(xn)
			print "parameter:", parameter
		#xk.append(parameter)
	print xk
	return xk

#cond_2(A) = cond_2(R) aus VL bekannt
def condition(xi,dk):
	k =[]
	for d in dk:
		#erstellen der Matrix A
		xi = [float(i) for i in xi]
		A = []
		for x in xi:
			m = float(x)*d
			ai = [np.exp(m),1.0/np.exp(m),1]
			A.append(ai)
		print A
		return A
		print A
		K = np.cond.linalg(A)
		k.append(K)
		print K
	return k
	

def eukldNorm(z2k):
	abw =[]
	for i in z2k:
		minAbw = np.sqrt(np.array(z2k[i])*np.array(z2k[i]))
		abw.append(minAbw)


#main-methode#
##############


def main():
	pandas.set_option('display.max_colwidth', -1)
	lines = loadtxt("daten.txt", comments="#", delimiter=",", unpack=True)
	
	xi=lines[0]
	yi=lines[1]
	n = raw_input("Geben Sie eine natuerliche Zahl n ein mit der die Genauigkeit der gesuchten Parameter bestimmt wird,je groesser desto genauer: ")
	n = int(n)
	#xi, yi =data(lines)
	print "Die eingegebenen Messdaten xi:", xi
	print "und die dazugehoerigen Messdaten yi:", yi
	#dk = DK(n)
	#dat = [[0,0,0]
	#for d in dk:
	#
	#	M = condition(A)
	#	E = condition(np.array(np.transpose(A)).dot(np.array(A)))
	#	data = np.append(data, [[d, M, E]], axis=0)
	#print pandas.DataFrame(dat, columns=['Genauigkeit dk', 'Kondition A', 'Kondiktion AtA'])
	
	k = condition(xi,dk)
	print "die kondition", k
	print "Die schrittweise-Genauigkeit im Exponenten der E-Funktion, dk: ", dk
	z1k, z2k, r1k = qr(xi,yi,dk)
	print "z1k ", z1k
	print "z2k ", z2k
	print "r1k ", r1k
	xk = solve(z1k,z2k,r1k)
	print "Die errechnete Loesungsmenge fuer verschieden dk: ", xk
	
	
	

    #import matplotlib
    #matplotlib.use('TkAgg')  
    #import matplotlib.pyplot as plt
	
	
   

            #plt.figure(i) 
            #plt.plot(i,  'r--')
            #plt.show()
        
        
            #for n in range(0, data.n +1, 1):
        
            #plt.plot([i/np.float(n) for i in xrange(n+1)],
    


#
