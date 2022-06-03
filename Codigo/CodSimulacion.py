import numpy as np
import matplotlib.pyplot as plt
import csv

def ev_posicion(gamma, k, sigma, Delta_t, x_old):
    xi = np.random.normal(loc=0, scale=sigma)
    x_new = (Delta_t/gamma) * (xi - k*x_old) + x_old
    
    return x_new

def isoterma_opt(l, Delta_t, k_0, y_0, k_f, y_f, T_0):
    t_f = l*Delta_t
    t = np.linspace(0, t_f, l)
    
    T = np.ones(l)*T_0
    
    y_opt = (np.sqrt(y_0) + (np.sqrt(y_f) - np.sqrt(y_0))*t/t_f)**2

    k = T/y_opt - (np.sqrt(y_f/y_opt) - np.sqrt(y_0/y_opt))/t_f

    k[0] = k_0
    k[-1] = k_f
    
    return t, k, T, y_opt

def adiabata_opt(l, k_0, y_0, T_0, k_f, y_f, T_f):
    t_f = (y_f - y_0)**2/(2*(y_f*T_f - y_0*T_0))
    
    t = np.linspace(0, t_f, l)
    
    y_opt = y_0 + (y_f - y_0)*t/t_f
    T_opt = (y_0*T_0 + (y_f*T_f - y_0*T_0)*t/t_f)/y_opt
    k_opt = T_opt/y_opt - (y_f - y_0)/(2*y_opt*t_f)

    k_opt[0] = k_0
    k_opt[-1] = k_f
    
    return t, k_opt, T_opt, y_opt

def proceso(l, Delta_t, k, T):
    x = np.zeros(l)
    sigma = np.sqrt(2*gamma*kb*T)
    
    for i in range(5000):
       x[0] = ev_posicion(gamma, k[0], sigma[0], Delta_t, x[0]) 

    for i in range(l - 1):
        x[i + 1] = ev_posicion(gamma, k[i + 1], sigma[i + 1], Delta_t, x[i])
    
    return x


#Constantes y parametros
gamma = 1
kb = 1

chi = 0.6
nu = 0.6
c = 0.96
d = 1.03

l = 10000

#Compresion isotermica

#Condiciones iniciales
Delta_t1 = 1E-3

k_A = 1
y_A = 1
k_B = chi
y_B = 1/chi
T_AB = 1

#Proceso optimo
t1, k1, T1, y_opt1 = isoterma_opt(l, Delta_t1, k_A, y_A, k_B, y_B, T_AB)

#Iteracion del ciclo
cant = 1000

y1 = np.zeros(l)
W1_stochastic = []

for i in range(cant):
    x1 = proceso(l, Delta_t1, k1, T1)    
    U1 = (k1/2)*(x1)**2
    W1 = 0
        
    for j in range(1, l - 1):
        W1 += (U1[j] - U1[j - 1])
        
    W1_stochastic.append(W1*1E3)
    y1 += (x1)**2
        
y1 /= cant

plt.figure()
plt.scatter(k1[0:-1:200], y1[0:-1:200]*1E3, c="red", label="Simulado")
plt.plot(k1, y_opt1, label="Óptimo")
plt.xlabel("$\kappa$")
plt.ylabel("y")
plt.legend()
plt.title("Compresión isotérmica")
plt.savefig("FigAB-1.png")

W1_prom = np.mean(W1_stochastic)
W1_std = np.std(W1_stochastic)
W1_calc = (T_AB/2)*np.log(k_B/k_A) + (T_AB/t1[-1])*(1/np.sqrt(k_B) - 1/np.sqrt(k_A))**2

res1_prom = " = {:.3f} $\pm$ {:.3f}".format(W1_prom, W1_std)
res1_calc = " = {:.3f}".format(W1_calc)

plt.figure()
plt.hist(W1_stochastic, bins=40)
plt.axvline(W1_prom, c="red", label="Promedio: $\mathcal{W}$" + res1_prom)
plt.axvline(W1_calc, c="blue", label="Óptimo: $\mathcal{W}$" + res1_calc, linestyle="--")
plt.title("Distribución trabajo estocástico")
plt.xlabel("$\mathcal{W}_{\mathrm{estocastico}}$")
plt.ylabel("Repeticiones")
plt.legend()
plt.savefig("FigAB-2.png")



#Compresion adiabática

#Condiciones iniciales
k_B = chi
y_B = 1/chi
T_B = 1
k_C = c*(nu**2)*chi
y_C = (c*nu*chi)**(-1)
T_C = nu

#Proceso optimo
t2, k2, T2, y_opt2 = adiabata_opt(l, k_B, y_B, T_B, k_C, y_C, T_C)

#Iteracion del ciclo
Delta_t2 = 1E-3
cant = 1000

y2 = np.zeros(l)
W2_stochastic = []

for i in range(cant):
    x2 = proceso(l, Delta_t2, k2, T2)    
    U2 = (k2/2)*(x2)**2
    W2 = 0
    
    for j in range(1, l - 1):
        W2 += (U2[j] - U2[j - 1])
    
    W2_stochastic.append(W2*1E3)
    y2 += (x2)**2
    
y2 /= cant

plt.figure()
plt.scatter(k2[0:-1:200], y2[0:-1:200]*1E3, c="red", label="Simulado")
plt.plot(k2, y_opt2, label="Óptimo")
plt.xlabel("$\kappa$")
plt.ylabel("y")
plt.legend()
plt.title("Compresión adiabática")
plt.savefig("FigBC-1.png")

W2_prom = np.mean(W2_stochastic)
W2_std = np.std(W2_stochastic)
W2_calc = T_C - T_B

res2_prom = " = {:.3f} $\pm$ {:.3f}".format(W2_prom, W2_std)
res2_calc = " = {:.3f}".format(W2_calc)

plt.figure()
plt.hist(W2_stochastic, bins=40)
plt.axvline(W2_prom, c="red", label="Promedio: $\mathcal{W}$" + res2_prom)
plt.axvline(W2_calc, c="blue", label="Óptimo: $\mathcal{W}$" + res2_calc, linestyle="--")
plt.title("Distribución trabajo estocástico")
plt.xlabel("$\mathcal{W}_{\mathrm{estocastico}}$")
plt.ylabel("Repeticiones")
plt.legend()
plt.savefig("FigBC-2.png")



#Expansion isotermica

#Condiciones iniciales
Delta_t3 = 1E-3

k_C = c*(nu**2)*chi
y_C = (c*nu*chi)**(-1)
k_D = d*nu**2
y_D = (d*nu)**(-1)
T_CD = nu

#Proceso optimo

t3, k3, T3, y_opt3 = isoterma_opt(l, Delta_t3, k_C, y_C, k_D, y_D, T_CD)

#Iteracion del ciclo

cant = 1000

y3 = np.zeros(l)
W3_stochastic = []

for i in range(cant):
    x3 = proceso(l, Delta_t3, k3, T3)    
    U3 = (k3/2)*(x3)**2
    W3 = 0
    
    for j in range(1, l - 1):
        W3 += (U3[j] - U3[j - 1])
    
    W3_stochastic.append(W3*1E3)
    y3 += (x3)**2
    
y3 /= cant

plt.figure()
plt.scatter(k3[0:-1:200], y3[0:-1:200]*1E3, c="red", label="Simulado")
plt.plot(k3, y_opt3, label="Óptimo")
plt.xlabel("$\kappa$")
plt.ylabel("y")
plt.legend()
plt.title("Expansión isotérmica")
plt.savefig("FigCD-1.png")

W3_prom = np.mean(W3_stochastic)
W3_std = np.std(W3_stochastic)
W3_calc = (T_CD/2)*np.log(k_D/k_C) + (T_CD/t3[-1])*(1/np.sqrt(k_D) - 1/np.sqrt(k_C))**2

res3_prom = " = {:.3f} $\pm$ {:.3f}".format(W3_prom, W3_std)
res3_calc = " = {:.3f}".format(W3_calc)

plt.figure()
plt.hist(W3_stochastic, bins=40)
plt.axvline(W3_prom, c="red", label="Promedio: $\mathcal{W}$" + res3_prom)
plt.axvline(W3_calc, c="blue", label="Óptimo: $\mathcal{W}$" + res3_calc, linestyle="--")
plt.title("Distribución trabajo estocástico")
plt.xlabel("$\mathcal{W}_{\mathrm{estocastico}}$")
plt.ylabel("Repeticiones")
plt.legend()
plt.savefig("FigCD-2.png")



#Expansion adiabatica

#Condiciones iniciales
k_D = d*(nu**2)
y_D = (d*nu)**(-1)
T_D = nu
k_A = 1
y_A = 1
T_A = 1

#Proceso optimo
t4, k4, T4, y_opt4 = adiabata_opt(l, k_D, y_D, T_D, k_A, y_A, T_A)

#Iteracion del ciclo
Delta_t4 = 1E-3

cant = 1000

y4 = np.zeros(l)
W4_stochastic = []

for i in range(cant):
    x4 = proceso(l, Delta_t4, k4, T4)    
    U4 = (k4/2)*(x4)**2
    W4 = 0
    
    for j in range(1, l - 1):
        W4 += (U4[j] - U4[j - 1])
    
    W4_stochastic.append(W4*1E3)
    y4 += (x4)**2
    
y4 /= cant

plt.figure()
plt.scatter(k4[0:-1:200], y4[0:-1:200]*1E3, c="red", label="Simulado")
plt.plot(k4, y_opt4, label="Óptimo")
plt.xlabel("$\kappa$")
plt.ylabel("y")
plt.legend()
plt.title("Expansión adiabática")
plt.savefig("FigDA-1.png")

W4_prom = np.mean(W4_stochastic)
W4_std = np.std(W4_stochastic)
W4_calc = T_A - T_D

res4_prom = " = {:.3f} $\pm$ {:.3f}".format(W4_prom, W4_std)
res4_calc = " = {:.3f}".format(W4_calc)

plt.figure()
plt.hist(W4_stochastic, bins=40)
plt.axvline(W4_prom, c="red", label="Promedio: $\mathcal{W}$" + res4_prom)
plt.axvline(W4_calc, c="blue", label="Óptimo: $\mathcal{W}$" + res4_calc, linestyle="--")
plt.title("Distribución trabajo estocástico")
plt.xlabel("$\mathcal{W}_{\mathrm{estocastico}}$")
plt.ylabel("Repeticiones")
plt.legend()
plt.savefig("FigDA-2.png")



#Ciclo

plt.figure()
plt.scatter(k1[0:-1:200], y1[0:-1:200]*1E3, c="red", alpha=0.5)
plt.scatter(k2[0:-1:200], y2[0:-1:200]*1E3, c="green", alpha=0.5)
plt.scatter(k3[0:-1:200], y3[0:-1:200]*1E3, c="red", alpha=0.5)
plt.scatter(k4[0:-1:200], y4[0:-1:200]*1E3, c="green", alpha=0.5)

plt.plot(k1, y_opt1, c="red")
plt.plot(k2, y_opt2, c="green")
plt.plot(k3, y_opt3, c="red")
plt.plot(k4, y_opt4, c="green")

plt.xlabel("$\kappa$")
plt.ylabel("y")
plt.title("Ciclo irreversible con base en el de Carnot")
plt.savefig("FigCiclo-1.png")

W_ciclo_stochastic = W1_stochastic + W2_stochastic + W2_stochastic + W4_stochastic

W_ciclo_prom = np.mean(W_ciclo_stochastic)
W_ciclo_std = np.std(W_ciclo_stochastic)
W_ciclo_calc = W1_calc + W2_calc + W3_calc + W4_calc

res_ciclo_prom = " = {:.3f} $\pm$ {:.3f}".format(W_ciclo_prom, W_ciclo_std)
res_ciclo_calc = " = {:.3f}".format(W_ciclo_calc)

plt.figure()
plt.hist(W_ciclo_stochastic, bins=40)
plt.axvline(W_ciclo_prom, c="red", label="Promedio: $\mathcal{W}$" + res_ciclo_prom)
plt.axvline(W_ciclo_calc, c="blue", label="Óptimo: $\mathcal{W}$" + res_ciclo_calc, linestyle="--")
plt.title("Distribución trabajo estocástico del ciclo")
plt.xlabel("$\mathcal{W}_{\mathrm{estocastico}}$")
plt.ylabel("Repeticiones")
plt.legend()
plt.savefig("FigCiclo-2.png")

plt.show()
