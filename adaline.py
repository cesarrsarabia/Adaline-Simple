#Ramirez Sarabia César Eduardo 
#Perceptron. Práctica 4
import numpy as np
import matplotlib
from pylab import xlim
from pylab import ylim
import matplotlib.pyplot as plt
import time
import numpy as np
import math

def grafica_puntos():
    colores = ['*b','*g','pm','or','+k']
    aux_arr = []
    xlim([-2,3])
    ylim([-2,3])
    last_plot=-1
    for cont,this_element in enumerate(salidas_deseadas):
        str_salida = '-'.join(str(e) for e in this_element)
        if str_salida not in aux_arr:
            last_plot+=1
            aux_arr.append(str_salida)
            x_1 = datos_entradas[cont]
            ax.plot(x_1[0],x_1[1], colores[cont])
    return last_plot

    
    
def grafica_lineas(this_w,last_plot,is_last_plot):
    clases = ['--k','--b','--r','--y','--g']
    if not is_last_plot:
        index_plot = last_plot
        for c,item in enumerate(this_w):
            ax.plot([-2,3],[(-item[0]/item[1]) *
            (-2)-(b[c]/item[1]), (-item[0]/item[1])*(3)-(b[c]/item[1])],clases[c])
            plt.pause(0.0000000000000000000000000000000000000000000000000000000001)
            index_plot += 1
        for i in range(index_plot,last_plot,-1):
            ax.lines.pop(i)
    else:
        plt.cla()
        grafica_puntos()
        for c,item in enumerate(this_w):
            ax.plot([-2,3],[(-item[0]/item[1]) *
            (-2)-(b[c]/item[1]), (-item[0]/item[1])*(3)-(b[c]/item[1])],clases[c])
        
        #plt.close('all')
    #for c in range(1,num_col_salidas):
    #   ax.lines.pop(last_plot + c)

def funcion_escalon(x):
    exponencial = math.e
    return 1/(1 + pow(exponencial,-x))


def calculaPromErrores():
    sumErroes = 0
    for x_e in errores:
        sumErroes += pow(x_e,2)
    return (1/num_col_salidas) * sumErroes

def obtiene_Salida(r,c):
    posicion = salidas_deseadas[r]
    if c == 0:
        return posicion[0]
    else:
        return posicion[c+1]


#training set contiene entradas y salidas
training_set = []

#plt.ion()
with open('entradas.txt') as f_1:
    datos_entradas = f_1.read().splitlines() 

with open('salidas_deseadas.txt') as f_2:
    salidas_deseadas = f_2.read().splitlines() 

#Convierte entrada a lista de enteros
for i, element in enumerate(datos_entradas):
    x =  [int(k) for k in element.split()]
    datos_entradas[i] = x

#Convierte salidas deseadas a lista de enteros
for i, element in enumerate(salidas_deseadas):
    x =  [int(k) for k in element.split()]
    salidas_deseadas[i] = x
num_col_salidas = len(salidas_deseadas[0]) - salidas_deseadas[0].count(' ')

# Inicializacion de parametros ( pesos, eta , b)
w = np.random.rand(num_col_salidas,2)
errores = [] 
eta = .6 #Ajustar para la velocidad en la que ajusta
b = np.zeros(num_col_salidas)
error_global = 0.28 #Error a usar para el criterio de paro

for i in range(num_col_salidas):
    row = 0
    training_set_aux = []
    for j in datos_entradas:
        aux = salidas_deseadas[row]
        a = ((int(j[0]),
                int(j[1])),
                aux[i])
        training_set_aux.append(a)
        row += 1
    training_set.append(training_set_aux)

fig, ax = plt.subplots()


lst_i_plot = grafica_puntos()
errores_prom = []

#ENTRENA
while True:
    for count,fila_training_item in enumerate(training_set):
        for x,y in fila_training_item:
            u = sum(x*w[count]) + b[count] 
            error = y - funcion_escalon(u)
            errores.append(error)            
            val = funcion_escalon(u)
            for index, value in enumerate(x):          
                w[count][index] += eta * error * value * (val*(1-val))
                b[count] += eta*error
                grafica_lineas(w,lst_i_plot,False)
    
    calculo_errores = calculaPromErrores()
    print(calculo_errores)
    if calculo_errores < error_global:
        print('TERMINAR DE ENTRENAR')
        break
    else:
        errores.clear()
        errores_prom.append(calculo_errores)
    
ax = plt.subplot(111)
ax.plot(errores_prom, c='#aaaaff', label='Errores de Entrenamiento')
#ax.set_xscale("log")
#plt.title("ADALINE Errors (2,-2)")
plt.legend()
plt.xlabel('Error')
plt.ylabel('Epocas')
#plt.show()

'''
np.linspace(0,)
for index, value in enumerate(errores):        
    ax.plot(index,value,'*b')
    ax.plot(index,value,'--g')
    plt.pause(0.0001)
    '''
'''
t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)
plt.figure(2)
plt.subplot(211)
plt.plot(t, s1)
plt.subplot(212)
plt.plot(t, 2*s1)
'''
print("PERCEPTRON ENTRENADOS")
#grafica_lineas(w,lst_i_plot,True)
plt.show()
