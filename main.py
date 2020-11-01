import sys
import math
import pandas as pd
import random
import numpy as np
from multiprocessing import Pool
import time

#usando exemplo_backprop_rede1.txt
reg_lambda=0.000
neunos_por_camada=[1, 2, 1]
#já considera os termos de bias
theta=[]
theta1=np.matrix([[0.40000 , 0.10000  ], [0.30000 , 0.20000 ]])
theta2=np.matrix([0.70000,  0.50000,  0.60000 ])
theta.append(theta1)
theta.append(theta2)
treino=	np.matrix([[0.13000, 0.90000], [0.42000, 0.23000]])
#treino x->y

#função sigmoide
def fun_g(x):
    return 1.0/(1.0+np.exp(-x))

def rede(x):
    #variaveis locais a, z
    print("entrada x: ",x)
    z=np.matrix(x)#insere valor de entrada
    for i in range(len(neunos_por_camada)):
        a=np.matrix([1.0])#termo de bias
        a=np.concatenate((a,z), axis=1)
        print("a: ",a)
        if i==len(neunos_por_camada)-1:
            print("ACABOU")
            f=a[0,1]
            print("saida f: ",f)
            return f
        z=a*np.transpose(theta[i])#calcula saidas da camada
        print("z: ",z)
        z=fun_g(z)

def main():
    #usando exemplo_backprop_rede1.txt
    delta=[]
    delta.append(rede(treino[0,0])-treino[0,1])
    print("delta da ultima camada= ", delta)
    
    #delta_camada=[]
    #for i in range(neunos_por_camada[1]):

    #delta.insert(0,)#,theta[-1]*delta[0+1])

    return

if __name__ == "__main__":
    main()