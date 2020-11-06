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
a_list=[]

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
        a_list.append(a)
        print("a: ",a)
        if i==len(neunos_por_camada)-1:
            print("ACABOU")
            f=a[0,1]
            #a_list[-1]=np.matrix(f)
            print("saida f: ",f)
            return f
        z=a*np.transpose(theta[i])#calcula saidas da camada
        print("z: ",z)
        z=fun_g(z)


def main():
    #usando exemplo_backprop_rede1.txt
    delta=[]
    #utiliza o primeiro exemplo
    erro=rede(treino[0,0])-treino[0,1]
    delta.append(np.matrix([erro]))
    print("delta da ultima camada= ", delta)
    print(a_list)
    for i in range(len(neunos_por_camada)-2,0,-1):#delta da penultima camada até a segunda
        print("camada ",i)
        x=(np.transpose(theta[i])*delta[0])
        a_mod=np.array(a_list[i].tolist())*np.array((1-a_list[i]).tolist())
        x=np.array(np.transpose(x))*np.array(a_mod.tolist())
        x=np.matrix(x)
        x=np.delete(x, 0, 1)#deleta a primeira(0) coluna(1)
        print(x)
        delta.insert(0,x)
    
    delta.insert(0,x)#duplica ultimo só pra ficar alinhado
    print("deltas")
    print(delta)

    #D é o gradiente
    
    return

if __name__ == "__main__":
    main()