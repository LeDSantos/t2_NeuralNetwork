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
#a_list=[]
D=[]
D.append(np.matrix([[0, 0], [0,0]]))#gradiente 1 inicial
D.append(np.matrix([[0, 0,0]]))  

#função sigmoide
def fun_g(x):
    return 1.0/(1.0+np.exp(-x))

def rede(x):
    #variaveis locais a, z e a_list
    a_list=[]
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
            return a_list, f
        z=a*np.transpose(theta[i])#calcula saidas da camada
        print("z: ",z)
        z=fun_g(z)


def main():
    #usando exemplo_backprop_rede1.txt
    #utiliza 2 exemplos
    alfa=0.01#####ULTIMA LINHA DO ALG, NÃO SEI O MELHOR VALOR
    for n_exemplo in range(len(treino)):
        print("\n\n---------------------\nEXEMPLO ",n_exemplo)
        a_list, previsao = rede(treino[n_exemplo,0])
        erro=previsao-treino[n_exemplo,1]
        delta=[]
        delta.append(np.matrix([erro]))
        print("delta da ultima camada= ", delta)
        #print(a_list)
        print("deltas")
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
        #print(delta)
        
        print("calculando D")
        for i in range(len(neunos_por_camada)-2,-1,-1):#D da penultima camada até a primeira
            print("camada ",i)
            teste=np.transpose(delta[i+1])*(a_list[i])
            #print(teste)
            D[i]=D[i]+teste
            print(D[i])

    #return
    print("\nDADOS DE TREINO PROCESSADOS\n-----------------\ncalculando D regularizado")
    for i in range(len(neunos_por_camada)-2,-1,-1):#D da penultima camada até a primeira, regularizando
        print("camada ",i)
        theta_sem_bias=np.zeros([len(D[i]),1])
        theta_sem_bias=np.concatenate((theta_sem_bias,np.delete(theta[i], 0, 1)), axis=1)
        #print(theta_sem_bias)
        P=reg_lambda*theta_sem_bias
        #print(P)
        n=len(treino)
        D[i]=(D[i]+P)/n
        print(D[i])

    for i in range(len(neunos_por_camada)-2,-1,-1):#atualiza pesos penultima camada até a primeira
        theta[i]=theta[i]-alfa*D[i]

    print("Pessos/thetas atualizados\n",theta)

    return

if __name__ == "__main__":
    main()