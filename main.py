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
theta_original=[]
theta1=np.matrix([[0.40000 , 0.10000  ], [0.30000 , 0.20000 ]])
theta2=np.matrix([0.70000,  0.50000,  0.60000 ])
theta_original.append(theta1)
theta_original.append(theta2)
treino_original=	np.matrix([[0.13000, 0.90000], [0.42000, 0.23000]])
#treino x->y

#função sigmoide
def fun_g(x):
    return 1.0/(1.0+np.exp(-x))

#calcula a previsao da rede
def rede(x, theta, num_camadas):
    #variaveis locais a, z e a_list
    a_list=[]
    print("entrada x: ",x)
    z=np.matrix(x)#insere valor de entrada
    for i in range(len(neunos_por_camada)):
        a=np.matrix([1.0])#termo de bias
        a=np.concatenate((a,z), axis=1)
        a_list.append(a)
        print("a",i,": ",a)
        if i==num_camadas-1:
            print("-> Saida f: ", a[0,1])
            return a_list
        z=a*np.transpose(theta[i])#calcula saidas da camada
        print("z",i,": ",z)
        z=fun_g(z)

#chama rede e propaga o erro encontrado quando a previsao é comparada com o valor esperado
def backpropagation(treino, num_camadas, theta, alfa):
    D=[]
    for i in range(num_camadas-1):
        D.append(np.matrix([0]))#gradiente inicial

    for n_exemplo in range(len(treino)):#depois dá de mudar isso
        print("\n---------------------\nEXEMPLO ",n_exemplo)
        a_list= rede(treino[n_exemplo,0], theta, num_camadas)
        erro=a_list[-1][0,1]-treino[n_exemplo,1]# previsto - esperado
        delta=[]
        delta.append(np.matrix([erro]))
        print("-> Delta da ultima camada= ", delta,"\n")
        print("-> Deltas")
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
        
        print("\n-> Calculando D")
        for i in range(len(neunos_por_camada)-2,-1,-1):#D da penultima camada até a primeira
            print("camada ",i)
            D[i]=D[i]+np.transpose(delta[i+1])*(a_list[i])
            print(D[i])

    print("\nDADOS DE TREINO PROCESSADOS\n-----------------\n\n-> Calculando D regularizado")
    for i in range(len(neunos_por_camada)-2,-1,-1):#D da penultima camada até a primeira, regularizando
        print("camada ",i)
        theta_sem_bias=np.concatenate((np.zeros([len(D[i]),1]),np.delete(theta[i], 0, 1)), axis=1)
        P=reg_lambda*theta_sem_bias
        n=len(treino)
        D[i]=(D[i]+P)/n
        print(D[i])

    for i in range(len(neunos_por_camada)-2,-1,-1):#atualiza pesos penultima camada até a primeira
        theta[i]=theta[i]-alfa*D[i]

    print("\n-> Pesos/thetas atualizados\n",theta)

    return theta#atualizado


def main():
    #usando exemplo_backprop_rede1.txt
    #utiliza 2 exemplos
    alfa=0.01#####ULTIMA LINHA DO ALG BACK, NÃO SEI O MELHOR VALOR
    
    print("-> Informações iniciais:")
    print("Neuronios em cada camada: ",neunos_por_camada)
    print("Theta inicial:\n",theta_original)
    print("Conjunto de treino:\n", treino_original)

    theta_atualizado=backpropagation(treino_original, len(neunos_por_camada), theta_original, alfa)

    return

if __name__ == "__main__":
    main()