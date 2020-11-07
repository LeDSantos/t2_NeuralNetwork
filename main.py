import sys
import math
import pandas as pd
import random
import numpy as np
from multiprocessing import Pool
import time



#função sigmoide
def fun_g(x):
    return 1.0/(1.0+np.exp(-x))

#calcula a previsao da rede
def rede(x, theta, num_camadas):
    #variaveis locais a, z e a_list
    a_list=[]
    print("entrada x: ",x)
    z=np.matrix(x)#insere valor de entrada
    for i in range(num_camadas):
        a=np.matrix([1.0])#termo de bias
        a=np.concatenate((a,z), axis=1)
        a_list.append(a)
        print("a",i,": ",a)
        if i==num_camadas-1:
            print("-> Saida f: ", a)
            return a_list
        z=a*np.transpose(theta[i])#calcula saidas da camada
        print("z",i,": ",z)
        z=fun_g(z)

def gradiente_J_numerico(J_rede, theta, num_camadas, epsilon, n, reg_lambda):
    print("GRADIENTE NUMERICO")
    S_total=0
    
    for n_camada in range(num_camadas-2,-1,-1):#S da penultima camada até a primeira, regularizando
        theta_mod=theta
        print(theta_mod[n_camada])
        theta_mod[n_camada]=theta_mod[n_camada]+epsilon
        print("SOMA")
        print(theta_mod[n_camada])
        for i in range(num_camadas-2,-1,-1):#S da penultima camada até a primeira, regularizando
            print("camada ",i)
            theta_sem_bias=np.concatenate((np.zeros([len(theta_mod[i]),1]),np.delete(theta_mod[i], 0, 1)), axis=1)
            S=np.array(theta_sem_bias.tolist())*np.array(theta_sem_bias.tolist())
            S_total=S_total+S.sum()
            print(S_total)
        #J_rede=J_rede/n #já ta dividido
        S_total=(reg_lambda/(2*n))*S_total
        custo_mais_ep=J_rede+S_total
        print("CUSTO + epsilon: ", custo_mais_ep)

        theta_mod=theta
        print(theta_mod[n_camada])
        theta_mod[n_camada]=theta_mod[n_camada]-epsilon
        print("MENOS")
        print(theta_mod[n_camada])
        for i in range(num_camadas-2,-1,-1):#S da penultima camada até a primeira, regularizando
            print("camada ",i)
            theta_sem_bias=np.concatenate((np.zeros([len(theta_mod[i]),1]),np.delete(theta_mod[i], 0, 1)), axis=1)
            S=np.array(theta_sem_bias.tolist())*np.array(theta_sem_bias.tolist())
            S_total=S_total+S.sum()
            print(S_total)
        #J_rede=J_rede/n #já ta dividido
        S_total=(reg_lambda/(2*n))*S_total
        custo_menos_ep=J_rede+S_total
        print("CUSTO - epsilon: ", custo_menos_ep)
    
        grad_num=(custo_mais_ep-custo_menos_ep)/(2*epsilon)
        print(">>>>>>>>>>>>>>>>>>Grad numerico da camada ",n_camada,":",grad_num)
    
    return

#chama rede e propaga o erro encontrado quando a previsao é comparada com o valor esperado
def backpropagation(treino, num_camadas, theta, alfa, J_rede, reg_lambda):
    D=[]
    for i in range(num_camadas-1):
        D.append(np.matrix([0]))#gradiente inicial

    for n_exemplo in range(len(treino)):#depois dá de mudar isso
        print("\n---------------------\nEXEMPLO ",n_exemplo)
        print(treino[n_exemplo])
        a_list= rede(treino[n_exemplo][0], theta, num_camadas)
        previsto=a_list[-1]
        print(previsto)
        previsto=np.delete(previsto, 0, 1)#deleta a primeira(0) coluna(1)
        print(previsto)
        esperado=treino[n_exemplo][1]
        print(esperado)
        erro=previsto-esperado# previsto - esperado
        print(erro)
        delta=[]
        delta.append(np.matrix(erro))

        J_exemplo=np.sum(-np.array(esperado.tolist())*np.array(np.log(previsto).tolist())-np.array((1-esperado).tolist())*np.array(np.log(1-previsto).tolist()))
        print("-->>>>>J: ",J_exemplo)
        
        J_rede=J_rede+J_exemplo

        print("-> Delta da ultima camada= ", delta,"\n")
        print("-> Deltas")
        for i in range(num_camadas-2,0,-1):#delta da penultima camada até a segunda
            print("camada ",i)
            x=(np.transpose(theta[i])*np.transpose(delta[0]))
            print(x)
            a_mod=np.array(a_list[i].tolist())*np.array((1-a_list[i]).tolist())
            print(a_mod)
            x=np.array(np.transpose(x))*np.array(a_mod.tolist())
            x=np.matrix(x)
            x=np.delete(x, 0, 1)#deleta a primeira(0) coluna(1)
            print(x)
            delta.insert(0,x)
        
        delta.insert(0,x)#duplica ultimo só pra ficar alinhado
        
        print("\n-> Calculando D")
        for i in range(num_camadas-2,-1,-1):#D da penultima camada até a primeira
            print("camada ",i)
            D[i]=D[i]+np.transpose(delta[i+1])*(a_list[i])
            print(D[i])

    print("\nDADOS DE TREINO PROCESSADOS\n-----------------\n\n-> Calculando D regularizado")
    n=len(treino)
    S_total=0
    for i in range(num_camadas-2,-1,-1):#D da penultima camada até a primeira, regularizando
        print("camada ",i)
        theta_sem_bias=np.concatenate((np.zeros([len(theta[i]),1]),np.delete(theta[i], 0, 1)), axis=1)
        S=np.array(theta_sem_bias.tolist())*np.array(theta_sem_bias.tolist())
        S_total=S_total+S.sum()
        #print(S)
        P=reg_lambda*theta_sem_bias        
        D[i]=(D[i]+P)/n
        print(D[i])
    
    #print("\n-> Calculando J regularizado")
    J_rede=J_rede/n
    S_total=(reg_lambda/(2*n))*S_total
    #S=np.array(theta.tolist())*np.array(theta.tolist())#*np.array((1-a_list[i]).tolist())
    #print(S)
    custo=J_rede+S_total
    ###############
    #J numerico ######NÃO FUNCIONA
    epsilon=0.0000010000
    #gradiente_J_numerico(J_rede, theta, num_camadas, epsilon, n, reg_lambda)
    ###############
    print("-> Custo regularizado J+S: ",custo)
    
    for i in range(num_camadas-2,-1,-1):#atualiza pesos penultima camada até a primeira
        theta[i]=theta[i]-alfa*D[i]

    print("\n-> Pesos/thetas atualizados\n",theta)

    return theta, custo#atualizado


def main(args):#chamar com python3 main.py networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt
    print("Infos network: ",args[1])
    infos = open(args[1], 'r')
    reg_lambda=float(infos.readline())
    #n_camadas=sum(1 for line in infos)
    neunos_por_camada=infos.readlines()
    #for linha in infos:
    #    neunos_por_camada.append(int(infos.readline()))
    #    print(neunos_por_camada[-1])
    num_camadas=len(neunos_por_camada)
    for i in range(num_camadas):
        neunos_por_camada[i]=int(neunos_por_camada[i])
    #print(neunos_por_camada)

    print("Pesos iniciais: ",args[2])
    arq_theta_inicial = open(args[2], 'r')

    theta_original=[]
    for i in range(num_camadas-1):
        theta_original.append(np.matrix(arq_theta_inicial.readline()))

    print(theta_original)
    
    print("Dataset: ",args[3])
    arq_treino_inicial = open(args[3], 'r')
    treino=[]
    novalinha=arq_treino_inicial.readline()
    while (novalinha != ""):
        treino.append(np.matrix(novalinha))
        novalinha=arq_treino_inicial.readline()
        
    print(treino)
    
    #./backpropagation network.txt initial_weights.txt dataset.txt

    #usando exemplo_backprop_rede1.txt
    #utiliza 2 exemplos

    #taxa de aprendizado
    alfa=0.01#####ULTIMA LINHA DO ALG BACK, NÃO SEI O MELHOR VALOR
    
    print("-> Informações iniciais:")
    print("Neuronios em cada camada: ",neunos_por_camada)
    print("Theta inicial:\n",theta_original)
    print("Conjunto de treino:\n", treino)

    J_rede=0.0
    theta_atualizado, J_rede=backpropagation(treino, num_camadas, theta_original, alfa, J_rede, reg_lambda)

    return

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 