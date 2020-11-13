import sys
import math
import pandas as pd
import random
import numpy as np
from multiprocessing import Pool
import time
import backpropagation as bp

DEBUG=0

def J_numerico(treino, theta, reg_lambda):
    J_rede=0.0
    num_camadas=len(theta)+1#numero de camadas da rede
    for n_exemplo in range(len(treino)):
        if(DEBUG):
            print("\n---------------------\nEXEMPLO ",n_exemplo)
            print(treino[n_exemplo])
        a_list= bp.rede(treino[n_exemplo][0], theta)#, num_camadas)
        previsto=np.delete(a_list[-1], 0, 1)#deleta a primeira(0) coluna(1) do bias
        esperado=treino[n_exemplo][1]
        erro=previsto-esperado

        J_exemplo=np.sum(-np.array(esperado.tolist())*np.array(np.log(previsto).tolist())-np.array((1-esperado).tolist())*np.array(np.log(1-previsto).tolist()))
        '''
        Essa gambiarra toda é para fazer a multiplicação elemento por elemento:
        J(i) = sum( -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i))))
        (PODE TER UMA FORMA MAIS FÁCIL DE FAZER, MAS O IMPORTANTE É QUE ESSA FUNCIONA)
        Ou seja: A.*B -> np.array(A.tolist())*np.array(B.tolist())
        '''

        if(DEBUG): print("-->>>>>J: ",J_exemplo)  
        J_rede=J_rede+J_exemplo    

    n=len(treino)#numero de exemplos processados
    S_total=0#vai receber a soma dos quadrados de todos os thetas/pesos, MENOS OS DE BIAS
    for i in range(num_camadas-2,-1,-1):#D da penultima camada até a primeira, regularizando
        theta_sem_bias = np.concatenate((np.zeros([len(theta[i]),1]),np.delete(theta[i], 0, 1)), axis=1)
        S=np.array(theta_sem_bias.tolist())*np.array(theta_sem_bias.tolist())
        S_total=S_total+S.sum()

    J_rede=J_rede/n
    S_total=(reg_lambda/(2*n))*S_total
    custo=np.longdouble(J_rede+S_total)

    if(DEBUG): print("-> Custo regularizado J+S: ",custo)
    return custo


def main(args):
    '''
    Chamar com: python3 gradiente_numerico.py networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt

    DEFINIÇÃO DO TRABALHO: ./gradiente_numerico network.txt initial_weights.txt dataset.txt
    '''
    if(len(args)<4):
        print("ESQUECEU DOS ARGUMENTOS")
        return
    
    inicio=time.time()
    #TRATAMENTO DOS ARQUIVOS DE ENTRADA
    print("Infos network: ",args[1])#contém lambda de regularização e número de neuronios por camada
    infos = open(args[1], 'r')
    reg_lambda=float(infos.readline())
    neunos_por_camada=infos.readlines()
    num_camadas=len(neunos_por_camada)
    for i in range(num_camadas):
        neunos_por_camada[i]=int(neunos_por_camada[i])

    print("Pesos iniciais: ",args[2])#contém thetas iniciais
    arq_theta = open(args[2], 'r')
    theta_original=[]
    for i in range(num_camadas-1):
        theta_original.append(np.matrix(arq_theta.readline()))
    #print(theta_original)
    
    print("Dataset: ",args[3])#contém conjunto de treinamento, com entradas e saídas
    arq_treino_inicial = open(args[3], 'r')
    treino=[]
    novalinha=arq_treino_inicial.readline()
    while (novalinha != ""):
        exemplo=novalinha.split(";")

        entradas=exemplo[0].split(",")
        for i in range(len(entradas)):
            entradas[i]=float(entradas[i])
        entradas=np.array(entradas)

        saidas=exemplo[1].split(",")
        for i in range(len(saidas)):
            saidas[i]=float(saidas[i])
        saidas=np.array(saidas)

        treino.append([entradas,saidas])
        novalinha=arq_treino_inicial.readline()        

  
    print("-> Informações iniciais:")
    print("Neuronios em cada camada: ",neunos_por_camada)
    print("Theta inicial:\n",theta_original)
    print("Conjunto de treino:\n", treino)

    arq_saida=open("saida_grad_num_rede_"+str(neunos_por_camada)+".txt", "w")

    epsilon=0.000001

    arq_saida.write("Gradiente numerico com epsilon "+str(epsilon)+"\n")
    for i in range(len(theta_original)):
        for j in range(len(theta_original[i])):
            for k in range(len(theta_original[i][j])):
                for l in range(len(theta_original[i][j][k].tolist()[0])):
                    theta_original[i][j][k,l]=theta_original[i][j][k,l]+epsilon
                    Jmaior=J_numerico(treino,theta_original, reg_lambda)
                    theta_original[i][j][k,l]=theta_original[i][j][k,l]-2*epsilon
                    Jmenor=J_numerico(treino,theta_original, reg_lambda)

                    gradiente=(Jmaior-Jmenor)/(2*epsilon)
                    arq_saida.write("{:.5f}".format(gradiente))
                    if(l != len(theta_original[i][j][k].tolist()[0])-1): arq_saida.write(", ")
                arq_saida.write("; ")
        arq_saida.write("\n")

    print("Informações no arquivo saida_grad_num_rede_"+str(neunos_por_camada)+".txt")

    arq_saida.write("reg_lambda: "+ str(reg_lambda))
    arq_saida.close()
    return

if __name__ == '__main__':
    DEBUG=0
    sys.exit(main(sys.argv)) 
