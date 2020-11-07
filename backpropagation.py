import sys
import math
import pandas as pd
import random
import numpy as np
from multiprocessing import Pool
import time

DEBUG=0

def fun_g(x):
    '''Função sigmoide: transforma qualquer número real em um número no intervalo (0,1)'''
    return 1.0/(1.0+np.exp(-x))

def rede(x, theta):
    '''Retorna a propagação da entrada x e pesos theta em uma rede.'''
    #variaveis locais a, z e a_list
    a_list=[]
    if(DEBUG): print("entrada x: ",x)
    z=np.matrix(x)#insere valor de entrada
    n=len(theta)+1#numero de camadas da rede
    for i in range(n):
        a=np.matrix([1.0])#termo de bias
        a=np.concatenate((a,z), axis=1)
        a_list.append(a)
        if(DEBUG): print("a",i,": ",a)
        if i==n-1:
            if(DEBUG): print("-> Saida f: ", a)
            return a_list
        z=a*np.transpose(theta[i])#calcula saidas da camada
        if(DEBUG): print("z",i,": ",z)
        z=fun_g(z)

def gradiente_J_numerico(J_rede, theta, num_camadas, epsilon, n, reg_lambda):
    '''NÃO ESTÁ FUNCIONANDO'''
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

def backpropagation(treino, theta, alfa, J_rede, reg_lambda):
    '''
    Retorna theta atualizado e custo J+S.

    Para cada exemplo de treino:
    
        Chama rede para propagar a entrada, recebe valores em a_list.
        Calcula custo J_exemplo usando valor esperado e previsto.
        Atualiza J_rede.
        Calcula Delta da saída usando valor esperado e previsto, propaga da saída até segunda camada.
        Atualiza D(gradiente) do utilizando Delta e a_list.
    
    Depois de processar os dados de treino:

        Calcula D regularizado e S.
        Calcula custo da rede J+S.
        FALTA GRADIENTE NUMÉRICO, que usa epsilon.
        Atualiza theta com D e fator alfa.    
    '''
    D=[]
    num_camadas=len(theta)+1#numero de camadas da rede
    for i in range(num_camadas-1):
        D.append(np.matrix([0]))#gradiente inicial

    for n_exemplo in range(len(treino)):
        if(DEBUG):
            print("\n---------------------\nEXEMPLO ",n_exemplo)
            print(treino[n_exemplo])
        a_list= rede(treino[n_exemplo][0], theta)#, num_camadas)
        previsto=np.delete(a_list[-1], 0, 1)#deleta a primeira(0) coluna(1) do bias
        esperado=treino[n_exemplo][1]
        erro=previsto-esperado
        #print(erro)

        J_exemplo=np.sum(-np.array(esperado.tolist())*np.array(np.log(previsto).tolist())-np.array((1-esperado).tolist())*np.array(np.log(1-previsto).tolist()))
        '''
        Essa gambiarra toda é para fazer a multiplicação elemento por elemento:
        J(i) = sum( -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i))))
        (PODE TER UMA FORMA MAIS FÁCIL DE FAZER, MAS O IMPORTANTE É QUE ESSA FUNCIONA)
        Ou seja: A.*B -> np.array(A.tolist())*np.array(B.tolist())
        '''

        if(DEBUG): print("-->>>>>J: ",J_exemplo)        
        J_rede=J_rede+J_exemplo

        delta=[]
        delta.append(np.matrix(erro))
        if(DEBUG):
            print("-> Delta(erro) da ultima camada= ", delta,"\n")
            print("-> Deltas")
        for i in range(num_camadas-2,0,-1):#delta da penultima camada até a segunda
            if(DEBUG): print("camada ",i)
            x=(np.transpose(theta[i])*np.transpose(delta[0]))
            a_mod=np.array(a_list[i].tolist())*np.array((1-a_list[i]).tolist())#multiplicação por elemento
            x=np.array(np.transpose(x))*np.array(a_mod.tolist())#multiplicação por elemento
            x=np.matrix(x)
            x=np.delete(x, 0, 1)#deleta a primeira(0) coluna(1)
            if(DEBUG): print(x)
            delta.insert(0,x)
        
        delta.insert(0,x)#duplica ultimo só pra ficar alinhado, pois a primeira camada não tem delta
        
        if(DEBUG): print("\n-> Acumulando D(gradiente)")
        for i in range(num_camadas-2,-1,-1):#D da penultima camada até a primeira
            D[i]=D[i]+np.transpose(delta[i+1])*(a_list[i])
            if(DEBUG):
                print("camada ",i)
                print(D[i])

    if(DEBUG): print("\nDADOS DE TREINO PROCESSADOS\n-----------------\n\n-> Calculando D(gradiente) regularizado")
    n=len(treino)#numero de exemplos processados
    S_total=0#vai receber a soma dos quadrados de todos os thetas/pesos, MENOS OS DE BIAS
    for i in range(num_camadas-2,-1,-1):#D da penultima camada até a primeira, regularizando
        theta_sem_bias = np.concatenate((np.zeros([len(theta[i]),1]),np.delete(theta[i], 0, 1)), axis=1)
        S=np.array(theta_sem_bias.tolist())*np.array(theta_sem_bias.tolist())
        S_total=S_total+S.sum()
        P=reg_lambda*theta_sem_bias        
        D[i]=(D[i]+P)/n
        if(DEBUG):
            print("camada ",i)
            print(D[i])
        
    J_rede=J_rede/n
    S_total=(reg_lambda/(2*n))*S_total
    custo=J_rede+S_total
    ###############
    #J numerico ######NÃO FUNCIONA
    epsilon=0.0000010000
    #gradiente_J_numerico(J_rede, theta, num_camadas, epsilon, n, reg_lambda)
    ###############
    if(DEBUG): print("-> Custo regularizado J+S: ",custo)
    
    for i in range(num_camadas-2,-1,-1):#atualiza pesos penultima camada até a primeira
        theta[i]=theta[i]-alfa*D[i]

    if(DEBUG): print("\n-> Pesos/thetas atualizados\n",theta)

    return theta, custo

def teste():
    print("AQUIIIIII")
    return


def main(args):
    '''
    Chamar com: python3 backpropagation.py networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt

    DEFINIÇÃO DO TRABALHO: ./backpropagation network.txt initial_weights.txt dataset.txt
    '''
    if(len(args)<4):
        print("ESQUECEU DOS ARGUMENTOS")
        return
    
    DEBUG=0

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

    #taxa de aprendizado
    alfa=0.01#####ULTIMA LINHA DO ALG BACK, NÃO SEI O MELHOR VALOR
    
    print("-> Informações iniciais:")
    print("Neuronios em cada camada: ",neunos_por_camada)
    print("Theta inicial:\n",theta_original)
    print("Conjunto de treino:\n", treino)

    J_rede=0.0#erro da rede
    #CHAMA O BACK
    theta_atualizado, J_rede = backpropagation(treino, theta_original, alfa, J_rede, reg_lambda)

    fim=time.time()
    print("\n\nTEMPO DE EXECUÇÃO: ",fim-inicio)
    return

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 