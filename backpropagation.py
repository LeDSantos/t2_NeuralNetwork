import sys
import math
import pandas as pd
import random
import numpy as np
from multiprocessing import Pool
import time
from random import shuffle
from datetime import datetime

DEBUG=0
IMPRIME_J=1
SUB_DIR= "" #"ajuste_alpha/"

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

def predict(x, theta):
    '''Retorna previsão de saída para a entrada x em uma rede com pesos theta'''
    if(DEBUG): print("NO PREDICT\nENTRADA: ",x)
    a_list= rede(x, theta)#, num_camadas)

    previsto=np.delete(a_list[-1], 0, 1)#deleta a primeira(0) coluna(1) do bias
    return previsto


def chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def str_date_time():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
    return dt_string


def backpropagation(treino, theta, alfa, reg_lambda, estrutura_rede, K, EXECUTA_UMA_VEZ, batch_size ):
    '''
    Retorna theta e custo J+S FINAIS.

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

    VOLTA PARA O INICIO SE ACONTECER GRANDE MUDANÇA NO CUSTO  
    '''

    D=[]
    num_camadas=len(theta)+1#numero de camadas da rede
    for i in range(num_camadas-1):
        D.append(np.matrix([0]))#gradiente inicial

    if not batch_size:
        batch_size = len(treino)

    if (IMPRIME_J):
        nome_arq= SUB_DIR+"custoJrede"+str_date_time()+"_"+str(alfa)+"_"+str(reg_lambda)+"_"+str(batch_size)+"_"+str(estrutura_rede)+"fold"+str(K)+".txt"
        #print(nome_arq)
        #arq_J=open(nome_arq,"w")

    if(DEBUG):
        saida = open("saida_backprop_rede_"+str(estrutura_rede)+".txt", 'w')
        #saida.write("Pesos / Gradiente\n")

    shuffle(treino)

    custo=0
    #custo_ant=10
    interacoes=0
    custo_medio =0
    custo_medio_list = []
    custo_medio_ant =10
    mini_batch_list = list(chunk(treino, batch_size))
    variancia = 10

#    while abs(custo_medio - custo_medio_ant) > 0.0001 and interacoes<500:#repete o back até 500 vezes
    while (interacoes < 2 or (variancia > 0.001 or abs(custo_medio - custo_medio_ant) > 0.00005)  ) and interacoes<1000:#repete o back até 500 vezes
        if(DEBUG): print(interacoes)
        #batch_treino = treino[0:batch_size]
        custo_batch = []
        shuffle(mini_batch_list)
        for batch_treino in mini_batch_list:
            J_rede = 0 
            for n_exemplo in range(len(batch_treino)):
                if(DEBUG):
                    print("\n---------------------\nEXEMPLO ",n_exemplo)
                    print(batch_treino[n_exemplo])
                a_list = rede(batch_treino[n_exemplo][0], theta)#, num_camadas)
                previsto = np.delete(a_list[-1], 0, 1)#deleta a primeira(0) coluna(1) do bias
                esperado = batch_treino[n_exemplo][1]
                erro = previsto - esperado

                J_exemplo = np.sum(-np.multiply( esperado,     np.log(previsto))
                                   -np.multiply( (1-esperado), np.log(1-previsto)))
                #J_exemplo = np.sum(-np.array(esperado.tolist())*np.array(np.log(previsto).tolist())-np.array((1-esperado).tolist())*np.array(np.log(1-previsto).tolist()))
                #'''
                #Essa gambiarra toda é para fazer a multiplicação elemento por elemento:
                #J(i) = sum( -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i))))
                #(PODE TER UMA FORMA MAIS FÁCIL DE FAZER, MAS O IMPORTANTE É QUE ESSA FUNCIONA)
                #Ou seja: A.*B -> np.array(A.tolist())*np.array(B.tolist())
                #'''

                if(DEBUG): print("-->>>>>J: ",J_exemplo)  
                J_rede = J_rede + J_exemplo    

                delta=[]
                delta.append(np.matrix(erro)) #delta da camada de saida
                if(DEBUG):
                    print("-> Delta(erro) da ultima camada= ", delta,"\n")
                    print("-> Deltas")
                for i in range(num_camadas-2,0,-1):#delta da penultima camada até a segunda
                    if(DEBUG): print("camada ",i)
                    x=(np.transpose(theta[i])*np.transpose(delta[0]))
                    #a_mod = np.array(a_list[i].tolist())*np.array((1-a_list[i]).tolist())#multiplicação por elemento
                    a_mod = np.multiply(a_list[i], 1-a_list[i])
                    #x1 = np.array(np.transpose(x))*np.array(a_mod.tolist())#multiplicação por elemento
                    x = np.multiply(np.transpose(x), a_mod)#multiplicação por elemento

                    x = np.matrix(x)
                    x = np.delete(x, 0, 1)#deleta a primeira(0) coluna(1)
                    if(DEBUG): print(x)
                    delta.insert(0,x)
                
                delta.insert(0,x)#duplica ultimo só pra ficar alinhado, pois a primeira camada não tem delta
                
                if(DEBUG): print("\n-> Acumulando D(gradiente)")
                for i in range(num_camadas-2,-1,-1):#D da penultima camada até a primeira
                    D[i] = D[i] + np.transpose(delta[i+1])*(a_list[i])
                    if(DEBUG):
                        print("camada ",i)
                        print(D[i])

            if(DEBUG): print("\nDADOS DE TREINO PROCESSADOS\n-----------------\n\n-> Calculando D(gradiente) regularizado")
            n = len(batch_treino)#numero de exemplos processados
            S_total=0#vai receber a soma dos quadrados de todos os thetas/pesos, MENOS OS DE BIAS
            for i in range(num_camadas-2,-1,-1):#D da penultima camada até a primeira, regularizando
                theta_sem_bias = np.concatenate((np.zeros([len(theta[i]),1]),np.delete(theta[i], 0, 1)), axis=1)
                #S = np.array(theta_sem_bias.tolist())*np.array(theta_sem_bias.tolist())
                S = np.multiply(theta_sem_bias, theta_sem_bias)

                S_total += S.sum()
                P = reg_lambda*theta_sem_bias        
                D[i] = (D[i]+P)/n
                if(DEBUG):
                    print("camada ",i)
                    print(D[i])
                
            J_rede = J_rede/n
            S_total = (reg_lambda/(2*n))*S_total
            #custo_ant = custo
            custo = J_rede+S_total

            custo_batch.append(custo)

            if(DEBUG): print("-> Custo regularizado J+S: ",custo)
            #if(IMPRIME_J):
            #        #print(J_rede)
            #        arq_J.write("{:.5f}\n".format(custo))


            for i in range(num_camadas-1):#atualiza pesos
                theta[i]=theta[i]-alfa*D[i]
                if(DEBUG):
                    #for linha in range(len(theta[i])):
                    #    theta[i][linha].tofile(saida,sep=", ",format='%.5f')
                    #    saida.write("; ")
                    #saida.write("/")
                    for linha in range(len(D[i])):
                        D[i][linha].tofile(saida,sep=", ",format='%.5f')
                        saida.write("; ")
                    saida.write("\n")

            if(DEBUG):
                print("\n-> Pesos/thetas atualizados\n",theta,"\n Informações no arquivo saida_backprop_rede_"+str(estrutura_rede)+".txt")

        custo_medio_ant = custo_medio
        custo_medio_list.append(np.average(custo_batch))
        custo_medio =  np.average(custo_medio_list[-8:])
        variancia = np.var(custo_medio_list[-8:])
        if(IMPRIME_J):
            #print(J_rede)
            with open(nome_arq, "a") as arq_j:
                arq_j.write("{:.6f}\n".format(custo_medio_list[-1]))

        #print(custo_medio, custo_medio_ant, abs(custo_medio-custo_medio_ant))
        interacoes += 1
        if(EXECUTA_UMA_VEZ): break
    
    if(DEBUG):        
        saida.write("reg_lambda: "+ str(reg_lambda))
        saida.close()
    
    #if(IMPRIME_J):  arq_J.close()
    print("Fold",K,":",interacoes," interações para convergir")
    return theta, custo

def main(args):
    '''
    Chamar com: python3 backpropagation.py networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt

    DEFINIÇÃO DO TRABALHO: ./backpropagation network.txt initial_weights.txt dataset.txt
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

    #taxa de aprendizado
    alfa=0.01#####ULTIMA LINHA DO ALG BACK, NÃO SEI O MELHOR VALOR
    
    print("-> Informações iniciais:")
    print("Neuronios em cada camada: ",neunos_por_camada)
    print("Theta inicial:\n",theta_original)
    print("Conjunto de treino:\n", treino)

    J_rede=0.0#erro da rede
    #CHAMA O BACK
    theta_atualizado, J_rede = backpropagation(treino, theta_original, alfa, reg_lambda, neunos_por_camada,0,1, None)

    fim=time.time()
    print("\n\nTEMPO DE EXECUÇÃO: ",fim-inicio)
    return

if __name__ == '__main__':
    DEBUG=1
    sys.exit(main(sys.argv)) 
