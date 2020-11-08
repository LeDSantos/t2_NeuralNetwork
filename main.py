import sys
import math
import pandas as pd
import random
import numpy as np
from multiprocessing import Pool
import time
import backpropagation as bp

dataset = {}
dataset["votos"] = {\
    "data": ('house-votes-84.tsv', '\t'), \
    "types": ('house-votes-84_types.csv', ';')}
dataset["vinho"] = {\
    "data":  ('wine-recognition.tsv','\t'), \
    "types": ('wine-recognition_types.csv', ';')}

#def treina_e_testa(treino, teste, target_coluna, n_arvores):
def treina_e_testa(args):
    treino , teste, target_coluna, config_rede = args

    #INICIANDO THETAS COM VALORES ALEATÓRIOS
    theta=[]
    #valor_padrao=0.5
    #print(config_rede)np.random.rand(3,2)
    for i in range(len(config_rede)-1):
        theta.append(np.matrix(np.random.rand(config_rede[i+1], config_rede[i]+1)))
    
    #print(theta)

    #matrix_treino=df_train.to_numpy()
    #print(matrix_treino)
    #previsto=np.delete(a_list[-1], 0, 1)#deleta a primeira(0) coluna(1) do bias


    matrix_treino=np.matrix(treino.to_numpy())
    #print(matrix_treino)
    treino_organizado=[]
    for i in range(len(treino)):
        #print("deletar coluna ",target_coluna)
        #print("exemplo ",i,": ",matrix_treino[i])
        #print("saida: ",matrix_treino[i,target_coluna])
        entradas=np.delete(matrix_treino[i], target_coluna, 1)
        #print(entradas)

        saida=matrix_treino[i,target_coluna]
        #saida=0.5
        #config_rede[-1]=3
        lista_saidas=[]#cria uma lista com as classes de saida possíveis
        for j in range(config_rede[-1]):
            lista_saidas.append(0.0)
        
        lista_saidas[int(saida*(config_rede[-1]-1))]=1.0#coloca 1 na classe esperada
        treino_organizado.append([ np.array(entradas), np.array(lista_saidas)])
        #print("AQUIIIIIII")

    #print(treino_organizado)
    matrix_teste=np.matrix(teste.to_numpy())
    #print(matrix_treino)
    teste_organizado=[]
    for i in range(len(teste)):
        #print("deletar coluna ",target_coluna)
        #print("exemplo ",i,": ",matrix_treino[i])
        #print("saida: ",matrix_treino[i,target_coluna])
        entradas=np.delete(matrix_teste[i], target_coluna, 1)
        #print(entradas)

        #saida=matrix_treino[i,target_coluna]
        #saida=0.5
        #config_rede[-1]=3
        #lista_saidas=[]#cria uma lista com as classes de saida possíveis
        #for j in range(config_rede[-1]):
        #    lista_saidas.append(0.0)
        
        #lista_saidas[int(saida*(config_rede[-1]-1))]=1.0#coloca 1 na classe esperada
        teste_organizado.append([ np.array(entradas), matrix_teste[i,target_coluna]])

    alfa=0.00001
    reg_lambda=0.250
    print("NO BACK")
    theta_modelo, custoJ_S = bp.backpropagation(treino_organizado,theta,alfa,0,reg_lambda) #TEM Q ARRUMAR FlorestaAleatoria(treino, target_coluna, n_arvores); #parametro n_arvores
    print("novo theta")
    print(theta_modelo)
    
    #np.delete(np.matrix(treino.to_numpy()), -1, 1)
    key_list = list(treino.columns)
    target_name= key_list[target_coluna]

    #inicializa a matriz de confusao
    table_of_confusion = {'CERTO':0,'ERRADO':0}

    nrow = 0
    for test_row in teste_organizado:
        resp= bp.predict(test_row[0], theta_modelo)
        #print("[{}/{}]{:0.2f}% complete...".format(i+1, K,100*(nrow+1)/len(test.index) ), end='\r')
        nrow += 1
        #print("EXEMPLO: ",test_row)
        print("RESPOSTA DO PREDICT")
        print(resp)
        #result = numpy.where(arr == numpy.amax(arr))
        localizado=np.where(resp == np.max(resp))
        predito=localizado[1][0]
        print("maior em ",predito)
        esperado=int(test_row[1]*(config_rede[-1]-1))
        print("esperado ",esperado)
        #return
        #atualiza a matriz de confusao
        if predito == esperado:
            table_of_confusion['CERTO'] += 1
        else:
            table_of_confusion['ERRADO'] += 1

    return table_of_confusion


def cross_validation(df, target, config_rede, K):
    target_classes = df[target].value_counts() # TODO: unused var
    #dividir o dataset nas classes do atributo alvo
    df_list= []

    for name, df in df.groupby(target):
        df_list.append(df)

#    print("numero de instancias por classe: ", [len(x) for x in df_list])
    #tamanho de cada fold para negativos e positivos
    fold_size_per_target = list(map(lambda x: round(len(x.index)/K), df_list))

#    print(fold_size_per_target)

    #divide em K folds
    fold_list_per_target=[]

    for df, fold_size in zip(df_list, fold_size_per_target):
        fold_class_list =[]
        for i in range(0,K):
            if(i==K-1):
                fold_class_list.append(df[fold_size*i :])
            else:
                fold_class_list.append(df[fold_size*i : fold_size*(i+1)])

        fold_list_per_target.append(fold_class_list)


    #junta os K folds em uma lista unica, contendo folds estratificados
    fold_list= []
    for fold in zip(*fold_list_per_target):
        fold_list.append(pd.concat(list(fold), axis=0))

#    print(fold_list)

    target_coluna = list(df.columns).index(target)

    
    train = [None]*K
    test = [None]*K
    #monta os folds de treino e teste
    for i in range(K):
        train[i] = pd.concat(fold_list[:i] + fold_list[i+1:], axis=0).copy()
        test[i] = fold_list[i].copy()

    #roda os treinos em paralelo
    with Pool(processes=1) as pool:
        #encapsula os argumentos
        arg_list = (zip(train, test, [target_coluna]*K, [config_rede]*K))#[]*K é para ir uma copia igual para cada processo
        result_list = pool.map(treina_e_testa, arg_list)

#    print(result_list)

    #calculo das metricas
    acc = [None]*len(result_list)
    for i in range (len(result_list)):
        total = result_list[i]['CERTO'] + result_list[i]['ERRADO']
        acc[i] = result_list[i]['CERTO']/total

    acuracia_media = np.average(acc)
    desvio_padrao = np.std(acc)

    return config_rede, acuracia_media, desvio_padrao


def main():
    #bp.teste()
    '''
    PARA RODAR O BACK COM OS DATASETS GRANDES:
    precisa processar os arquivos de entrada com panda para extrair:
    treino

    Definir a estrutura(quantas camadas e quantos neuronios por camada)
    DE ACORDO COM O TAMANHO DA ESTRUTURA(+ BIAS): theta inicia com valores aleatórios
    reg_lambda a gente define;
    alfa(taxa de aprendizado) a gente define;
    J_rede inicia com 0
    '''
    random.seed(10)
    np.random.seed(10)

    if len(sys.argv) > 1 and sys.argv[1] in dataset.keys():
        ds = dataset[sys.argv[1]]
    else: 
        ds = dataset["votos"] # default dataset 

    df_train = pd.read_csv(ds["data"][0], sep=ds["data"][1])
    df_train_attribute = pd.read_csv(ds["types"][0], sep=ds["types"][1])

    key_list=[]
    type_list=[]
    for i in range(df_train_attribute.shape[0]):
        print(df_train_attribute.values[i])
        key_list.append(df_train_attribute.values[i][0])
        type_list.append(df_train_attribute.values[i][1])

    print(key_list)
    
    target_attribute = key_list[-1]

    print(target_attribute)

    attr_type_dict = dict(zip(key_list, type_list))
    df_train = df_train.astype(attr_type_dict)
    print(df_train.dtypes)

    print(df_train)
    category_columns = df_train.select_dtypes(['category']).columns
    df_train[category_columns] = df_train[category_columns].apply(lambda x: x.cat.codes)
    print(df_train.dtypes)
    print(df_train)

    #normalização
    df_train=((df_train-df_train.min())/(df_train.max()-df_train.min()))
    print(df_train)

    #ORGANIZAÇÃO DA REDE
    neunos_por_camada=[]
    neunos_por_camada.append(len(df_train.dtypes)-1)#neuros iniciais

    neuros_ocultos=[5, 5]#[2, 4, 3, 2]

    neunos_por_camada=neunos_por_camada+neuros_ocultos

    neunos_por_camada.append(len(df_train[target_attribute].unique()))#neuros de saida, um para cada categoria
    print("neuronios por camada: ", neunos_por_camada)
    n_camadas=len(neunos_por_camada)

    print("Rede; Acuracia; desvio_padrao; tempo_exe")
    #for n in valores_de_teste:
    ini = time.time()
    config_rede = neunos_por_camada
    n, acc, stdev = cross_validation(df_train, target_attribute, config_rede, 10)
    fim = time.time()
    print("Rede; Acuracia; desvio_padrao; tempo_exe")
    print("{}; {}; {}; {}".format(n, acc, stdev, fim-ini))

    return

if __name__ == "__main__" :
    main()