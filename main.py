import sys
import math
import pandas as pd
import random
import numpy as np
from multiprocessing import Pool
import time
import backpropagation as bp

def main(args):
    bp.teste()
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
    return

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 