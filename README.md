# t2_NeuralNetwork
Redes Neurais - Backpropagation - Trabalho 2 de Aprendizado de Máquina\
Objetivo: implementação de uma rede neural treinada via backpropagation.

Implementação foi realizada utilizando linguagem de programação python. Para se manipular o dataset se usou a biblioteca pandas.

## Execução do programa:

```prompt
$ python3 main.py [votos | vinho]
```

Onde o argumento votos corresponde ao dataset 1984 United States Congressional Voting e vinho corresponde ao Wine Data Set.

O programa gera uma tabela na saída padrão com as informações de:

**Rede; Alfa; Reg_lambda; batch_size; Acuracia; desvio_padrao; tempo_exe**

## Teste do backpropagation:

```prompt
$ python3 backpropagation.py [networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt | networkEXP2.txt initial_weightsEXP2.txt datasetEXP2.txt]
```

Saída com gradiente e alfa em **saida_backprop_rede_[estrutura da rede].txt**

## Teste do gradiente numérico:

```prompt
$ python3 gradiente_numerico.py [networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt | networkEXP2.txt initial_weightsEXP2.txt datasetEXP2.txt]
```

Saída com gradiente numérico e alfa em **saida_grad_num_rede_[estrutura da rede].txt**
