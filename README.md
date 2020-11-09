# t2_NeuralNetwork
Redes Neurais - Backpropagation - Trabalho 2 de Aprendizado de Máquina\
Objetivo: implementação de uma rede neural treinada via backpropagation.

## Executar rede
python3 main.py\
python3 main.py vinho \
python3 main.py votos
- Rede, Alfa, Reg_lambda, Acuracia, desvio_padrao e tempo_exe para cada configuração de rede impressos no terminal.
- **SE IMPRIME_J(em backpropagation.py)==1:** J de cada exemplo treinado em **custoJrede_[estrutura da rede]foldK.txt**

## Executar verificação do backpropagation
python3 backpropagation.py networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt\
python3 backpropagation.py networkEXP2.txt initial_weightsEXP2.txt datasetEXP2.txt\
python3 backpropagation.py networkEXP2mod.txt initial_weightsEXP2mod.txt datasetEXP2mod.txt
- Detalhes do funcionamento impressos no terminal.
- Pesos, gradiente e fator de regularização lambda em: **saida_backprop_rede_[estrutura da rede].txt** 

## Executar gradiente numérico
python3 gradiente_numerico.py networkEXP1.txt initial_weightsEXP1.txt datasetEXP1.txt\
python3 gradiente_numerico.py networkEXP2.txt initial_weightsEXP2.txt datasetEXP2.txt\
python3 gradiente_numerico.py networkEXP2mod.txt initial_weightsEXP2mod.txt datasetEXP2mod.txt
- Detalhes do funcionamento impressos no terminal.
- Gradiente e fator de regularização lambda em: **saida_grad_num_rede_[estrutura da rede].txt**
