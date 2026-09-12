# preprocessing

Pré-processamento de dados de RRi para cálculo da HRV.

Para executar o código, inicie instalando as dependências necessárias:

    pip install -r requirements.txt

Em seguida, garanta que os dados estejam organizados na seguinte estrutura ou altere os caminhos no arquivo `src/config.py`:

```plaintext
vfc-diabeticos/
├── data/
│   ├── control/
│   └── diabetic/
├── src/
│   ├── main.py
│   └── ...
├── README.md
└── requirements.txt
```


**A metodologia adotada pode ser resumida em 4 etapas:**

1. Descarte das 10 primeiras entradas de RRi de cada arquivo (paciente);

2. Avaliação da estabilidade dos sinais de RRi e descarte dos arquivos que não atenderem
   o limiar estabelecido (90%). A estabilidade do sinal é calculada a partir da:
    - Detecção de Outliers (RRi < 300 ou RRi > 2000);
    - Detecção de Batimentos Ectópicos (RRi+1/RRi não pode variar mais que 20%, para mais ou para menos);

3. Transformação dos sinais de RRi em NNi (com 3 casas decimais):
    - Substituindo os Outliers por meio da interpolação linear;
    - Substituindo os Batimentos Ectópicos por meio da interpolação linear;
    
4. Truncamento dos arquivos de NNi, mantendo a parte inicial, em função de
   um determinado valor de tempo (não em quantidade de NNi). Se não informado
   o tempo mínimo desejado, considera-se o arquivo com menor duração como referência.


**Ao final, os resultados serão salvos no diretório `data/output/`, considerando 2 subdiretórios:**

- `denoised/`: com os NNi completos
- `truncated/`: com os NNi truncados
    
- Além disso, serão salvos relatórios com informações estatísticas básicas sobre os dados
  iniciais e sobre os resultados, considerando o diretório `truncated/`, conforme exemplo abaixo:
    
        ---------- GRUPO: X ----------
        Diretório: ../data/output/truncated/X
        Número de Arquivos: X
        Maior Duração (min): X
        Arquivo com Maior Duração: filename_trunc_X_min.txt
        Menor Duração (min): X
        Arquivo com Menor Duração: filename_trunc_X_min.txt
        Duração Média (min): X
        Limite de Qualidade (%): 90.0
        Qualidade Média (%): X
        Arquivos Abaixo do Limite: 0
        Arquivos Acima do Limite: 167
