# SVM linear para análise da potabilidade da água
Projeto acadêmico que investiga a potabilidade da água usando SVM Linear com validação cruzada estratificada e métricas robustas de avaliação. Foco em equilíbrio de classes e generalização do modelo.

## Autores e afiliação
-Marcos Figueiredo: Universidade Estadual da Bahia

-Anthony Kauan: Universidade Estadual da Bahia

-Felipe Dantas: Universidade Estadual da Bahia

-Gabriel Boaventura: Universidade Estadual da Bahia

-Caique Silva: Universidade Estadual da Bahia

## Visão geral
O projeto prevê se uma amostra de água é potável com base em indicadores físico-químicos. Explora o “espaço de hipóteses” com SVM Linear, enfatizando generalização, validação cruzada e métricas que capturam desempenho em cenários com classes desbalanceadas.

-*Pergunta central:* É possível prever potabilidade com dados corretos?

-*Ferramenta:* Máquina de Vetores de Suporte linear para separação clara entre classes.

-*Justiça na avaliação:* Validação cruzada estratificada para evitar sobreajuste.

-*Critérios de desempenho*: Acurácia, ROC AUC, PR AUC e matriz de confusão.

## Dados e features
Fonte: Conjunto de indicadores de qualidade da água (ph, chloramines, sulfate, conductivity, trihalomethanes, turbidity, potability).

### Seleção de variáveis:

Mantidas: ph, Chloramines, Sulfate, Conductivity, Trihalomethanes, Turbidity.

Removidas: Solids.

### Pré-processamento:

Padronização: StandardScaler aplicado às features.

Divisão: StratifiedKFold em 100 partes para treino e teste.

## Modelo, perda e hiperparâmetros
Modelo: LinearSVC (SVM Linear).

Função de perda: hinge para penalizar classificações no lado errado da margem.

Regularização: Parâmetro C controla o equilíbrio entre ajuste aos dados e generalização.

Hiperparâmetros críticos:

-C: 1

-Loss: hinge

-Max iter: 500000

-Class weight: balanced (atenção igual às classes, mesmo com desbalanceamento)

> A regularização evita que o modelo “decore” casos específicos. Um C alto foca em acertos no treino; um C baixo favorece generalização.

## Protocolo experimental
Validação: StratifiedKFold(n_splits=100, shuffle=True, random_state=42).

Pipeline: Padronização + LinearSVC.

Métricas calculadas:
'''
Acurácia média por fold.

ROC AUC via cross_val_score.

PR AUC via cross_val_score.

Matriz de confusão com Yellowbrick.
'''
## Resultados
Acurácia: 83%

ROC AUC: 72.4%

PR AUC: 66.7%

Matriz de confusão: desempenho mediano, com distribuição de erros que sugere risco para uso operacional.

> Interpretação: Para o objetivo de análise de potabilidade, metas próximas de 90% seriam preferíveis; este SVM Linear apresenta desempenho apenas mediano e pode requerer ajustes ou modelos alternativos.

### Reprodutibilidade e execução
### Requisitos
Python: 3.9+

Bibliotecas:

Essenciais: numpy, pandas, scikit-learn

Visualização: plotly, seaborn, matplotlib, yellowbrick

Utilitários: statistics
