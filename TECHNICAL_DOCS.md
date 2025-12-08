# Documentação Técnica: Meta-Estratégia Quantitativa Heston-GARCH

## 1. Visão Geral
Este projeto implementa uma **Meta-Estratégia Quantitativa** projetada para operar no índice S&P 500 (^GSPC). O objetivo central é superar estratégias tradicionais de seguimento de tendência (Trend Following) utilizando **previsão avançada de volatilidade** para filtrar entradas e saídas.

A tese fundamental é que modelos estocásticos (Heston) e condicionais (GARCH) podem antecipar regimes de risco melhor do que métricas passadas (Volatilidade Histórica), protegendo o capital durante turbulências de mercado ("Panic Selling").

## 2. Componentes de Modelagem

### 2.1 Modelo Heston (Estocástico)
O modelo de Heston assume que a volatilidade do ativo não é constante nem determinística, mas segue um processo estocástico (movimento browniano) correlacionado com o preço do ativo.
-   **Arquivo**: `model_heston.py`
-   **Função**: `estimate_heston_vol(returns)`
-   **Mecânica**: Calibra parâmetros (kappa, theta, sigma, rho, v0) e realiza simulações de Monte Carlo para projetar a volatilidade futura esperada.
-   **Vantagem**: Captura a "cauda grossa" e a assimetria (skewness) dos retornos, características comuns em crashes de mercado.

### 2.2 Modelo GARCH (Condicional)
O Generalized Autoregressive Conditional Heteroskedasticity (GARCH) modela a volatilidade atual como uma função dos erros passados (choques de mercado) e da própria volatilidade passada.
-   **Arquivo**: `model_garch.py`
-   **Função**: `estimate_garch_vol(returns, variant="GARCH")`
-   **Mecânica**: Estima a variância condicional dia a dia baseada na série temporal de retornos.
-   **Vantagem**: Excelente para capturar "clusters de volatilidade" (períodos de calmaria seguidos por calmaria, e pânico seguido por pânico).

## 3. Lógica de Sinais (Estratégia Unificada Z-Score)

Todas as estratégias principais (Níveis 2, 3 e 4) utilizam a **mesma lógica de decisão**, variando apenas a fonte de dados de volatilidade. Isso garante uma comparação ceteris paribus da qualidade da informação.

### 3.1 O Indicador Z-Score
O coração da decisão é o **Z-Score de Volatilidade**, que mede quantos desvios padrão a volatilidade atual/projetada está acima ou abaixo de sua média histórica recente.

$$ Z = \frac{X - \mu_{benchmark}}{\sigma_{benchmark}} $$

Onde:
-   **X**: É a volatilidade de input (varia por nível, veja Seção 4).
-   **Benchmark**: Média móvel da volatilidade realizada (default 21 dias).
-   **Interpretação**:
    -   $Z > 2.0$: Volatilidade Extrema (Pânico).
    -   $Z < 0.0$: Volatilidade Baixa (Calmaria).

### 3.2 Regras de Negociação (`signals.py`)

#### Long (Compra)
O sistema entra comprado quando o mercado está em tendência de alta E o regime de volatilidade é favorável.
1.  **Tendência**: EMA Rápida (7) > EMA Lenta (21).
2.  **Filtro de Volatilidade**:
    -   Se o mercado está "calmo" (Vol Input <= Benchmark + Buffer), Compra permitida.
    -   Se o mercado está "nervoso" (Vol Input > Benchmark), Compra bloqueada.

#### Short (Venda) - "Short Sniper"
O sistema entra vendido apenas em condições específicas de deterioração rápida.
1.  **Gatilho de Volatilidade**: Vol Input > Benchmark + Buffer (Regime de Pânico).
2.  **Confirmação de Preço**: Inclinação da EMA Rápida deve ser negativa (Preço caindo).
3.  **Filtro de Tendência**: EMA Lenta não pode estar subindo fortemente.

#### Saída (Exit)
-   **Stop Loss / Take Profit**: Baseado na reversão das EMAs (cruzamento oposto).
-   **Saída de Pânico**: Se o Z-Score explodir (muito alto), a posição pode ser fechada ou invertida.

## 4. Análise Evolutiva (Os 4 Níveis)

O script `evolution_analysis.py` executa 4 backtests paralelos para provar a tese de valor da previsão.

### Nível 1: EMA Only (Baseline)
-   **Lógica**: Trend Following clássico (Cruzamento de Médias).
-   **Volatilidade**: Ignorada.
-   **Papel**: Serve como linha de base. Qualquer estratégia complexa deve bater isso.

### Nível 2: Z-Score Volatilidade Passada (Retrovisor)
-   **Input (X)**: Volatilidade Realizada dos últimos 5 dias (Shift +1).
-   **Lógica**: "O risco de amanhã será igual à média da semana passada."
-   **Limitação**: É um indicador atrasado (Lagging). Em choques súbitos, ele demora a reagir.

### Nível 3: Z-Score Meta-Estratégia (Nosso Modelo)
-   **Input (X)**: Combinação das previsões dos modelos **Heston** e **GARCH**.
-   **Lógica**: "O risco de amanhã é o que a dinâmica estocástica do mercado projeta."
-   **Vantagem**: Tenta antecipar a mudança de regime antes que ela se concretize nos preços passados.

### Nível 4: Z-Score Oráculo (Volatilidade Futura)
-   **Input (X)**: Volatilidade Realizada dos PRÓXIMOS 5 dias (Shift -5).
-   **Lógica**: "Eu tenho uma bola de cristal e sei exatamente o desvio padrão da próxima semana."
-   **Papel**: Limite teórico máximo (Ceiling). Nenhuma estratégia baseada em volatilidade pode performar melhor que isso matematicamente.

## 5. Metodologia de Comparação

Ao padronizar a lógica de decisão (Z-Score) para os Níveis 2, 3 e 4, isolamos a variável **Qualidade da Previsão**.

-   Se **Nível 3 > Nível 2**: Provamos que modelagem matemática (Heston/GARCH) adiciona valor real sobre simples análise técnica (Retrovisor).
-   A proximidade do **Nível 3** com o **Nível 4** indica o quão eficiente é a nossa previsão em relação à onisciência teórica.
