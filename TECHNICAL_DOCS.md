# Documentação Técnica: Meta-Estratégia Quantitativa Heston-GARCH

## 1. Visão Geral
Este sistema implementa uma **Meta-Estratégia Quantitativa** que opera no índice S&P 500 (^GSPC). Diferente de setups tradicionais que olham apenas para o preço, esta estratégia utiliza **previsão estocástica de volatilidade** para filtrar entradas e saídas.

A premissa central é a **Assimetria de Volatilidade**: O mercado quase sempre "avisa" antes de cair forte (a volatilidade explode). Se conseguirmos prever essa explosão matematicamente, podemos sair do mercado antes do crash ("Panic Selling").

## 2. Arquitetura do Sistema

O projeto está centralizado em `main.py`, que orquestra todo o pipeline:

1.  **Ingestão de Dados**: Baixa dados históricos via `yfinance`.
2.  **Modelagem Matemática**: Estima parâmetros Heston e GARCH.
3.  **Geração de Sinais**: Calcula Z-Scores e define posições (Long/Short).
4.  **Backtesting**: Simula a execução das estratégias dia-a-dia.
5.  **Visualização**: Gera gráficos interativos (Zoom/Pan) e relatórios de métricas.

## 3. Os Modelos Matemáticos

### 3.1 Modelo Heston (Estocástico)
*Arquivo: `model_heston.py`*

O modelo de Heston trata a volatilidade não como um número fixo, mas como um processo que varia no tempo seguindo uma Equação Diferencial Estocástica (SDE):
$$ dV_t = \kappa(\theta - V_t)dt + \sigma \sqrt{V_t} dW_2 $$

*   **O que ele faz:** Simula milhares de cenários futuros (Monte Carlo) para encontrar a distribuição de probabilidade da volatildade.
*   **Por que usamos:** Ele captura a "cauda grossa" (risco de eventos extremos) melhor que qualquer média móvel.

### 3.2 Modelo GARCH(1,1) (Condicional)
*Arquivo: `model_garch.py`*

O GARCH assume que a volatilidade de hoje depende dos choques de ontem.
*   **O que ele faz:** Detecta "clusters" de volatilidade (se ontem foi agitado, hoje provavelmente será também).
*   **Por que usamos:** É excelente para reagir rapidamente a mudanças de regime de curto prazo.

## 4. O Coração da Estratégia: Z-Score Unificado

Para comparar maçãs com maçãs, todas as estratégias (Níveis 2, 3 e 4) usam exatamente a mesma regra de decisão, variando apenas o **INPUT** de volatilidade.

$$ Z = \frac{VolInput - MediaHistorica(21d)}{DesvioPadrao(21d)} $$

### Regras de Trading (`signals.py`)

1.  **Compra (Long)**:
    *   Tendência de Alta (EMA 7 > EMA 21).
    *   Volatilidade Controlada ($Z < Threshold$).
2.  **Venda (Short)**:
    *   **Gatilho de Pânico Padrão**: $Z > Threshold$ E Tendência de Baixa (Slope < 0).
    *   **Panic Override (Prioridade Máxima)**: Se $Z > (Threshold \times PanicFactor)$, VENDE IMEDIATAMENTE ignorando a tendência. Isso serve para escapar de "Cisnes Negros" instantâneos.

## 5. Os 4 Níveis de Evolução

O sistema executa 4 backtests simultâneos para provar a tese:

### Nível 1: EMA Only (O "Ingênuo")
*   **Lógica**: Cruza médias móveis. Ignora volatilidade.
*   **Resultado**: Sofre em mercados laterais e crashes rápidos.

### Nível 2: Retrovisor (Volatilidade Passada)
*   **Input**: Volatilidade realizada de ontem.
*   **Defeito**: É atrasado (Lagging). Quando ele percebe que o risco subiu, o preço já caiu 10%.

### Nível 3: Meta-Estratégia (Heston + GARCH)
*   **Input**: Média das previsões dos modelos Heston e GARCH.
*   **Configuração Otimizada (Agressiva)**:
    *   *Threshold Venda*: 0.5 (Reage a qualquer sinal de fumaça).
    *   *Panic Factor*: 2.5 (Entra em modo pânico mais cedo).
*   **Vantagem**: Tenta prever o risco *antes* dele se materializar no preço.

### Nível 4: O Oráculo (Volatilidade Futura)
*   **Input**: A volatilidade real dos PRÓXIMOS 5 dias (Shift -5).
*   **Configuração**: Standard (Sem Panic Override).
*   **Lógica**: Ao contrário do Nível 3, o Oráculo *não* força a venda. Ele apenas "enxerga" o risco futuro e proíbe compras em momentos de alta vol. A venda só ocorre se a tendência de preço também reverter.
*   **Objetivo**: Mensurar o ganho puro de *"Saber o Futuro"* sem usar gatilhos de pânico artificiais.

## 6. Resultados e Conclusão

A análise comparativa (Gráfico de Equity) demonstra que:
1.  O **Nível 3 (Modelos)** supera consistentemente o Nível 2, provando que matemática avançada agrega alfa.
2.  A inclusão do **Panic Override** foi crucial para que os modelos pudessem proteger a carteira durante o COVID-19 (Março/2020), aproximando a curva do Nível 3 à do Oráculo.

---
*Gerado automaticamente pelo Sistema Antigravity v1.0*
