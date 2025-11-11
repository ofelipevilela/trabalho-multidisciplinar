# Meta-Estratégia de Trading Quantitativo

Sistema de trading quantitativo que utiliza modelos Heston e GARCH para previsão de volatilidade, combinando sinais de risco com filtros de tendência para gerar operações de compra e venda.

## 📋 Requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

## 🚀 Configuração Inicial

### 1. Ativar o Ambiente Virtual

Se você já criou um ambiente virtual anteriormente:

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt (cmd):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

Quando ativado, você verá `(venv)` no início do prompt.

### 2. Criar Ambiente Virtual (se ainda não criou)

Se você ainda não tem um ambiente virtual:

```bash
python -m venv venv
```

Depois, ative conforme as instruções acima.

### 3. Instalar Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

**Opção 1: Usando requirements.txt (recomendado)**
```bash
pip install -r requirements.txt
```

**Opção 2: Instalação manual**
```bash
pip install pandas numpy matplotlib yfinance arch
```

## ⚙️ Configuração do Projeto

### Arquivos Principais

- `main.py` - Script principal de execução
- `model_heston.py` - Modelo Heston (Monte Carlo)
- `model_garch.py` - Modelo GARCH
- `signals.py` - Lógica de geração de sinais de trading

### Configurações (em `main.py`)

Edite as variáveis no início do arquivo `main.py`:

```python
TICKER = "^GSPC"         # Ativo a analisar (ex: "^GSPC", "NVDA", "AAPL")
START  = "2018-01-01"    # Data inicial dos dados históricos
PROFILE = "aggressive"   # Perfil de risco: "conservative" | "moderate" | "aggressive"
```

## ▶️ Como Executar

### Opção 1: Terminal Integrado do Cursor/VS Code

1. Abra o terminal integrado: `Ctrl + `` (backtick) ou `Terminal > New Terminal`
2. Certifique-se de que o ambiente virtual está ativado (deve aparecer `(venv)` no prompt)
3. Execute:

```bash
python main.py
```

### Opção 2: Terminal do Sistema

1. Abra PowerShell ou Command Prompt
2. Navegue até a pasta do projeto:

```powershell
cd "D:\CODES\TRAB MULTI\trabalho-multidisciplinar"
```

3. Ative o ambiente virtual:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Execute:

```bash
python main.py
```

### Opção 3: Botão Run no Cursor

1. Abra o arquivo `main.py`
2. Clique no botão ▶️ (Run) no canto superior direito
3. Ou pressione `F5` (pode precisar configurar o launch.json)

## 📊 O que o Código Faz

### Fluxo de Execução

1. **Carrega Dados**: Baixa preços históricos do Yahoo Finance
2. **Calcula Retornos**: Calcula retornos diários do ativo
3. **Estima Volatilidade**:
   - **Heston**: Simulação de Monte Carlo (500 caminhos, 30 dias à frente)
   - **GARCH**: Modelo GARCH(1,1) com janela móvel
4. **Calcula Consenso**: Média das previsões Heston e GARCH
5. **Calcula Benchmark**: Média móvel de 7 dias da volatilidade realizada de 21 dias
6. **Calcula Z-Score**: `(Consenso - Benchmark) / StdDev(Benchmark)`
7. **Gera Sinais**:
   - **COMPRA**: Z-Score < threshold (Calmaria) + EMAs para cima
   - **VENDA**: Z-Score > threshold (Risco) + EMAs para baixo
8. **Aplica Filtros**: Confluência/divergência das EMAs por perfil
9. **Gerencia Saída**: Mantém posição até inversão de tendência (cruzamento das EMAs)
10. **Salva Resultados**: CSV em `outputs/signals.csv`
11. **Visualiza**: Gráficos interativos e métricas de performance

## 📁 Estrutura de Saída

```
trabalho-multidisciplinar/
├── main.py
├── model_heston.py
├── model_garch.py
├── signals.py
├── outputs/
│   └── signals.csv          # Resultados salvos
├── venv/                     # Ambiente virtual (não versionar)
└── README.md
```

## 🎯 Perfis de Risco

| Perfil | Buy Threshold | Sell Threshold | Descrição |
|--------|---------------|----------------|-----------|
| **Conservative** | -2.0 | +2.0 | Só opera em sinais extremos |
| **Moderate** | -1.0 | +1.0 | Opera em sinais relevantes |
| **Aggressive** | -0.5 | +0.5 | Opera em qualquer sinal direcional |

## 📈 Interpretação dos Resultados

### Arquivo `outputs/signals.csv`

Contém todas as colunas calculadas:
- `price`: Preço do ativo
- `returns`: Retornos diários
- `ema7`, `ema21`: Médias móveis exponenciais
- `garch_vol`, `heston_vol`: Previsões de volatilidade
- `vol_pred_cons`: Consenso (média das previsões)
- `vol_hist_benchmark`: Benchmark de volatilidade
- `zscore`: Z-Score do consenso vs benchmark
- `risk_state`: Calmaria / Risco / Neutral
- `buy_gate`, `sell_gate`: Gates de entrada
- `buy_signal`, `sell_signal`: Sinais de entrada
- `position`: +1 (LONG), -1 (SHORT), 0 (NEUTRO)

### Gráficos Gerados

1. **Preço + EMAs + Posições**: Mostra preços, EMAs, períodos em posição (fundo colorido) e marcadores de entrada/saída
2. **Volatilidades**: Compara previsões vs benchmark
3. **Z-Score**: Mostra Z-Score e thresholds por perfil
4. **Equity & Drawdown**: Curva de patrimônio e drawdown
5. **Histograma**: Distribuição dos retornos da estratégia

## 🔧 Solução de Problemas

### Erro: "yfinance is required"
```bash
pip install yfinance
```

### Erro: "arch package is required"
```bash
pip install arch
```

### Erro: "No module named 'pandas'"
```bash
pip install pandas numpy matplotlib
```

### Ambiente virtual não ativa (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Interpretador Python não encontrado no Cursor
1. Pressione `Ctrl + Shift + P`
2. Digite: `Python: Select Interpreter`
3. Selecione: `.\venv\Scripts\python.exe`

## 📝 Notas Importantes

- **Primeira execução**: Pode demorar alguns minutos devido à simulação de Monte Carlo do Heston
- **Dados históricos**: O código baixa dados do Yahoo Finance automaticamente
- **Período mínimo**: Recomenda-se pelo menos 1 ano de dados para cálculos confiáveis
- **Performance**: O modelo Heston usa 500 simulações por padrão (pode ser ajustado em `model_heston.py`)

## 🔄 Próximos Passos

Após validar que tudo funciona:
1. Calibrar parâmetros do modelo Heston
2. Ajustar thresholds por perfil
3. Testar diferentes períodos e ativos
4. Otimizar janelas de cálculo (EMA, volatilidade, etc.)

## 📧 Suporte

Em caso de problemas:
1. Verifique se o ambiente virtual está ativado
2. Confirme que todas as dependências estão instaladas
3. Verifique se há conexão com internet (para baixar dados)
4. Revise os logs de erro no terminal

---

**Última atualização**: Sistema com suporte a compra/venda, filtros de confluência por perfil, e visualização completa de entradas/saídas.

