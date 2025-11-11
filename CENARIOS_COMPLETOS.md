# Matriz Completa de Cenários - Meta-Estratégia de Trading

## 📊 Visão Geral da Lógica

### Variáveis de Decisão

1. **Z-Score (Sinal de Risco)**
   - `Z < buy_threshold` → **Calmaria** (buy_gate = True)
   - `Z > sell_threshold` → **Risco** (sell_gate = True)
   - Caso contrário → **Neutro** (nenhum gate ativo)

2. **Tendência de Preço (EMAs)**
   - `EMA_7 > EMA_21` → **trend_up** (tendência de alta)
   - `EMA_7 < EMA_21` → **trend_down** (tendência de baixa)

3. **Confluência das EMAs**
   - **Confluente Compra**: `EMA_7 ↑` AND `EMA_21 ↑` (ambas subindo)
   - **Confluente Venda**: `EMA_7 ↓` AND `EMA_21 ↓` (ambas descendo)
   - **Divergente**: Direções opostas (ex: `EMA_7 ↑` AND `EMA_21 ↓`)

4. **Perfis de Risco**
   - **Conservative**: `buy < -2.0`, `sell > +2.0`
   - **Moderate**: `buy < -1.0`, `sell > +1.0`
   - **Aggressive**: `buy < -0.5`, `sell > +0.5`

---

## 🎯 Matriz de Cenários de ENTRADA

### Cenário 1: COMPRA (LONG)

#### Condições Obrigatórias:
1. ✅ `Z < buy_threshold` (Calmaria detectada)
2. ✅ `EMA_7 > EMA_21` (trend_up - OBRIGATÓRIO)
3. ✅ Filtro de EMA conforme perfil

#### Tabela de Decisão por Perfil:

| Perfil           | Z-Score    | trend_up  | EMA Confluência         | buy_ema_filter    | buy_signal         | Resultado |
|---------|--------|------------|---------- |-------------------------|--------------------|-------------------|
| **Conservative** | `Z < -2.0` | ✅ True  | ✅ Confluente (ambas ↑) | ✅ True           | ✅ **ENTRA LONG** | ✅ |
| **Conservative** | `Z < -2.0` | ✅ True  | ❌ Divergente           | ❌ False          | ❌ Não entra      | ❌ |
| **Conservative** | `Z < -2.0` | ❌ False | - | -                    | ❌ **BLOQUEADO**  | ❌ |
| **Moderate**     | `Z < -1.0` | ✅ True  | ✅ Confluente (ambas ↑) | ✅ True           | ✅ **ENTRA LONG** | ✅ |
| **Moderate**     | `Z < -1.0` | ✅ True  | ❌ Divergente           | ❌ False          | ❌ Não entra      | ❌ |
| **Moderate**     | `Z < -1.0` | ❌ False | - | -                    | ❌ **BLOQUEADO**  | ❌ |
| **Aggressive**   | `Z < -0.5` | ✅ True  | ✅ Confluente (ambas ↑) | ✅ True            | ✅ **ENTRA LONG** | ✅ |
| **Aggressive**   | `Z < -0.5` | ✅ True  | ✅ Divergente (longa ↑) | ✅ True            | ✅ **ENTRA LONG** | ✅ |
| **Aggressive**   | `Z < -0.5` | ❌ False | - | -                   | ❌ **BLOQUEADO**    | ❌ |

#### Regras Especiais:
- ❌ **NUNCA entra em compra se `trend_down`** (mesmo com Calmaria)
- ❌ **Conservative/Moderate NUNCA entram em divergência**
- ✅ **Aggressive pode entrar em divergência, MAS apenas se `trend_up`**

---

### Cenário 2: VENDA (SHORT)

#### Condições Obrigatórias:
1. ✅ `Z > sell_threshold` (Risco detectado)
2. ✅ `EMA_7 < EMA_21` (trend_down - OBRIGATÓRIO)
3. ✅ Filtro de EMA conforme perfil

#### Tabela de Decisão por Perfil:

| Perfil | Z-Score | trend_down | EMA Confluência | sell_ema_filter | sell_signal | Resultado |
|--------|---------|------------|-----------------|-----------------|-------------|-----------|
| **Conservative** | `Z > +2.0` | ✅ True | ✅ Confluente (ambas ↓) | ✅ True | ✅ **ENTRA SHORT** | ✅ |
| **Conservative** | `Z > +2.0` | ✅ True | ❌ Divergente | ❌ False | ❌ Não entra | ❌ |
| **Conservative** | `Z > +2.0` | ❌ False | - | - | ❌ **BLOQUEADO** | ❌ |
| **Moderate** | `Z > +1.0` | ✅ True | ✅ Confluente (ambas ↓) | ✅ True | ✅ **ENTRA SHORT** | ✅ |
| **Moderate** | `Z > +1.0` | ✅ True | ❌ Divergente | ❌ False | ❌ Não entra | ❌ |
| **Moderate** | `Z > +1.0` | ❌ False | - | - | ❌ **BLOQUEADO** | ❌ |
| **Aggressive** | `Z > +0.5` | ✅ True | ✅ Confluente (ambas ↓) | ✅ True | ✅ **ENTRA SHORT** | ✅ |
| **Aggressive** | `Z > +0.5` | ✅ True | ✅ Divergente (mas trend_down) | ✅ True | ✅ **ENTRA SHORT** | ✅ |
| **Aggressive** | `Z > +0.5` | ❌ False | - | - | ❌ **BLOQUEADO** | ❌ |

#### Regras Especiais:
- ❌ **NUNCA entra em venda se `trend_up`** (mesmo com Risco)
- ❌ **Conservative/Moderate NUNCA entram em divergência**
- ✅ **Aggressive pode entrar em divergência, MAS apenas se `trend_down`**

---

### Cenário 3: NEUTRO (Sem Entrada)

#### Situações que resultam em NEUTRO:

| Z-Score | trend_up | trend_down | buy_gate | sell_gate | buy_signal | sell_signal | Resultado |
|---------|----------|------------|----------|-----------|------------|-------------|-----------|
| `-2.0 ≤ Z ≤ +2.0` (Conservative) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | **NEUTRO** |
| `-1.0 ≤ Z ≤ +1.0` (Moderate) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | **NEUTRO** |
| `-0.5 ≤ Z ≤ +0.5` (Aggressive) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | **NEUTRO** |
| `Z < threshold` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | **NEUTRO** (Calmaria mas trend_down) |
| `Z > threshold` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | **NEUTRO** (Risco mas trend_up) |
| `Z < threshold` | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | **NEUTRO** (Calmaria mas sem confluência) |
| `Z > threshold` | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | **NEUTRO** (Risco mas sem confluência) |

---

## 🔄 Matriz de Cenários de SAÍDA

### Estado Atual: LONG (position = +1)

| Dia Atual | trend_up | trend_down | Decisão | Nova Position | Motivo |
|-----------|----------|------------|---------|---------------|--------|
| ✅ | ✅ True | ❌ False | **MANTER** | `+1` (LONG) | Tendência ainda favorável |
| ✅ | ❌ False | ✅ True | **SAIR** | `0` (NEUTRO) | Tendência inverteu (cruzamento EMAs) |
| ✅ | ❌ False | ❌ False | **SAIR** | `0` (NEUTRO) | EMAs iguais (raro, mas fecha posição) |

**Regra**: Mantém LONG enquanto `trend_up = True`, sai quando `trend_up = False`

---

### Estado Atual: SHORT (position = -1)

| Dia Atual | trend_up | trend_down | Decisão | Nova Position | Motivo |
|-----------|----------|------------|---------|---------------|--------|
| ✅ | ❌ False | ✅ True | **MANTER** | `-1` (SHORT) | Tendência ainda favorável |
| ✅ | ✅ True | ❌ False | **SAIR** | `0` (NEUTRO) | Tendência inverteu (cruzamento EMAs) |
| ✅ | ❌ False | ❌ False | **SAIR** | `0` (NEUTRO) | EMAs iguais (raro, mas fecha posição) |

**Regra**: Mantém SHORT enquanto `trend_down = True`, sai quando `trend_down = False`

---

### Estado Atual: NEUTRO (position = 0)

| Dia Atual | buy_signal | sell_signal | Decisão | Nova Position | Motivo |
|-----------|------------|-------------|---------|---------------|--------|
| ✅ | ✅ True | ❌ False | **ENTRAR LONG** | `+1` (LONG) | Sinal de compra ativo |
| ✅ | ❌ False | ✅ True | **ENTRAR SHORT** | `-1` (SHORT) | Sinal de venda ativo |
| ✅ | ❌ False | ❌ False | **MANTER NEUTRO** | `0` (NEUTRO) | Nenhum sinal ativo |

---

## 📋 Matriz Completa de Transições de Estado

### Estados Possíveis:
- **LONG** (`+1`): Posição comprada
- **SHORT** (`-1`): Posição vendida
- **NEUTRO** (`0`): Sem posição

### Tabela de Transições:

| Estado Anterior | Condições Atuais | Ação | Estado Novo |
|-----------------|------------------|------|-------------|
| **NEUTRO (0)** | `buy_signal = True` | Entrar LONG | **LONG (+1)** |
| **NEUTRO (0)** | `sell_signal = True` | Entrar SHORT | **SHORT (-1)** |
| **NEUTRO (0)** | `buy_signal = False` AND `sell_signal = False` | Manter | **NEUTRO (0)** |
| **LONG (+1)** | `trend_up = True` | Manter LONG | **LONG (+1)** |
| **LONG (+1)** | `trend_up = False` | Sair (fechar posição) | **NEUTRO (0)** |
| **SHORT (-1)** | `trend_down = True` | Manter SHORT | **SHORT (-1)** |
| **SHORT (-1)** | `trend_down = False` | Sair (fechar posição) | **NEUTRO (0)** |

---

## 🎲 Exemplos Práticos de Cenários

### Exemplo 1: Entrada em LONG (Conservative)
```
Z-Score = -2.5  → buy_gate = True (Calmaria)
EMA_7 = 4500, EMA_21 = 4400  → trend_up = True
EMA_7_diff > 0, EMA_21_diff > 0  → ema_confluent_buy = True
buy_ema_filter = True (Conservative só aceita confluência)
buy_signal = True & True & True = True
→ ENTRADA EM LONG (+1)
```

### Exemplo 2: Bloqueio de Entrada (Calmaria mas trend_down)
```
Z-Score = -2.5  → buy_gate = True (Calmaria)
EMA_7 = 4400, EMA_21 = 4500  → trend_up = False (trend_down = True)
→ buy_signal = True & False & ... = False
→ NÃO ENTRA (bloqueado por ir contra a tendência)
```

### Exemplo 3: Entrada em SHORT (Aggressive com Divergência)
```
Z-Score = +0.6  → sell_gate = True (Risco)
EMA_7 = 4400, EMA_21 = 4500  → trend_down = True
EMA_7_diff < 0, EMA_21_diff > 0  → ema_divergent = True
sell_ema_filter = True (Aggressive aceita divergência se trend_down)
sell_signal = True & True & True = True
→ ENTRADA EM SHORT (-1)
```

### Exemplo 4: Saída de LONG (Cruzamento de EMAs)
```
Estado anterior: LONG (+1)
EMA_7 = 4400, EMA_21 = 4500  → trend_up = False (trend_down = True)
→ position = 0 (fecha posição)
```

### Exemplo 5: Manutenção de SHORT
```
Estado anterior: SHORT (-1)
EMA_7 = 4400, EMA_21 = 4500  → trend_down = True
→ position = -1 (mantém posição)
```

---

## 🔍 Cenários Especiais e Edge Cases

### Edge Case 1: Z-Score no Limite do Threshold
- **Conservative**: `Z = -2.0` → `buy_gate = False` (precisa ser `< -2.0`)
- **Moderate**: `Z = -1.0` → `buy_gate = False` (precisa ser `< -1.0`)
- **Aggressive**: `Z = -0.5` → `buy_gate = False` (precisa ser `< -0.5`)

### Edge Case 2: EMAs Iguais
- `EMA_7 == EMA_21` → `trend_up = False` AND `trend_down = False`
- Resultado: Fecha qualquer posição aberta (retorna para NEUTRO)

### Edge Case 3: Calmaria e Risco Simultâneos (Impossível)
- `Z < buy_threshold` AND `Z > sell_threshold` → **Impossível matematicamente**
- Exemplo: `Z < -2.0` AND `Z > +2.0` → Não pode ocorrer

### Edge Case 4: Zona Neutra (Entre Thresholds)
- **Conservative**: `-2.0 < Z < +2.0` → Nenhum gate ativo
- **Moderate**: `-1.0 < Z < +1.0` → Nenhum gate ativo
- **Aggressive**: `-0.5 < Z < +0.5` → Nenhum gate ativo
- Resultado: **NEUTRO** (não entra em nenhuma posição)

### Edge Case 5: Divergência de EMAs (Apenas Aggressive)
- **Conservative/Moderate**: Divergência → `buy_ema_filter = False` → Não entra
- **Aggressive**: Divergência + `trend_up` → `buy_ema_filter = True` → Pode entrar

---

## 📊 Resumo das Regras Críticas

### ✅ REGRAS QUE SEMPRE SE APLICAM:

1. **NUNCA operar contra a tendência:**
   - Calmaria (`Z < threshold`) + `trend_down` → ❌ NÃO ENTRA
   - Risco (`Z > threshold`) + `trend_up` → ❌ NÃO ENTRA

2. **Conservative e Moderate:**
   - ❌ NUNCA entram em divergência de EMAs
   - ✅ Apenas entram em confluência (ambas EMAs na mesma direção)

3. **Aggressive:**
   - ✅ Pode entrar em divergência, MAS apenas se:
     - Para COMPRA: `trend_up = True` (mesmo com divergência)
     - Para VENDA: `trend_down = True` (mesmo com divergência)

4. **Saída de Posições:**
   - LONG: Sai quando `trend_up = False` (cruzamento de EMAs)
   - SHORT: Sai quando `trend_down = False` (cruzamento de EMAs)
   - **Decoupled**: Lógica de saída é independente da lógica de entrada

---

## 🎯 Thresholds por Perfil

| Perfil | Buy Threshold | Sell Threshold | Zona Neutra |
|--------|---------------|----------------|-------------|
| **Conservative** | `Z < -2.0` | `Z > +2.0` | `-2.0 ≤ Z ≤ +2.0` |
| **Moderate** | `Z < -1.0` | `Z > +1.0` | `-1.0 ≤ Z ≤ +1.0` |
| **Aggressive** | `Z < -0.5` | `Z > +0.5` | `-0.5 ≤ Z ≤ +0.5` |

---

## 📝 Notas Finais

- **Swing Trade**: A estratégia mantém posições por vários dias até o cruzamento das EMAs
- **Decoupled Exit**: A saída não depende dos sinais de entrada (Z-Score), apenas das EMAs
- **Profile Discrimination**: Cada perfil tem regras diferentes para entrada, mas a saída é igual para todos
- **Trend Filter**: O filtro de tendência (EMAs) é OBRIGATÓRIO e não pode ser contornado

---

**Última atualização**: Baseado na lógica atual do código (sem filtro de inclinação)

