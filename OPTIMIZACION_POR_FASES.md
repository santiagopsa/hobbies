# Guía de Optimización por Fases

## ¿Por qué optimizar por fases?

Optimizar todos los parámetros al mismo tiempo puede generar mucho "ruido" - es difícil saber qué parámetros realmente están ayudando. Al dividir la optimización en dos fases:

1. **Fase 1 - Entrada (Entry)**: Encuentra los mejores parámetros para **CUANDO COMPRAR**
2. **Fase 2 - Salida (Exit)**: Encuentra los mejores parámetros para **CUANDO VENDER** y el trailing stop

Esto te permite:
- Identificar claramente qué parámetros mejoran las compras
- Identificar claramente qué parámetros mejoran las ventas
- Reducir el ruido en los resultados
- Entender mejor tu estrategia

## Cómo usar

### Opción 1: Optimizar ambas fases automáticamente (recomendado)

```bash
python optimize_backtest.py --phase both --method differential_evolution --iterations 50
```

Esto:
1. Primero optimiza los parámetros de entrada
2. Luego optimiza los parámetros de salida usando los mejores parámetros de entrada
3. Guarda resultados separados: `optimization_results_entry.json` y `optimization_results_exit.json`
4. Guarda resultados combinados: `optimization_results.json`

### Opción 2: Optimizar solo entrada primero

```bash
python optimize_backtest.py --phase entry --method differential_evolution --iterations 50
```

Esto optimiza solo los parámetros de compra:
- RISK_FRACTION
- ADX_MIN
- RSI_MIN
- RSI_MAX
- RVOL_BASE
- SCORE_GATE

Resultado guardado en: `optimization_results_entry.json`

### Opción 3: Optimizar solo salida (usando entrada ya optimizada)

Primero necesitas haber optimizado la entrada. Luego:

```bash
python optimize_backtest.py --phase exit --method differential_evolution --iterations 50 --entry-params-file optimization_results_entry.json
```

Esto optimiza solo los parámetros de venta/trailing:
- CHAN_K_MEDIUM
- INIT_STOP_ATR_MULT
- ARM_TRAIL_PCT
- BE_R_MULT

Usa los parámetros de entrada del archivo especificado.

### Opción 4: Optimizar todo junto (como antes)

```bash
python optimize_backtest.py --phase full --method differential_evolution --iterations 50
```

## Parámetros por fase

### Fase ENTRY (Compra)
- `RISK_FRACTION`: Cuánto dinero arriesgar por operación (10-30%)
- `ADX_MIN`: Fuerza mínima de tendencia para entrar (18-28)
- `RSI_MIN`: RSI mínimo para entrar (45-55)
- `RSI_MAX`: RSI máximo para entrar (65-75)
- `RVOL_BASE`: Umbral de volumen relativo (1.2-2.0)
- `SCORE_GATE`: Puntuación mínima para entrar (4.5-6.0)

### Fase EXIT (Venta/Trailing)
- `CHAN_K_MEDIUM`: Factor K del Chandelier Stop para volatilidad media (2.0-3.5)
- `INIT_STOP_ATR_MULT`: Multiplicador ATR para stop inicial (1.0-2.0)
- `ARM_TRAIL_PCT`: Activar trailing después de % ganancia (0.5-1.5%)
- `BE_R_MULT`: Break-even después de X veces el riesgo (1.0-2.0)

## Ejemplo de flujo completo

```bash
# Paso 1: Optimizar parámetros de entrada
python optimize_backtest.py --phase entry --method differential_evolution --iterations 50 --output entry_params.json

# Paso 2: Optimizar parámetros de salida usando los mejores de entrada
python optimize_backtest.py --phase exit --method differential_evolution --iterations 50 --entry-params-file entry_params.json --output exit_params.json
```

## Ventajas de este enfoque

1. **Menos ruido**: Cada fase se enfoca en un aspecto específico
2. **Mejor comprensión**: Sabes qué parámetros afectan las compras vs las ventas
3. **Más control**: Puedes ajustar manualmente una fase sin afectar la otra
4. **Resultados más claros**: Es más fácil identificar qué funciona y qué no

## Notas importantes

- Cuando optimizas la fase EXIT, los parámetros de ENTRY se mantienen fijos (usando los valores por defecto o los del archivo especificado)
- Cuando optimizas la fase ENTRY, los parámetros de EXIT se mantienen fijos (usando valores por defecto)
- Los resultados de cada fase se guardan en archivos separados para referencia
- Puedes usar cualquier método: `grid`, `random`, o `differential_evolution`



