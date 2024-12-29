def evaluate_crypto(symbol, additional_data):
    """
    Evalúa una criptomoneda y le asigna un puntaje basado en todos los indicadores disponibles.
    Filtra criptos con datos insuficientes o indicadores clave faltantes.
    Args:
        symbol (str): El símbolo de la cripto (e.g., BTC/USDT).
        additional_data (dict): Datos adicionales con indicadores clave.
    Returns:
        tuple: (puntaje_total, es_valida) donde es_valida indica si cumple criterios mínimos.
    """
    try:
        # Pesos para cada indicador
        weights = {
            "rsi": 0.1,                # 10% para RSI
            "adx": 0.1,                # 10% para ADX
            "relative_volume": 0.15,   # 15% para volumen relativo
            "market_sentiment": 0.15,  # 15% para sentimiento del mercado
            "support_resistance": 0.2, # 20% para soporte/resistencia
            "momentum_divergences": 0.1,  # 10% para divergencias de momentum
            "fear_greed": 0.1,         # 10% para miedo/codicia
            "candlestick_pattern": 0.1 # 10% para patrones de velas
        }

        # Inicializar puntajes
        scores = {}

        # Indicadores clave
        scores["rsi"] = score_rsi(additional_data.get("rsi"))
        scores["adx"] = score_adx(additional_data.get("adx"))
        scores["relative_volume"] = score_volume(additional_data.get("relative_volume"))
        scores["market_sentiment"] = score_sentiment(additional_data.get("market_sentiment"))
        scores["support_resistance"] = score_support_resistance(
            additional_data.get("current_price"),
            additional_data.get("support"),
            additional_data.get("resistance")
        )
        scores["momentum_divergences"] = score_momentum_divergences(
            additional_data.get("momentum_divergences")
        )
        scores["fear_greed"] = score_fear_greed(additional_data.get("fear_greed"))
        scores["candlestick_pattern"] = score_candlestick_pattern(
            additional_data.get("candlestick_pattern")
        )

        # Validar datos mínimos necesarios
        is_valid = validate_crypto_data(additional_data)
        if not is_valid:
            return 0, False  # Puntaje 0 si faltan indicadores críticos

        # Calcular puntaje total ponderado
        total_score = sum(weights[indicator] * scores[indicator] for indicator in scores)

        print(f"🔍 Puntaje para {symbol}: {total_score:.2f}")
        return total_score, True

    except Exception as e:
        print(f"❌ Error al evaluar {symbol}: {e}")
        return 0, False

# Funciones de puntaje específicas para cada indicador
def score_rsi(rsi):
    if rsi is None or isinstance(rsi, str):
        return 0
    if rsi < 30:
        return 1
    elif 30 <= rsi <= 70:
        return 0.5
    return 0

def score_adx(adx):
    if adx is None:
        return 0
    if adx > 25:
        return 1
    elif 20 <= adx <= 25:
        return 0.5
    return 0

def score_volume(relative_volume):
    if relative_volume is None:
        return 0
    if relative_volume > 1.5:
        return 1
    elif 1.0 <= relative_volume <= 1.5:
        return 0.5
    return 0

def score_sentiment(sentiment):
    if isinstance(sentiment, dict):
        sentiment = sentiment.get('overall_sentiment', 0)
    if sentiment is None:
        return 0
    if sentiment > 60:
        return 1
    elif 40 <= sentiment <= 60:
        return 0.5
    return 0

def score_support_resistance(current_price, support, resistance):
    if current_price is None or support is None or resistance is None:
        return 0
    if current_price <= support * 1.05:
        return 1
    elif support * 1.05 < current_price < resistance * 0.95:
        return 0.5
    return 0

def score_momentum_divergences(divergences):
    if not divergences:
        return 0
    bullish_count = sum(1 for div in divergences if div[0] == "bullish")
    bearish_count = sum(1 for div in divergences if div[0] == "bearish")
    if bullish_count > bearish_count:
        return 1
    elif bullish_count == bearish_count:
        return 0.5
    return 0

def score_fear_greed(fear_greed):
    if fear_greed is None:
        return 0
    # Convert fear_greed to a float if it's a string
    if isinstance(fear_greed, str):
        fear_greed = float(fear_greed)
    if fear_greed > 50:
        return 1
    elif 40 <= fear_greed <= 50:
        return 0.5
    return 0

def score_candlestick_pattern(patterns):
    if not patterns:
        return 0
    bullish_count = sum(1 for p in patterns if p[2] == "bullish")
    bearish_count = sum(1 for p in patterns if p[2] == "bearish")
    if bullish_count > bearish_count:
        return 1
    elif bullish_count == bearish_count:
        return 0.5
    return 0

def validate_crypto_data(additional_data):
    # Aquí puedes definir las condiciones que consideras necesarias para validar los datos
    required_keys = ["rsi", "adx", "relative_volume", "market_sentiment", "support", "resistance", "fear_greed", "candlestick_pattern"]
    
    # Verificar que todos los indicadores clave estén presentes y no sean None
    for key in required_keys:
        if key not in additional_data or additional_data[key] is None:
            return False
    
    # Si todos los indicadores están presentes y son válidos, retorna True
    return True