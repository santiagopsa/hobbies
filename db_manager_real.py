import sqlite3

DB_NAME = "trading_real.db"

def initialize_db():
    """Crea la base de datos y la tabla transactions_new desde cero."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Eliminar la tabla si existe para empezar desde cero
    cursor.execute("DROP TABLE IF EXISTS transactions_new")
    
    # Crear la nueva tabla con todas las columnas necesarias
    cursor.execute('''
        CREATE TABLE transactions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL,
            trade_id TEXT,
            rsi REAL,
            adx REAL,
            atr REAL,
            relative_volume REAL,
            divergence TEXT,
            bb_position TEXT,
            confidence INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Base de datos {DB_NAME} inicializada con la tabla transactions_new.")

def insert_transaction(symbol, action, price, amount, timestamp, trade_id=None, rsi=None, adx=None, atr=None, 
                      relative_volume=None, divergence=None, bb_position=None, confidence=None):
    """Inserta una transacción en la tabla transactions_new."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr, 
                                      relative_volume, divergence, bb_position, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr, relative_volume, divergence, 
          bb_position, confidence))
    
    conn.commit()
    conn.close()

def fetch_all_transactions():
    """Obtiene todas las transacciones de la tabla transactions_new."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM transactions_new")
    transactions = cursor.fetchall()
    
    conn.close()
    return transactions

# Inicializar la base de datos al cargar el módulo
if __name__ == "__main__":
    initialize_db()