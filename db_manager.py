import sqlite3

DB_NAME = "trading.db"

def initialize_db():
    """
    Crea la base de datos y la tabla si no existen.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL,
            profit_loss REAL,
            confidence_percentage REAL,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_transaction(symbol, action, price, amount, timestamp, profit_loss=None, confidence_percentage=None, summary=None):
    """
    Inserta una nueva transacción en la base de datos.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions (symbol, action, price, amount, timestamp, profit_loss, confidence_percentage, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, action, price, amount, timestamp, profit_loss, confidence_percentage, summary))
    conn.commit()
    conn.close()

def fetch_all_transactions():
    """
    Recupera todas las transacciones almacenadas.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions")
    rows = cursor.fetchall()
    conn.close()
    return rows

def upgrade_db_schema():
    """
    Verifica y actualiza el esquema de la base de datos si faltan columnas.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Obtener la lista de columnas existentes
    cursor.execute("PRAGMA table_info(transactions)")
    existing_columns = [info[1] for info in cursor.fetchall()]

    # Verificar y agregar columnas si no están presentes
    if "confidence_percentage" not in existing_columns:
        cursor.execute("ALTER TABLE transactions ADD COLUMN confidence_percentage REAL")
        print("✅ Columna 'confidence_percentage' añadida.")

    if "summary" not in existing_columns:
        cursor.execute("ALTER TABLE transactions ADD COLUMN summary TEXT")
        print("✅ Columna 'summary' añadida.")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    print(fetch_all_transactions())
