import sqlite3

# Cambiar el nombre de la base de datos para el entorno real
DB_NAME = "trading_real.db"

def initialize_db():
    """
    Crea la base de datos y la tabla si no existen.
    """
    try:
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
    except sqlite3.Error as e:
        print(f"❌ Error al inicializar la base de datos: {e}")
    finally:
        conn.close()

def insert_transaction(symbol, action, price, amount, timestamp, profit_loss=None, confidence_percentage=None, summary=None):
    """
    Inserta una nueva transacción en la base de datos.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions (symbol, action, price, amount, timestamp, profit_loss, confidence_percentage, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, action, price, amount, timestamp, profit_loss, confidence_percentage, summary))
        conn.commit()
    except sqlite3.Error as e:
        print(f"❌ Error al insertar la transacción: {e}")
    finally:
        conn.close()

def fetch_all_transactions():
    """
    Recupera todas las transacciones almacenadas.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions")
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"❌ Error al recuperar transacciones: {e}")
        return []
    finally:
        conn.close()

def upgrade_db_schema():
    """
    Verifica y actualiza el esquema de la base de datos si faltan columnas.
    """
    try:
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
    except sqlite3.Error as e:
        print(f"❌ Error al actualizar el esquema de la base de datos: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print(fetch_all_transactions())  # Asegúrate de que la base de datos esté configurada