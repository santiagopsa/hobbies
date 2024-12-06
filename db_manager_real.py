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
    Inserta una nueva transacción en la base de datos con reintentos en caso de bloqueo.
    """
    max_retries = 5
    retry_delay = 2  # Segundos entre reintentos
    attempt = 0

    while attempt < max_retries:
        try:
            conn = sqlite3.connect(DB_NAME, timeout=10)  # Añadimos un timeout
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (symbol, action, price, amount, timestamp, profit_loss, confidence_percentage, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, price, amount, timestamp, profit_loss, confidence_percentage, summary))
            conn.commit()
            print(f"✅ Transacción insertada: {symbol}, {action}, {amount}")
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                attempt += 1
                print(f"⚠️ La base de datos está bloqueada. Intentando nuevamente ({attempt}/{max_retries})...")
                time.sleep(retry_delay)  # Esperar antes de reintentar
            else:
                print(f"❌ Error al insertar la transacción: {e}")
                break
        finally:
            try:
                conn.close()
            except Exception:
                pass
    else:
        print("❌ No se pudo insertar la transacción después de varios intentos.")

        
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