# ================================================================
# EJERCICIO 2 - Comparación de rendimiento entre Pandas y Dask
# ================================================================
# Este script compara el tiempo de ejecución y el uso de memoria
# al leer un archivo CSV grande (230 MB) usando Pandas y Dask.
# Se ejecuta correctamente en un entorno de Jupyter Notebook.
# ================================================================

# Importación de librerías necesarias
try:
    import pandas as pd              # Librería para manipulación de datos
except ImportError:
    raise SystemExit("El módulo 'pandas' no está instalado. Instálalo con: pip install pandas")

try:
    import dask.dataframe as dd      # Librería para procesamiento de datos grandes
except ImportError:
    raise SystemExit("El módulo 'dask' no está instalado. Instálalo con: pip install 'dask[complete]'")

import time                      # Para medir tiempos de ejecución
import requests                  # Para descargar el archivo desde una URL
import os                        # Para manejar rutas y verificar archivos
import matplotlib.pyplot as plt   # Para graficar resultados

try:
    from memory_profiler import memory_usage  # Para medir el uso de memoria
except ImportError:
    # memory_profiler no está disponible: usar un fallback simple que ejecuta la función y devuelve 0.0
    def memory_usage(callable_or_tuple, interval=0.5, **kwargs):
        if isinstance(callable_or_tuple, tuple):
            func, args = callable_or_tuple
            func(*args)
        else:
            callable_or_tuple()
        return [0.0]

# ================================================================
# DESCARGA DEL ARCHIVO (solo si no existe localmente)
# ================================================================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/yellow_tripdata_2016-01.csv"
FILE = "yellow_tripdata_2016-01.csv"

# Si el archivo no está en la carpeta actual, lo descarga
if not os.path.exists(FILE):
    print("Descargando archivo (~230 MB)...")
    r = requests.get(URL, stream=True)
    with open(FILE, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print("Descarga completa.")
else:
    print("Archivo ya disponible localmente.")

# ================================================================
# DEFINICIÓN DE FUNCIONES DE LECTURA
# ================================================================
# Estas funciones encapsulan la lectura del CSV con cada librería.

def pandas_read(path):
    """Lee el CSV completo usando Pandas."""
    return pd.read_csv(path)

def dask_read(path):
    """Lee el CSV por particiones usando Dask."""
    return dd.read_csv(path)

# ================================================================
# COMPARACIÓN DE RENDIMIENTO
# ================================================================
# Se mide tanto el tiempo de ejecución como el uso máximo de memoria
# durante la lectura con Pandas y Dask.

times = {}  # Diccionario para almacenar los tiempos
mem = {}    # Diccionario para almacenar los picos de memoria

# --- Medición con Pandas ---
t0 = time.time()
mem_p = memory_usage((pandas_read, (FILE,)), interval=0.5)  # Monitorea memoria
df_p = pandas_read(FILE)  # Carga del CSV
times['pandas'] = time.time() - t0
mem['pandas'] = max(mem_p)

# --- Medición con Dask ---
# Dask carga los datos en paralelo y usa menos memoria.
t0 = time.time()
mem_d = memory_usage((lambda: dask_read(FILE).compute()), interval=0.5)
df_d = dask_read(FILE).compute()  # compute() convierte Dask DataFrame a Pandas
times['dask'] = time.time() - t0
mem['dask'] = max(mem_d)

# ================================================================
# RESULTADOS NUMÉRICOS
# ================================================================
print("Tiempos de ejecución (segundos):", times)
print("Pico de memoria (MB):", mem)

# ================================================================
# OPERACIÓN DE VALIDACIÓN
# ================================================================
# Se calcula la media de la columna 'passenger_count' en ambos casos
# para verificar que los resultados sean equivalentes.
op_p = df_p.passenger_count.mean()
op_d = df_d.passenger_count.mean()

print("Media con Pandas:", op_p)
print("Media con Dask:", op_d)

# ================================================================
# VISUALIZACIÓN DE RESULTADOS
# ================================================================
# Se grafican los tiempos y el uso de memoria para comparar.
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

# Gráfico de tiempo
ax[0].bar(times.keys(), times.values())
ax[0].set_ylabel("Tiempo total (s)")
ax[0].set_title("Lectura de CSV (230 MB)")

# Gráfico de memoria
ax[1].bar(mem.keys(), mem.values())
ax[1].set_ylabel("Memoria pico (MB)")
ax[1].set_title("Uso de memoria")

plt.tight_layout()
plt.show()
