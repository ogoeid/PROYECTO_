# EJERCICIO 1 - Cargar y explorar datos de estudiantes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Configuración general
pd.set_option('display.max_columns', None)
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    # Not running inside IPython/Jupyter — ignore
    pass

# === 1. CARGA DEL DATASET ===
# Construye la ruta al archivo CSV de forma robusta, relativa a la ubicación del script.
try:
    # Obtiene el directorio donde se encuentra este script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- CAMBIO: Nombre del archivo actualizado a 'student-mat.csv' ---
    nombre_archivo = "student-mat.csv"
    ruta = os.path.join(script_dir, "datos", "ejer_1", nombre_archivo)
    
    print(f"Intentando cargar el dataset desde: {ruta}")
    # Nota: El archivo student-mat.csv suele usar ';' como separador. Si da error, prueba con sep=','
    df = pd.read_csv(ruta, sep=';')
    print("✅ Dataset cargado exitosamente.")

except FileNotFoundError:
    # Si el archivo no se encuentra, muestra un error claro y detiene el script.
    ruta_esperada = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "ejer_1", "student-mat.csv")
    print(f"❌ ERROR: No se pudo encontrar el archivo en la ruta esperada:\n{ruta_esperada}")
    print(f"\nPor favor, asegúrate de que el archivo '{nombre_archivo}' exista en la carpeta 'datos/ejer_1'.")
    sys.exit() # Detiene la ejecución.
except Exception as e:
    print(f"❌ Ocurrió un error inesperado al cargar el archivo: {e}")
    sys.exit()


# === 2. EXPLORACIÓN BÁSICA ===
print("=== DIMENSIONES DEL DATASET ===")
print(df.shape)

print("\n=== INFORMACIÓN GENERAL ===")
df.info()

print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
# La función display solo funciona en entornos como Jupyter/IPython.
# Usaremos print para compatibilidad general.
try:
    display(df.describe(include='all').T)
except NameError:
    print(df.describe(include='all').T)

# === 3. VALORES NULOS ===
print("\n=== VALORES NULOS ===")
valores_nulos = df.isnull().sum()
print(valores_nulos[valores_nulos > 0])

# === 4. GRÁFICO DE EJEMPLO CON NUEVOS DATOS ===
# Graficar la distribución de edades de los estudiantes
if 'age' in df.columns:
    plt.figure(figsize=(8,5))
    df['age'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribución de edades de los estudiantes')
    plt.xlabel('Edad')
    plt.ylabel('Cantidad de estudiantes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("⚠️ No se encontró una columna llamada 'age' en el dataset.")
