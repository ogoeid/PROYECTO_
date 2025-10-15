#aqui va el script del problema 3
#AQUI VA EL SCRIPT DEL EJERCICIO 3
#AQUI
# Script para el Ejercicio 3: Comparación Pandas vs PySpark con el dataset AirQualityUCI
# 1 Importación de Librerias
import pandas as pd
import numpy as np
#  2Ruta del Archivo
#la 'r' al inicio es importante en Windows/SEGUN el video, se recomienda eso, igual puedo poner que recise la existencia para luego descargar 
CSV_FILE = r'C:\Users\diego\Desktop\PROYECTO_\datos\ejer_3\AirQualityUCI.csv'
# 3 Funciones de Procesamiento
def limpiar_data(df):
    """
    Realiza una limpieza exhaustiva del DataFrame de Calidad del Aire.
    - Elimina columnas y filas vacías.
    - Reemplaza el marcador de nulos (-200) por NaN.
    - Convierte las columnas de fecha y hora a un índice de tipo datetime.
    """
    print("[Limpieza] Inciando limpieza de datos...")
    #Eliminar las dos columnas(creo que estan vacias)
    df = df.iloc[:, :-2]
    #Eliminar filas donde todos los valores son nulo
    df.dropna(how='all', inplace=True)
    # Reemplazar el valor -200 (marcador de nulos específico de este dataset) con NaN de numpy
    df.replace(to_replace=-200, value=np.nan, inplace=True)
    # Combinar Date y Time en una sola columna de tipo datetime para análisis de series de tiempo
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df.set_index('DateTime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    print("[Limpieza] Limpieza completada.")
    return df
def _pandas(df):
    """
    Realiza las operaciones de filtrado, agrupamiento e imputación con Pandas
    y muestra los resultados en la consola.
    """
    print("\n" + "="*50)
    print("ANALISIS CON PANDAS")
    print("="*50)
    # 1 filtro que puede servir: dias con temperatura mayor a 25C
    print("\n[PANDAS] 1. Filtrando registros con Temperatura > 25C...")
    dias_calurosos = df[df['T'] > 25]
    print(f"Se encontraron {len(dias_calurosos)} registros/horas con T > 25°C.")
    print("Mostrando los 5 primeros registros de días calurosos:")
    # .to_string() asegura que se vea bien en la terminal o que se vea, n0 logro que se vea bien 
    print(dias_calurosos[['T', 'RH', 'AH']].head().to_string())
    print("-" * 50)
    # 2 calcular y mostrar la media de tem  diaria? o de algun lugar especifico (benceno (C6H6)(porque este???))
    print("\n[PANDAS] 2. Calculando la media diaria de Benceno (C6H6(GT))...")
    # .resample('D') es una poderosa función para agrupar por día
    media_diaria_benceno = df['C6H6(GT)'].resample('D').mean()
    print("Mostrando la media de los últimos 5 días con registros:")
    print(media_diaria_benceno.dropna().tail().to_string())
    print("-" * 50)
    # 3 Benceno(no encerio porque este?)mostramos la media, mediana y el metodo ffill para hacer algo con los nulos, deja buscar que es ffill
    # ffill es fill forward, osea que llena con el ultimo valor conocido, osea que si hay nulos seguidos, los llena con el ultimo valor que no es nulo
    #aver que resulta(no le tengo mucha fe(lol))
    print("\n[PANDAS] 3 realizando imputación en la columna 'C6H6(GT)'...")
    columna_benceno = df[['C6H6(GT)']].copy()
    columna_benceno['C6H6_con_media'] = columna_benceno['C6H6(GT)'].fillna(columna_benceno['C6H6(GT)'].mean())
    columna_benceno['C6H6_con_mediana'] = columna_benceno['C6H6(GT)'].fillna(columna_benceno['C6H6(GT)'].median())
    columna_benceno['C6H6_con_ffill'] = columna_benceno['C6H6(GT)'].fillna(method='ffill')
    print("\nVerificación de imputación (mostrando 5 filas donde el valor original era nulo):")
    print(columna_benceno[columna_benceno['C6H6(GT)'].isnull()].head().to_string())
#dejo esto o lo paso al java?/igual lo dejo expresado en el java(tengo muchas dudas si se puede java y python en el mismo proyecto)
def _pyspark():
    """
    como seria, muestra de lo que seria el codigo en pyspark
    """
    print("\n" + "="*50)
    print("EJEMPLO DE CÓDIGO(compraracion)")
    print("="*50)
    
    codigo_pyspark = """
# Para ejecutar esto, se necesita una instal de Java y Spark.(por ahora eso es lo necesario)
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, avg, to_date
# spark = SparkSession.builder.appName("AirQualitySpark").getOrCreate()

# # lectura del csv para el spark.manejo de decimales con ','
# # puede requerir filtrar data.(segun la guia(video))
# df_spark = spark.read.csv(
#     r'C:\\Users\\diego\\Desktop\\PROYECTO_\\datos\\ejer_3\\AirQualityUCI.csv',
#     header=True,
#     sep=';',
#     inferSchema=True
# ).replace(-200, None) # Reemplazar -200 por nulos

# # convertir columna de fecha y agrupar /utilizar la misma locacion de benceno 
# df_spark = df_spark.withColumn('Date', to_date(col('Date'), 'dd/MM/yyyy'))
# media_diaria_benceno_spark = df_spark.groupBy('Date').agg(avg('C6H6(GT)').alias('media_benceno'))

# print("Resultado de PySpark (media diaria de Benceno):")
# media_diaria_benceno_spark.orderBy('Date', ascending=False).show(5)
# spark.stop()
    """
    print(codigo_pyspark)
#4 Ejecución!(esta bien escrito/creo)
def main():
    """
    Funcin principal que orquesta la ejecución del script.
    """
    try:
        # Carga de datos especificos 
        print(f"Cargando dataset desde: {CSV_FILE}...")
        air_df = pd.read_csv(CSV_FILE, sep=';', decimal=',')
        # Limpieza de basura
        air_df_limpio = limpiar_data(air_df)
        print(f"Dimensiones del dataset limpio: {air_df_limpio.shape}")
        # se hace analisis
        _pandas(air_df_limpio)
        # Mostrar el código de ejemplo de PySpark
        _pyspark()
    except FileNotFoundError:
        print(f"error: No se encontro el archivo.")
        print(f"Ruta: '{CSV_FILE}'")
        print("Por favor, verifica que la ruta sea correcta y que el archivo exista.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
if __name__ == "__main__":
    main()
    print("\n--- Script Finalizado ---")