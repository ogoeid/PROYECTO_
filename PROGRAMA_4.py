# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Configuración opcional para mejorar la estética de los gráficos de Seaborn
sns.set_theme(style="whitegrid")
# La ruta de tu archivo
file_path = r'C:\Users\diego\Desktop\PROYECTO_\datos\ejer_1\student-por.csv'

# Cargar los datos especificando el separador ';'
try:
    df = pd.read_csv(file_path, sep=';')
    print("Dataset cargado con éxito.")
    print("Forma del dataset (filas, columnas):", df.shape)
    
    # Mostrar las primeras filas para verificar
    print("\nPrimeras 5 filas del dataset:")
    print(df.head())# en el jupiter poner display 
    
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo en la ruta:\n{file_path}")
# Crear un histograma con Matplotlib
plt.figure(figsize=(10, 6)) # Define el tamaño de la figura

plt.hist(df['G3'], bins=20, color='skyblue', edgecolor='black')

# Añadir títulos y etiquetas para mayor claridad
plt.title('Distribución de las Notas Finales (G3)', fontsize=16)
plt.xlabel('Nota Final (G3)', fontsize=12)
plt.ylabel('Frecuencia (Número de Estudiantes)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.show()
# Crear un boxplot con Seaborn
plt.figure(figsize=(12, 7)) # Define el tamaño de la figura

sns.boxplot(x='studytime', y='G3', data=df)

# Añadir títulos y etiquetas
plt.title('Notas Finales (G3) vs. Tiempo de Estudio Semanal', fontsize=16)
plt.xlabel('Tiempo de Estudio (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)', fontsize=12)
plt.ylabel('Nota Final (G3)', fontsize=12)
### 3. Plotly: Correlación entre Notas con Detalles Interactivos

 #**Plotly** es una librería ideal para crear gráficos interactivos listos para la web. La usaremos para generar un **gráfico de dispersión** que explore la relación entre las notas del primer período (`G1`) y las notas finales (`G3`). La interactividad nos permitirá pasar el mouse sobre cada punto para ver información adicional.
# Mostrar el gráfico
plt.show()
# Crear un gráfico de dispersión interactivo con Plotly Express
fig = px.scatter(
    df,
    x='G1',
    y='G3',
    color='sex',  # Colorear los puntos según el sexo del estudiante
    hover_data=['absences', 'studytime', 'age'], # Datos que aparecen al pasar el mouse
    title='Relación entre Notas del Primer Período (G1) y Finales (G3)'
)

# Mejorar el diseño de los ejes y el título
fig.update_layout(
    xaxis_title='Nota del Primer Período (G1)',
    yaxis_title='Nota Final (G3)',
    legend_title='Sexo'
)

# Mostrar el gráfico interactivo
fig.show()