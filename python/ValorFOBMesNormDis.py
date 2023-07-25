
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats



nombreExcel = 'excels\Impo-Franca9007.xlsx'
nombreHoja ='Valor FOB'

#Graficas de distribucion normal del valor FOB
# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de valor FOB por mes
promedio_valor_fob = df['Valor FOB']

media = np.mean(promedio_valor_fob)
varianza = np.var(promedio_valor_fob)
desviacion_estandar = np.std(promedio_valor_fob)
print("Valor del FOB")
print("Media:", media)
print("Varianza:", varianza)
print("Desviación Estándar:", desviacion_estandar)
print("\n")


# Transformación logarítmica
valor_fob_transformado = np.log(promedio_valor_fob)

# Ajuste de la distribución normal
mu, sigma = stats.norm.fit(valor_fob_transformado)

# Crear el histograma con los datos transformados
plt.hist(valor_fob_transformado, bins=8, density=True, alpha=0.5, color="green")

# Crear la línea de la distribución normal
x = np.linspace(valor_fob_transformado.min(), valor_fob_transformado.max(), 100)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y, color="seagreen", linewidth=2)  # Usamos "mediumorchid" como tonalidad de violeta

# Configuración del gráfico
plt.xlabel("Valor FOB (Transformado)")
plt.ylabel("Densidad")
plt.title("Transformación Logarítmica y Distribución Normal del Valor FOB")
plt.legend(["Distribución Normal", "Datos Transformados"])
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

"""
#Graficas de valor FOB promedio por mes

# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de valor FOB por mes
promedio_valor_fob = df.groupby('Mes')['Valor FOB'].mean()

# Cálculo de la media y la desviación estándar
media = np.mean(promedio_valor_fob)
desviacion_estandar = np.std(promedio_valor_fob)

# Cálculo de los valores de densidad de probabilidad
rango_valor = np.linspace(promedio_valor_fob.min(), promedio_valor_fob.max(), 100)
densidad_probabilidad = (1 / (desviacion_estandar * np.sqrt(2 * np.pi))) * np.exp(-((rango_valor - media) ** 2) / (2 * desviacion_estandar ** 2))

# Creación del gráfico
plt.plot(rango_valor, densidad_probabilidad, label='Distribución normal', color="purple")  # Usamos "purple" como color violeta
plt.hist(promedio_valor_fob, density=True, bins=8, alpha=0.5, color="mediumorchid", label='Datos observados')  # Usamos "mediumorchid" como tonalidad de violeta

# Etiquetas de los ejes y título del gráfico
plt.xlabel('Promedio Valor FOB')
plt.ylabel('Densidad de probabilidad')
plt.title('Distribución normal del Valor FOB por Mes')

# Mostrar leyenda
plt.legend()

# Mostrar gráfico
plt.show()










# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de valor FOB por mes
promedio_valor_fob = df.groupby('Mes')['Valor FOB'].mean()

media = np.mean(promedio_valor_fob)
varianza = np.var(promedio_valor_fob)
desviacion_estandar = np.std(promedio_valor_fob)
print("Valor promedio por mes del FOB")
print("Media:", media)
print("Varianza:", varianza)
print("Desviación Estándar:", desviacion_estandar)


# Transformación logarítmica
valor_fob_transformado = np.log(promedio_valor_fob)

# Ajuste de la distribución normal
mu, sigma = stats.norm.fit(valor_fob_transformado)

# Crear el histograma con los datos transformados
plt.hist(valor_fob_transformado, bins=8, density=True, alpha=0.5, color="steelblue")

# Crear la línea de la distribución normal
x = np.linspace(valor_fob_transformado.min(), valor_fob_transformado.max(), 100)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y, color="orange", linewidth=2)

# Configuración del gráfico
plt.xlabel("Valor FOB (Transformado)")
plt.ylabel("Densidad")
plt.title("Transformación Logarítmica y Distribución Normal del Valor promedio FOB por Mes")
plt.legend(["Distribución Normal", "Datos Transformados"])
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
"""