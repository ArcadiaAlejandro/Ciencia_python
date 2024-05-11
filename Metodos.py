"""
Método: capitalize()
Descripcion: CONVERTIR A MAYÚSCULA LA PRIMERA LETRA
"""
cadena = "bienvenido a mi aplicación" 
resultado = cadena.capitalize()
print(resultado)

#  Resultado  -> Bienvenido a mi aplicación

"""
Método: lower()
Descripcion: CONVERTIR UNA CADENA A MINÚSCULA
"""
cadena = "Hola Mundo" 
resultado = cadena.lower()
print(resultado)

#  Resultado  -> hola mundo

"""
Método:  upper()
Descripcion: CONVERTIR UNA CADENA A MAYÚSCULAS
"""
cadena = "hola mundo" 
resultado = cadena.upper()
print(resultado)
#  Resultado  -> HOLA MUNDO


"""
Método:  swapcase()
Descripcion: INVERTIR LAS MAYÚSCULAS A MINÚSCULAS Y MINÚSCULAS A  MAYÚSCULAS 
"""
cadena = "Hola Mundo" 
resultado = cadena.swapcase()
print(resultado)
#  Resultado  -> hOLA mUNDO

"""
Método:  title()
Descripcion: CONVERTIR UNA CADENA EN FORMATO TÍTULO
"""
cadena = "hola mundo" 
resultado = cadena.title()
print(resultado)
#  Resultado  -> Hola Mundo

"""
Método:  center(longitud[cantidad, “caracter de relleno”])
Descripcion: CENTRAR UN TEXTO
"""
cadena = "bienvenido a mi aplicación".capitalize()
resultado = cadena.center(50, "=")
print(resultado)
#  Resultado  -> ===========Bienvenido a mi aplicación============ 

resultado = cadena.center(50, " ")
print(resultado)
#  Resultado  ->            Bienvenido a mi aplicación 

"""
Método:  : ljust(longitud[, “caracter de relleno”])
Descripcion: ALINEAR TEXTO A LA IZQUIERDA
"""




"""
Descripcion: Declaracion de variables en una linea
"""

a1 = 10 #declaracion de variable x linea
b2 = 20 #declaracion de variable x linea

a2, b1 = 30, 40 #asignamos a las variables en una misma linea

"""
Descripcion: Conversion de variables

int - Entero
float - Decimal
str - Cadena de texto
bool - Booleano
list - Lista 
tuple - Tupla
dict - Diccionario
set - Conjunto

"""

num = int("1")
type(num)

cadena = str(1)
type(cadena)

type("mensaje")
type(100)

valor = True
type(valor)


"""

Descripcion: USO DE ARREGLOS

list - Lista  |  Colección ordenada y mutable de elementos.
[1, 2, 3]

tuple - Tupla  |  Colección ordenada e inmutable de elementos.
(1, 2, 3)

dict - Diccionario | Colección de pares clave-valor.
{'nombre': 'Alice', 'edad': 25}

set - Conjunto  | Colección no ordenada de elementos únicos.
{1, 2, 3}, {'apple', 'banana', 'cherry'}
"""

nombres = ['max','jose','maria']
print(nombres[1])
len(nombres)
nombres.append('raul')
print(nombres)

"""
indexación a elementos individuales en estructuras de datos que pueden contener múltiples elementos

"""

#Indexación en Cadenas de Texto (Strings)   
#Descripcion: Puedes acceder a un carácter específico de una cadena mediante su índice. Los índices en Python comienzan en 0.9

texto = "Hola mundo"
primer_caracter = texto[0]  # 'H'
ultimo_caracter = texto[-1] # 'o' (último carácter usando índice negativo)

#Descripcion: obtener subcadenas utilizando la sintaxis de slicing [inicio:fin], donde inicio es incluido y fin es excluido.
subcadena = texto[1:4]  # 'ola'


# Indexación en Listas

lista = [10, 20, 30, 40, 50]
primer_elemento = lista[0]  # 10
ultimo_elemento = lista[-1] # 50

#Descripcion>  Slicing: También puedes usar slicing para obtener sublistas.

sublista = lista[1:3]  # [20, 30]

# Indexación en Tuplas

tupla = (1, 2, 3, 4, 5)
primer_elemento = tupla[0]  # 1

# Indexación en Diccionarios

diccionario = {'a': 1, 'b': 2, 'c': 3}
valor_a = diccionario['a']  # 1

"""
Valores estadisticos

"""

lista = [1,2,3,4,5,6,7,8,9,10]

len(lista) #devuelve la longitud de la lista
sum(lista) # uma todos los elementos de la lista n(n+1)/2
min(lista) # devuelve el valor mínimo en la lista
max(lista) # devuelve el valor maximo en la lista

rango = max(lista) - min(lista) #calcula el "rango" de los valores en la lista,

import statistics as s
lista = [1,2,3,4,5,6,7,8,9,10]
# Media> Calcula el promedio de los elementos de la lista
media = s.mean(lista)
print("Media:", media)  # Calcula el promedio de los elementos de la lista

# Mediana> Encuentra el valor medio de los números en la lista
mediana = s.median(lista) 
print("Mediana:", mediana)  # Encuentra el valor medio de los números en la lista


# Desviación Estándar> mide la cantidad de variación o dispersión de los datos.
desviacion_estandar = s.stdev(lista)
print("Desviación Estándar:", desviacion_estandar)  # Mide la cantidad de variación o dispersión de los datos

#NOTA: Si los datos estan en años, la desviancion estandar tambien estara en años

#Calcula los cuantiles de la lista, donde n=4 significa que quieres los cuantiles que dividen la distribución en cuatro partes iguales (es decir, los cuartiles). Esto devolverá tres puntos de corte: el primer cuartil (Q1), el segundo cuartil o mediana (Q2), y el tercer cuartil (Q3).

cuantiles = s.quantiles(lista, n=4) #4 CUARTILES \ 100 PERCENTILES
print("Cuartiles:", cuantiles)  # Calcula los cuantiles de la lista, donde n=4 significa cuartiles



"""
Función para Evaluar Media y Mediana, y Detectar Valores Atípicos
"""
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

def evaluar_y_detectar_atipicos(datos):
    """
    Evalúa si la media y la mediana son iguales, detecta valores atípicos en los datos,
    y devuelve los datos originales en una lista.
    
    Args:
    datos (list): Lista de datos numéricos.
    
    Returns:
    dict: Diccionario con la igualdad de media y mediana, lista de valores atípicos, y los datos originales.
    """
    media = stats.mean(datos)
    mediana = stats.median(datos)
    desviacion_estandar = stats.stdev(datos)
    media_igual_mediana = media == mediana
    
    # Calculando el rango intercuartílico (IQR)
    Q1 = np.percentile(datos, 25)
    Q3 = np.percentile(datos, 75)
    IQR = Q3 - Q1
    
    # Definir límites para valores atípicos
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Detectar valores atípicos
    atipicos = [x for x in datos if x < limite_inferior or x > limite_superior]
    
    return {
        'media_igual_mediana': media_igual_mediana,
        'media': media,
        'mediana': mediana,
        'valores_atipicos': atipicos,
        'datos_originales': datos,
        'desviacion_estandar': desviacion_estandar
    }

# Función para graficar los resultados estadísticos
def graficar_estadisticas(datos, media, mediana, desviacion_estandar, atipicos):
    plt.figure(figsize=(12, 18))

    # Histograma de los datos originales
    plt.subplot(3, 2, 1)
    plt.hist(datos, bins=10, color='lightblue', edgecolor='black')
    plt.title('Histograma de los Datos')
    plt.xlabel('Datos')
    plt.ylabel('Frecuencia')

    # Media
    plt.subplot(3, 2, 2)
    plt.hist(datos, bins=10, color='lightgrey', edgecolor='black')
    plt.axvline(media, color='red', label=f'Media: {media}')
    plt.title('Media de los Datos')
    plt.legend()

    # Mediana
    plt.subplot(3, 2, 3)
    plt.hist(datos, bins=10, color='lightgrey', edgecolor='black')
    plt.axvline(mediana, color='green', label=f'Mediana: {mediana}')
    plt.title('Mediana de los Datos')
    plt.legend()

    # Desviación Estándar
    plt.subplot(3, 2, 4)
    plt.hist(datos, bins=10, color='lightgrey', edgecolor='black')
    plt.axvline(media - desviacion_estandar, color='orange', linestyle='dashed', label='Media - 1 SD')
    plt.axvline(media + desviacion_estandar, color='orange', linestyle='dashed', label='Media + 1 SD')
    plt.title('Desviación Estándar de los Datos')
    plt.legend()

    # Valores Atípicos
    plt.subplot(3, 2, 5)
    plt.boxplot(datos, vert=False)
    plt.scatter(atipicos, [1]*len(atipicos), color='red', zorder=5)
    plt.title('Boxplot y Valores Atípicos')
    
    # Datos Originales
    plt.subplot(3, 2, 6)
    plt.plot(datos, marker='o', linestyle='-', color='blue')
    plt.title('Datos Originales')
    plt.xlabel('Índice')
    plt.ylabel('Valor')

    plt.tight_layout()
    plt.show()

# Datos de ejemplo y uso de las funciones
datos = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 100]  # Incluye un valor atípico evidente (100)
resultado = evaluar_y_detectar_atipicos(datos)
graficar_estadisticas(datos=resultado['datos_originales'],media=resultado['media'],mediana=resultado['mediana'],desviacion_estandar=resultado['desviacion_estandar'],atipicos=resultado['valores_atipicos'])

#for

lista = [10,20,30,40]
for i in lista:
    if i < 25:
        print(i,"<25")
    else:
        print(i,">25")
        
for k in range(10):
    print(k)
    
list(range(10))

#while

tmp = 1
while (tmp<=10):
    print(tmp)
    tmp = tmp + 1
    
#Imprimir argumentos de manera dinamica 

def imprimir(*argumentos):
    for arg in argumentos:
        print(arg)
        
imprimir(10,12,13,14,15)


"""
Función map()
La función map() toma dos argumentos principales: una función y un iterable. La función se aplica a cada elemento del iterable, y map() devuelve un objeto map (un iterador), que puede ser convertido fácilmente a una lista o a otro tipo de secuencia.

Sintaxis:
    map(función, iterable)
    
Función lambda()
Una función lambda en Python es una pequeña función anónima definida sin un nombre. Puedes definir una función lambda usando la palabra clave lambda. Las funciones lambda pueden tener cualquier número de argumentos, pero solo pueden tener una expresión.

Sintaxis:   
    lambda arguments: expression
"""
numeros = [1, 2, 3, 4, 5]
cuadrados = list(map(lambda x: x ** 2, numeros))
print(cuadrados)  # Salida: [1, 4, 9, 16, 25]

nombres = ["alice", "bob", "charlie"]
mayusculas = list(map(lambda x: x.upper(), nombres))
print(mayusculas)  # Salida: ['ALICE', 'BOB', 'CHARLIE']

precios = [100, 200, 300]
precios_con_impuesto = list(map(lambda x: x * 1.10, precios))
print(precios_con_impuesto)  # Salida: [110.0, 220.0, 330.0]

"""
Filter() y transformar
Podrías querer aplicar una transformación a elementos que cumplen cierta condición. Por ejemplo, elevar al cuadrado solo los números impares de una lista.
"""
numeros = [1, 2, 3, 4, 5, 6]
cuadrados_impares = list(map(lambda x: x ** 2, filter(lambda x: x % 2 != 0, numeros)))
print(cuadrados_impares)  # Salida: [1, 9, 25]


"""
La función filter() 
En Python es otra herramienta útil en programación funcional, que permite filtrar elementos de un iterable (como una lista) basándose en una función que comprueba si cada elemento cumple con un criterio específico. Los elementos que retornan True se incluyen en el resultado.
Sintaxis:
    filter(function, iterable)
"""

#Supongamos que quieres obtener solo los números impares de una lista. Puedes hacerlo usando filter() junto con una función lambda.

numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
impares = list(filter(lambda x: x % 2 != 0, numeros))
print(impares)  # Salida: [1, 3, 5, 7, 9]

#Eliminar valores vacíos de una lista
datos = ["Alice", "", "Bob", None, "Charlie", "  ", "Dave"]
datos_filtrados = list(filter(None, datos))
print(datos_filtrados)  # Salida: ['Alice', 'Bob', 'Charlie', 'Dave']

#Filtrar números mayores de un cierto valor
numeros = [10, 20, 30, 40, 50, 60]
mayores_de_30 = list(filter(lambda x: x > 30, numeros))
print(mayores_de_30)  # Salida: [40, 50, 60]

#Filtrar palabras que contienen una letra específica
palabras = ["manzana", "banana", "cereza", "datil"]
contiene_a = list(filter(lambda palabra: 'a' in palabra, palabras))
print(contiene_a)  # Salida: ['manzana', 'banana', 'cereza', 'datil']


# Crear un array bidimensional (matriz)
import numpy as np

matriz = np.array([[1, 2, 3], [4, 5, 6]])
print('Matriz: ' , matriz)
print('Dimensiones: ' , matriz.ndim) #numero de dimensiones
print('Matriz: (cantidad de filas, cantidad de columnas en la matriz) ' , matriz.shape) #cantidad de filas y columnas
print('Dimensiones: ' , matriz.size) #cantidad de elemntos del arreglo
print('Transpuesta de la matriz: ' , matriz.T) #Invierte la matriz en la transpsuesta

print('Acceso Completo: ',matriz[:,:]) #Esto imprime todos los elementos de la matriz. : se usa para seleccionar todos los elementos en esa dimensión.
print('Acceso a un Elemento Específico: ',matriz[1,2]) #Accede al elemento en la segunda fila y tercera columna, que es 6 
print('Acceso a una Columna Específica',matriz[:,0])#Esto imprime todos los elementos de la primera columna. El resultado será [1, 4]. 



# Sumar elementos de dos arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b
print('Suma de Array: ', c)  # Output: [5 7 9]

#Array de un Valor Constante:
siete = np.full((3, 5), 7)
print('Array de sietes:\n', siete)

#Array de Números Aleatorios:
aleatorios = np.random.random((2, 2))
print('Array de números aleatorios:\n', aleatorios)

#Acceder a la última fila de una matriz:
ultima_fila = matriz[-1]
print('Última fila:', ultima_fila)

#Acceder a una submatriz (las primeras dos columnas
submatriz = matriz[:, :2]
print('Submatriz:\n', submatriz)

#Crear un Array con un Valor Específico

array_de_sietes = np.full((3, 4), 7)
print(array_de_sietes)

# Crear un array de ceros
ceros = np.zeros((2, 3))
print('Array de 0: ',ceros)

# Crear un array de unos
unos = np.ones((3, 3))
print('Array de 1: ',unos)

# Crear una matriz identidad
identidad = np.eye(3)
print('Matriz Identidad: ',identidad)

# Crear un array con un rango de valores
rango = np.arange(10)
print('Array con un rango de valores: ',rango)

# Crear un array con valores espaciados uniformemente
espacio = np.linspace(0, 1, num=5)
print('array con valores espaciados uniformemente: ',espacio)

#==========================================================
#Estadisticas con Numpy se usa para algebra lineal

import numpy as np

datos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Media y mediana
print("Media:", np.mean(datos))
print("Mediana:", np.median(datos))

# Desviación estándar
print("Desviación estándar:", np.std(datos))

# Mínimo y máximo
print("Mínimo:", np.min(datos))
print("Máximo:", np.max(datos))

#Variacion 

print("Varacion:", np.var(datos))

#==========================================================
#Paquetes

"""
data_analysis/
│
├── __init__.py
├── reader.py        # Módulo para leer datos
├── processor.py     # Módulo para procesar datos
└── visualization.py # Módulo para visualizar datos
Cada uno de estos módulos puede definir funciones y clases relacionadas con sus propósitos respectivos.

Ventajas de Usar Paquetes
Modularidad: Facilita la organización de código en componentes separados con funcionalidades específicas.
Reusabilidad: Permite reutilizar código fácilmente en diferentes programas.
Espacio de Nombres: Evita conflictos entre nombres al agrupar módulos relacionados.
"""

#Instalacion de paquetes
#   pip install (nombre del paquete)

"""
pip install numpy            #Instala el paquete
pip show numpy               #Muestra la version del paquete
pip uninstall numpy          #Desinstala el paquete
pip freeze                   #Muesta paquetes isntalados

"""
#Uso de paquetes
#    importación y organización del código

from package import module1

#para importar una función específica de un submódulo

from package.subpackage.submodule1 import some_function

#Disponibilizar
# FORMA 1
import math as m  

"""
Importación de la biblioteca math con un alias: La primera línea import math as m importa la biblioteca estándar de Python llamada math, que proporciona acceso a funciones matemáticas definidas por el estándar C. Al utilizar as m, el código asigna un alias a la biblioteca math. Este alias es m, lo que significa que en lugar de escribir math.funcion(), puedes simplemente escribir m.funcion() para acceder a cualquier función disponible en la biblioteca math.
Uso de la función sqrt para calcular la raíz cuadrada: La segunda línea m.sqrt(16) llama a la función sqrt() del módulo math, pero utilizando el alias m. La función sqrt() calcula la raíz cuadrada del argumento proporcionado. En este caso, calcula la raíz cuadrada de 16, que es 4.
"""

# Calcular la raíz cuadrada de varios números
numeros = [4, 16, 25, 36]
raices = [m.sqrt(numero) for numero in numeros]

print(raices)  # Imprime: [2.0, 4.0, 5.0, 6.0]

# FORMA 2
import math 
math.sqrt(16)

# FORMA 3

from math import sqrt
sqrt(16)


#Comprension de listas

lista_en_blanco = []
for item in range(1,6):
    lista_en_blanco.append(item)
print(lista_en_blanco)

#usando comprension de listas
"""
Sintaxis: 
    [expression for item in iterable if condition]
"""
# Crear una lista mediante el uso de rango
lista2 = [item for item in range(1,6)]
print(lista2)
# Salida: [1, 2, 3, 4, 5]

# Crear una lista con los cuadrados de los primeros 10 números enteros.
cuadrados = [x**2 for x in range(1, 11)]
print(cuadrados)
# Salida: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Filtrar y crear una lista solo con los números pares.
pares = [x for x in range(1, 11) if x % 2 == 0]
print(pares)
# Salida: [2, 4, 6, 8, 10]

# Convertir temperaturas de Celsius a Fahrenheit.
celsius = [0, 10, 20, 34.5]
fahrenheit = [((9/5) * temp + 32) for temp in celsius]
print(fahrenheit)
# Salida: [32.0, 50.0, 68.0, 94.1]

# Crear una matriz plana a partir de una matriz 2D.
matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
planos = [num for fila in matriz for num in fila]
print(planos)
# Salida: [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Números divisibles por 2 y 3 en un rango.
divisibles = [x for x in range(1, 101) if x % 2 == 0 if x % 3 == 0]
print(divisibles)
# Salida: [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]

# Crear una lista con "Par" o "Impar" para los primeros 10 números.
par_impar = ["Par" if x % 2 == 0 else "Impar" for x in range(1, 11)]
print(par_impar)
# Salida: ['Impar', 'Par', 'Impar', 'Par', 'Impar', 'Par', 'Impar', 'Par', 'Impar', 'Par']

#Crear una lista de potencia utilizando los 5 primeros numeros
elevado_al_cuadrado = [item **2 for item in range (1,6)]
print(elevado_al_cuadrado)
# Salida: [1, 4, 9, 16, 25]

#Crear una lista que ponga en mayuscula las vocales
vocales = ['a','e','i','o','u']
vocales_up = [vocal.upper() for vocal in vocales]
print(vocales_up)
# Salida: ['A', 'E', 'I', 'O', 'U']
