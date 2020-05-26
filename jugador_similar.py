import numpy as np
import pandas as pd
import pickle
from datetime import datetime as dt

# Cargar objetos guardados previamente
with open('insumos/data.pickle', 'rb') as handle:
    datos_jugadores = pickle.load(handle)

# Extraer objetos del diccionario
x_comp = datos_jugadores['x_comp']
x_id = datos_jugadores['x_id']
x_control = datos_jugadores['x_control'] 

# Función para obtener la edad (en días) el día de hoy para una fecha de nacimiento en particular 
def get_age_days(dob, now_date=dt.now()):
    return (now_date - dob).days

# Obtener edad de los jugadores (en días)
x_comp['edad'] = x_comp.dob.apply(get_age_days, args=(dt.now(),))
del x_comp['dob']

# Mínimos y máximos de cada columna, para hacer normalización min max
minimos = x_comp.min()
maximos = x_comp.max()

# Normalización min max de los datos de jugadores
normalized_df = np.array((x_comp - minimos) / (maximos - minimos))

# Datos de ejemplo (ingresados por el usuario)
in_height = 175
in_weight = 80
in_right_foot = 1
in_dob = '1994-02-25'
in_edad = get_age_days(pd.to_datetime(in_dob))

# Vectorizar y normalizar datos de entrada
entrada = [in_height, in_weight, in_right_foot, in_edad]
entrada_norm = np.array((entrada - minimos) / (maximos - minimos))

# Función para calcular distancias euclideanas entre vectores
def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

# Calcular distancias entre datos de entrada y todos los jugadores
distancias = np.apply_along_axis(euclidean_distance, 1, normalized_df, entrada_norm)

# Mostrar información del jugador más similar
similar_player_index = np.argmin(distancias)
x_id.loc[similar_player_index,:]
x_control.loc[similar_player_index,:]

# Url de la página de info sobre el jugador
x_id.loc[similar_player_index,'player_url']

