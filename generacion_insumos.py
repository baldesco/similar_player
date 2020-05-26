import numpy as np
import pandas as pd
import pickle
from os import path

datasets_dir = '../../datasets/'

fifa_20_path = path.join(datasets_dir, 'fifa-20-complete-player-dataset', 'players_20.csv')
columnas_path = path.join(datasets_dir, 'fifa-20-complete-player-dataset', 'columnas_jugadores.csv')

players = pd.read_csv(fifa_20_path)
cols_df = pd.read_csv(columnas_path)

players.dob = pd.to_datetime(players.dob)

# Columnas de interés de la base de jugadores
cols_id = list(cols_df[pd.notna(cols_df.id)].Columna)
cols_comp = list(cols_df[pd.notna(cols_df.comp)].Columna)
cols_comp_simple = list(cols_df[pd.notna(cols_df.comp_simple)].Columna)
cols_perf_field = list(cols_df[pd.notna(cols_df.perf_field)].Columna)
cols_perf_field_simple = list(cols_df[pd.notna(cols_df.perf_field_simple)].Columna)
cols_perf_goal = list(cols_df[pd.notna(cols_df.perf_goal)].Columna)
cols_control = list(cols_df[pd.notna(cols_df.control)].Columna)

x_comp = players[cols_comp_simple]
x_id = players[cols_id]
x_control = players[cols_control]

## Adecuación de variables ##

# Pie dominante como una variable dicotómica
x_comp['right_foot'] = pd.Series(np.where(x_comp.preferred_foot == 'Right', 1, 0))
del x_comp['preferred_foot']

# Armar diccionario con las dataframes, para guardarlo como objeto
datos_jugadores = {
    'x_comp': x_comp,
    'x_id': x_id,
    'x_control': x_control
}

with open('insumos/data.pickle', 'wb') as handle:
    pickle.dump(datos_jugadores, handle, protocol=pickle.HIGHEST_PROTOCOL)

