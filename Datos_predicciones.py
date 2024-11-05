import pandas as pd
import re
# Seleccionamos el número de la jornada
jornada = 12

url= "https://fbref.com/es/comps/12/horario/Resultados-y-partidos-en-La-Liga"
tables = pd.read_html(url)
df = tables[0]
# seleccionamos las variables
df = df[['Sem.', 'Día', 'Fecha','Local','Visitante']]
## Hacemos la columna fecha del formato correspondiente
df["Fecha"] = pd.to_datetime(df["Fecha"])
# Hacemos el encoding de los días
dias_map = {
    'Lun': 1,
    'Mar': 2,
    'Mié': 3,
    'Jue': 4,
    'Vie': 5,
    'Sáb': 6,
    'Dom': 7
}
df["Día"] = df["Día"].map(dias_map)
# Filtramos por jornada
df = df[df["Sem."] == jornada]
# Agregamos la columnda de sede
df["Sedes"] = 1
# Renombramos las columnas
df = df[["Día","Sedes","Visitante","Local"]]
df = df.rename(columns = {"Local":"Anfitrion","Visitante":"Adversario"})
# Duplicamos el dataframe e invertimos las columnas para hacer la concatenacion
df_2 = df.copy()
df_2 = df_2.rename(columns={"Adversario":"Anfitrion","Anfitrion":"Adversario"})
df_2["Sedes"] = 0
df = pd.concat([df, df_2], ignore_index=True)
df["Día"] = df["Día"].astype(int)
### Estadisticas básicas
url= "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
tables = pd.read_html(url)
df_basic = tables[0]
df_basic=df_basic[['RL','Equipo','PG','PE', 'PP','GF','GC', 'xG', 'xGA','Últimos 5','Máximo Goleador del Equipo']]
df_basic['Máximo Goleador del Equipo'] = df_basic['Máximo Goleador del Equipo'].apply(lambda x: int(re.search(r'\b(\d+)\b', x).group(1)) if re.search(r'\b(\d+)\b', x) else None)
df_basic['Últimos 5'] = df_basic['Últimos 5'].apply(lambda resultados: sum([3 if resultado == 'PG' else (1 if resultado == 'PE' else 0) for resultado in resultados.split()]))
### Estadisticas de Ofensiva
url= "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
tables = pd.read_html(url)
df_ataque = tables[2]
df_ataque = df_ataque.drop(["Tiempo Jugado","Expectativa",'Por 90 Minutos'],axis=1)
df_ataque.columns = df_ataque.columns.droplevel(level=0)
df_ataque=df_ataque[['Equipo', 'Edad', 'Pos.','Ass','TPint', 'PrgC', 'PrgP']]
# Disparos
url= "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
tables = pd.read_html(url)
df_disparos = tables[8]
df_disparos.columns=df_disparos.columns.droplevel(level=0)
df_disparos=df_disparos[['Equipo', '% de TT','Dist']]
df_ataque = pd.merge(df_ataque, df_disparos, left_on='Equipo', right_on='Equipo', how='left')
# Pases
url= "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
tables = pd.read_html(url)
df_pases = tables[10]
df_pases = df_pases.drop(["Cortos","Medios",'Largos','Expectativa'],axis=1)
df_pases.columns=df_pases.columns.droplevel(level=0)
df_pases=df_pases[['Equipo', '% Cmp','Dist. tot.']]
df_ataque = pd.merge(df_ataque, df_pases, left_on='Equipo', right_on='Equipo', how='left')
### Estadisticas de defensa
url= "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
tables = pd.read_html(url)
df_porteria = tables[4]
df_porteria = df_porteria.drop(["Tiempo Jugado","Tiros penales"],axis=1)
df_porteria.columns = df_porteria.columns.droplevel(level=0)
df_porteria=df_porteria[['Equipo', 'GC', 'DaPC', 'Salvadas','PaC']]
url= "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
tables = pd.read_html(url)
df_defensa = tables[16]
df_defensa = df_defensa.drop(['Desafíos'],axis=1)
df_defensa.columns=df_defensa.columns.droplevel(level=0)
df_defensa=df_defensa[['Equipo', 'TklG','Int','Err']]
df_final = pd.merge(df_ataque, df_defensa, left_on='Equipo', right_on='Equipo', how='left')
df_final = pd.merge(df_final, df_basic, left_on='Equipo', right_on='Equipo', how='left')
df_opp=df_final.copy()
df_tm=df_final.copy()
# Renombramos las columnas
columns_to_rename = ['Edad', 'Pos.', 'Ass', 'TPint', 'PrgC', 'PrgP', '% de TT',
       'Dist', '% Cmp', 'Dist. tot.', 'TklG', 'Int', 'Err', 'RL', 'PG', 'PE',
       'PP', 'GF', 'GC', 'xG', 'xGA', 'Últimos 5',
       'Máximo Goleador del Equipo']
new_column_names_tm = [f"{col}(tm)" for col in columns_to_rename]
df_tm.rename(columns=dict(zip(columns_to_rename, new_column_names_tm)), inplace=True)
columns_to_rename = ['Edad', 'Pos.', 'Ass', 'TPint', 'PrgC', 'PrgP', '% de TT',
       'Dist', '% Cmp', 'Dist. tot.', 'TklG', 'Int', 'Err', 'RL', 'PG', 'PE',
       'PP', 'GF', 'GC', 'xG', 'xGA', 'Últimos 5',
       'Máximo Goleador del Equipo']
new_column_names_opp = [f"{col}(opp)" for col in columns_to_rename]
df_opp.rename(columns=dict(zip(columns_to_rename, new_column_names_opp)), inplace=True)
df = pd.merge(df, df_opp, left_on='Adversario', right_on='Equipo', how='left')
df = pd.merge(df, df_tm, left_on='Anfitrion', right_on='Equipo', how='left')
df=df.drop(['Equipo_x','Equipo_y'],axis=1)
df.head()