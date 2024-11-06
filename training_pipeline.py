# Importamos las librerias
import pandas as pd
import re
import pickle
import mlflow
import dagshub
import pathlib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from prefect import task, flow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import RandomizedSearchCV

# Definimos el primer task que es actualizar el dataset
@task(name="Actualilzar dataset")
def actualizar_dataset(jornada:int) -> pd.DataFrame:
    import pandas as pd
    import re

    ## Seleccionamos el numero de jornada
    jornada = 11
    url = "https://fbref.com/es/comps/12/horario/Resultados-y-partidos-en-La-Liga"
    tables = pd.read_html(url)
    df = tables[0]
    # seleccionamos las variables
    df = df[['Sem.', 'Día', 'Fecha', 'Local', 'Visitante', 'Marcador']]
    # Filtramos por jornada
    df = df[df["Sem."] == jornada]
    # Obtenemos el marcador
    df[['GF', 'GC']] = df['Marcador'].str.split('–', expand=True)
    df['GF'] = df['GF'].astype(int)
    df['GC'] = df['GC'].astype(int)
    df.drop(columns=['Marcador'], inplace=True)
    ## Hacemos la columna fecha del formato correspondiente
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')
    df["Fecha"] = df["Fecha"].dt.strftime('%Y-%m-%d')
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
    # Agregamos la columnda de sede
    df["Sedes"] = 1
    # Renombramos las columnas
    df = df[["Fecha", "Día", "Sedes", "Visitante", "Local", "GF", "GC"]]
    df = df.rename(columns={"Local": "Anfitrion", "Visitante": "Adversario"})
    # Duplicamos el dataframe e invertimos las columnas para hacer la concatenacion
    df_2 = df.copy()
    df_2 = df_2.rename(columns={"Adversario": "Anfitrion", "Anfitrion": "Adversario", "GF": "GC", "GC": "GF"})
    df_2["Sedes"] = 0
    df = pd.concat([df, df_2], ignore_index=True)
    # Agregamos la columna resultado
    df['Resultado'] = df.apply(lambda row: 3 if row['GF'] > row['GC'] else (2 if row['GF'] == row['GC'] else 1), axis=1)
    # Cambiamos el tipo de dato
    df["Día"] = df["Día"].astype(int)
    ### Estadisticas básicas
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_basic = tables[0]
    df_basic = df_basic[
        ['RL', 'Equipo', 'PG', 'PE', 'PP', 'GF', 'GC', 'xG', 'xGA', 'Últimos 5', 'Máximo Goleador del Equipo']]
    df_basic['Máximo Goleador del Equipo'] = df_basic['Máximo Goleador del Equipo'].apply(
        lambda x: int(re.search(r'\b(\d+)\b', x).group(1)) if re.search(r'\b(\d+)\b', x) else None)

    df_basic['Últimos 5'] = df_basic['Últimos 5'].apply(lambda resultados: sum(
        [3 if resultado == 'PG' else (1 if resultado == 'PE' else 0) for resultado in resultados.split()]))
    ### Estadisticas de Ofensiva
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_ataque = tables[2]
    df_ataque = df_ataque.drop(["Tiempo Jugado", "Expectativa", 'Por 90 Minutos'], axis=1)
    df_ataque.columns = df_ataque.columns.droplevel(level=0)
    df_ataque = df_ataque[['Equipo', 'Edad', 'Pos.', 'Ass', 'TPint', 'PrgC', 'PrgP']]
    ##### Disparos
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_disparos = tables[8]
    df_disparos.columns = df_disparos.columns.droplevel(level=0)
    df_disparos = df_disparos[['Equipo', '% de TT', 'Dist']]
    df_ataque = pd.merge(df_ataque, df_disparos, left_on='Equipo', right_on='Equipo', how='left')
    ##### Pases
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_pases = tables[10]
    df_pases = df_pases.drop(["Cortos", "Medios", 'Largos', 'Expectativa'], axis=1)
    df_pases.columns = df_pases.columns.droplevel(level=0)
    df_pases = df_pases[['Equipo', '% Cmp', 'Dist. tot.']]
    df_ataque = pd.merge(df_ataque, df_pases, left_on='Equipo', right_on='Equipo', how='left')
    ### Estadisticas de defensa
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_porteria = tables[4]
    df_porteria = df_porteria.drop(["Tiempo Jugado", "Tiros penales"], axis=1)
    df_porteria.columns = df_porteria.columns.droplevel(level=0)
    df_porteria = df_porteria[['Equipo', 'GC', 'DaPC', 'Salvadas', 'PaC']]
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_defensa = tables[16]
    df_defensa = df_defensa.drop(['Desafíos'], axis=1)
    df_defensa.columns = df_defensa.columns.droplevel(level=0)
    df_defensa = df_defensa[['Equipo', 'TklG', 'Int', 'Err']]
    df_final = pd.merge(df_ataque, df_defensa, left_on='Equipo', right_on='Equipo', how='left')
    df_final = pd.merge(df_final, df_basic, left_on='Equipo', right_on='Equipo', how='left')
    df_opp = df_final.copy()
    df_tm = df_final.copy()
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
    df = df.drop(['Equipo_x', 'Equipo_y'], axis=1)
    # Nombre del archivo Excel y de la hoja
    archivo_excel = 'LaLiga Dataset 2023-2024.xlsx'

    df_existente = pd.read_excel(archivo_excel)

    df = pd.concat([df_existente, df], ignore_index=True)
    df.to_excel(archivo_excel, index=False)

    return df

# Definimos el segundo task que es preparar los datos para las predicciones
@task(name="Preparar Datos para Predicciones")
def preparar_datos_prediccion(jornada: int) -> pd.DataFrame:
    import pandas as pd
    import re
    # Seleccionamos el número de la jornada
    jornada = 12

    url = "https://fbref.com/es/comps/12/horario/Resultados-y-partidos-en-La-Liga"
    tables = pd.read_html(url)
    df = tables[0]
    # seleccionamos las variables
    df = df[['Sem.', 'Día', 'Fecha', 'Local', 'Visitante']]
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
    df = df[["Día", "Sedes", "Visitante", "Local"]]
    df = df.rename(columns={"Local": "Anfitrion", "Visitante": "Adversario"})
    # Duplicamos el dataframe e invertimos las columnas para hacer la concatenacion
    df_2 = df.copy()
    df_2 = df_2.rename(columns={"Adversario": "Anfitrion", "Anfitrion": "Adversario"})
    df_2["Sedes"] = 0
    df = pd.concat([df, df_2], ignore_index=True)
    df["Día"] = df["Día"].astype(int)
    ### Estadisticas básicas
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_basic = tables[0]
    df_basic = df_basic[
        ['RL', 'Equipo', 'PG', 'PE', 'PP', 'GF', 'GC', 'xG', 'xGA', 'Últimos 5', 'Máximo Goleador del Equipo']]
    df_basic['Máximo Goleador del Equipo'] = df_basic['Máximo Goleador del Equipo'].apply(
        lambda x: int(re.search(r'\b(\d+)\b', x).group(1)) if re.search(r'\b(\d+)\b', x) else None)
    df_basic['Últimos 5'] = df_basic['Últimos 5'].apply(lambda resultados: sum(
        [3 if resultado == 'PG' else (1 if resultado == 'PE' else 0) for resultado in resultados.split()]))
    ### Estadisticas de Ofensiva
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_ataque = tables[2]
    df_ataque = df_ataque.drop(["Tiempo Jugado", "Expectativa", 'Por 90 Minutos'], axis=1)
    df_ataque.columns = df_ataque.columns.droplevel(level=0)
    df_ataque = df_ataque[['Equipo', 'Edad', 'Pos.', 'Ass', 'TPint', 'PrgC', 'PrgP']]
    # Disparos
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_disparos = tables[8]
    df_disparos.columns = df_disparos.columns.droplevel(level=0)
    df_disparos = df_disparos[['Equipo', '% de TT', 'Dist']]
    df_ataque = pd.merge(df_ataque, df_disparos, left_on='Equipo', right_on='Equipo', how='left')
    # Pases
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_pases = tables[10]
    df_pases = df_pases.drop(["Cortos", "Medios", 'Largos', 'Expectativa'], axis=1)
    df_pases.columns = df_pases.columns.droplevel(level=0)
    df_pases = df_pases[['Equipo', '% Cmp', 'Dist. tot.']]
    df_ataque = pd.merge(df_ataque, df_pases, left_on='Equipo', right_on='Equipo', how='left')
    ### Estadisticas de defensa
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_porteria = tables[4]
    df_porteria = df_porteria.drop(["Tiempo Jugado", "Tiros penales"], axis=1)
    df_porteria.columns = df_porteria.columns.droplevel(level=0)
    df_porteria = df_porteria[['Equipo', 'GC', 'DaPC', 'Salvadas', 'PaC']]
    url = "https://fbref.com/es/comps/12/Estadisticas-de-La-Liga"
    tables = pd.read_html(url)
    df_defensa = tables[16]
    df_defensa = df_defensa.drop(['Desafíos'], axis=1)
    df_defensa.columns = df_defensa.columns.droplevel(level=0)
    df_defensa = df_defensa[['Equipo', 'TklG', 'Int', 'Err']]
    df_final = pd.merge(df_ataque, df_defensa, left_on='Equipo', right_on='Equipo', how='left')
    df_final = pd.merge(df_final, df_basic, left_on='Equipo', right_on='Equipo', how='left')
    df_opp = df_final.copy()
    df_tm = df_final.copy()
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
    df = df.drop(['Equipo_x', 'Equipo_y'], axis=1)
    df_prediccion = df

    return df_prediccion

# Definimos el task para cargar y preprocesar los datos
@task(name="Cargar y Procesar Dataset")
def cargar_procesar_dataset() -> tuple:
    df = pd.read_excel('LaLiga_Dataset_2023_2024.xlsx')

    X = df[['Día','Sedes','Edad(opp)','Pos.(opp)', 'Ass(opp)', 'TPint(opp)',
      'PrgC(opp)', 'PrgP(opp)','% de TT(opp)', 'Dist(opp)', '% Cmp(opp)', 'Dist. tot.(opp)','TklG(opp)', 'Int(opp)',
      'Err(opp)', 'RL(opp)', 'PG(opp)', 'PE(opp)','PP(opp)', 'GF(opp)', 'GC(opp)', 'xG(opp)', 'xGA(opp)','Últimos 5(opp)',
      'Máximo Goleador del Equipo(opp)', 'Edad(tm)', 'Pos.(tm)', 'Ass(tm)', 'TPint(tm)', 'PrgC(tm)', 'PrgP(tm)',
      '% de TT(tm)', 'Dist(tm)', '% Cmp(tm)', 'Dist. tot.(tm)', 'TklG(tm)','Int(tm)', 'Err(tm)', 'RL(tm)', 'PG(tm)',
      'PE(tm)', 'PP(tm)', 'GF(tm)','GC(tm)', 'xG(tm)', 'xGA(tm)', 'Últimos 5(tm)','Máximo Goleador del Equipo(tm)']]
    y = df['Resultado']

    # Dividimos en conjuntos de entrenamiento y prueba
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=15)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Guardamos el scaler
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/scaler.pkl", "wb") as f_out:
        pickle.dump(scaler, f_out)

    return X_train_scaled, X_val_scaled, y_train, y_val

# Creamos el task para entrenar los modelos
@task(name="Entrenar Modelo")
def entrenar_modelo(X_train, X_val, y_train, y_val, model_class, model_name, param_distributions, n_iter=10):
    with mlflow.start_run(run_name=model_name):
        # Realizamos la búsqueda de hiperparámetros
        search = RandomizedSearchCV(
            estimator=model_class(),
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='accuracy',
            n_jobs=-1,
            cv=3,
            random_state=42
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_val)

        # Calculamos métricas
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        # Registramos métricas y parámetros en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_params(search.best_params_)

        # Guardamos el modelo
        mlflow.sklearn.log_model(best_model, artifact_path="models")

        # Obtenemos el run_id
        run_id = mlflow.active_run().info.run_id

        return accuracy, run_id, model_name
# Creamos el task para registrar modelos en el model registry
@task(name="Registrar Modelo")
def registrar_modelo(run_id, model_name, alias):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/models"
    model_version = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, alias, model_version.version)
    return model_version.version

# Creamos el task para comparar los modelos y asignar los alías
@task(name="Comparar Modelos y Asignar Alias")
def comparar_modelos(resultados):
    # resultados es una lista de tuplas (accuracy, run_id, model_name)
    # Ordenamos por accuracy descendente
    resultados_ordenados = sorted(resultados, key=lambda x: x[0], reverse=True)

    # El mejor modelo es el "champion"
    champion_accuracy, champion_run_id, champion_model_name = resultados_ordenados[0]
    # El segundo mejor es el "challenger"
    challenger_accuracy, challenger_run_id, challenger_model_name = resultados_ordenados[1]

    # Registramos modelos y asignamos su alias
    champion_version = registrar_modelo(champion_run_id, "LaLiga_Model", "champion")
    challenger_version = registrar_modelo(challenger_run_id, "LaLiga_Model", "challenger")

    print(f"Champion model: {champion_model_name} (version {champion_version}), Accuracy: {champion_accuracy}")
    print(f"Challenger model: {challenger_model_name} (version {challenger_version}), Accuracy: {challenger_accuracy}")

# Definimos el flow principal
@flow(name="Pipeline de Entrenamiento y Registro de Modelos")
def pipeline_entrenamiento(jornada_actual: int):
    # Inicializamos MLflow y DagsHub
    dagshub.init(repo_owner='tu_usuario', repo_name='tu_repo', mlflow=True)
    mlflow.set_experiment("LaLiga_Experiment")

    # Actualizamos el dataset con los resultados de la jornada pasada
    actualizar_dataset(jornada_actual - 1)

    # Preparamos los datos para predicción (opcional)
    df_prediccion = preparar_datos_prediccion(jornada_actual)

    # Cargamos y procesamos el dataset
    X_train, X_val, y_train, y_val = cargar_procesar_dataset()

    # Definimos los modelos y sus distribuciones de hiperparámetros
    models = [
        (RandomForestClassifier, "RandomForest", {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }),
        (XGBClassifier, "XGBoost", {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 10],
            'n_estimators': [100, 200, 300],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0]
        }),
        (SVC, "SVC", {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'probability': [True]
        })
    ]

    resultados = []

    # Entrenamos los modelos
    for model_class, model_name, param_distributions in models:
        accuracy, run_id, model_name = entrenar_modelo(
            X_train, X_val, y_train, y_val,
            model_class, model_name, param_distributions, n_iter=10
        )
        resultados.append((accuracy, run_id, model_name))

    # Comparar modelos y asignar alias
    comparar_modelos(resultados)

if __name__ == "__main__":
    pipeline_entrenamiento(jornada_actual=12)


