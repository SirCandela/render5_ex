
#importaciones 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

# Carga de los datasets limpios
df_developer = pd.read_csv('df_developer.csv')
df_userdata = pd.read_csv('df_userdata.csv')
df_user_genre = pd.read_csv('df_user_genre.csv')
df_best_developer_year = pd.read_csv('best_developer_year.csv')
df_recomendacion_juego = pd.read_csv('df_recomendacion_juego.csv')
df_developer_reviews_analysis = pd.read_csv('developer_reviews_analysis.csv')

# Inicializamos la app FastApi
app = FastAPI()

# Rutas y Funciones de la API
@app.get('/')
def read_root():
    return {"message": "PROYECTO INDIVIDUAL Nº1 --> agregue '/docs' a la url para continuar"}


# Retorna la cantidad de películas estrenadas en un idioma específico.
@app.get('/developer')
def developer(Desarrollador: str):
    try:
        # Reemplazar los valores modificados en el ETL
        desarrollador = desarrollador.replace(',', '.')
        desarrollador = desarrollador.replace('"', '')

        # Filtra el DataFrame por el desarrollador buscado
        data_filtrada = df_developer[df_developer['developer'].str.contains(desarrollador, case=False, na=False)]
        
        if data_filtrada.empty:
            raise ValueError("No se encontraron datos para el desarrollador especificado.")

        # Convertir los precios a formato numérico
        data_filtrada['price'] = data_filtrada['price'].str.replace(',', '.').astype(float)

        # Cuenta los items por año
        cantidadPorAño = data_filtrada.groupby('year')['developer'].count().reset_index()

        # Cuenta los elementos Free por año
        cantidadFreeAño = data_filtrada[data_filtrada['price'] == 0].groupby('year')['developer'].count().reset_index()

        # Combina los DataFrames y renombra las columnas
        merged_df = pd.merge(cantidadPorAño, cantidadFreeAño, on='year', how='left', suffixes=('_total', '_free'))
        merged_df = merged_df.fillna(0)

        # Calcula el porcentaje de elementos gratis por año
        merged_df['porcentaje_gratis_por_año'] = (merged_df['developer_free'] / merged_df['developer_total'] * 100).astype(float).round(2)

        # Prepara la salida en formato de tabla
        result_table = merged_df[['year', 'developer_total', 'porcentaje_gratis_por_año']]
        result_table.columns = ['Año', 'Cantidad de Items', 'Contenido Free']
        result_table['Contenido Free'] = result_table['Contenido Free'].astype(str) + '%'

        return result_table
    
    except FileNotFoundError:
        print("Archivo CSV no encontrado.")
        return None
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return None
    except Exception as e:
        print(f"Ocurrió un error: {str(e)}")
        return None

# Retorna la duración y el año de una película dada
@app.get('/userdata')
def userdata(user_id): 
    
    # Filtra por el usuario de interés
    usuario = df_userdata[df_userdata['user_id'] == user_id]

    if usuario.empty:
        return {"Error": "Usuario no encontrado"}
        
    # Selecciona las columnas de interés y convierte el resultado a un diccionario
    return {
        "user_id": usuario['user_id'].values[0],
        "Dinero gastado": usuario['Dinero gastado'].values[0],
        "Cantidad de items": usuario['Cantidad de items'].values[0],
        "% de recomendación": usuario['% de recomendación'].values[0]
    }


# Retorna la cantidad y de películas y la ganancia total de una franquicia dada
@app.get('/UserForGenre')
def UserForGenre(genero: str):
    
    # Filtrar datos para el género especificado
    df_genero = df_user_genre[df_user_genre['genres'].str.contains(genero)]

    if df_genero.empty:
        return {"Error": f"No se encontraron datos para el género '{genero}'"}

    # Calcular horas totales jugadas por usuario para el género dado
    horas_por_usuario = df_genero.groupby('user_id')['Horas jugadas'].sum()

    # Identificar el usuario con la mayor cantidad de horas jugadas
    usuario_max_horas = horas_por_usuario.idxmax()
    max_horas_jugadas = horas_por_usuario.max()

    # Calcular la acumulación de horas jugadas por año de lanzamiento para el usuario con más horas jugadas
    horas_por_año = df_genero[df_genero['user_id'] == usuario_max_horas].groupby('year')['Horas jugadas'].sum()

    # Construir el diccionario de resultados
    resultados = {
        "Usuario con más horas jugadas para " + genero: usuario_max_horas,
        "Horas jugadas": [{"Año": año, "Horas": horas} for año, horas in horas_por_año.items()]
    }

    return resultados

# Retorna la cantidad de películas producidas en un país dado
@app.get('/best_developer_year')
def best_developer_year(año: int):

    # Filtrar datos para el año especificado
    df_año = df_best_developer_year[df_best_developer_year['year'] == año]

    if df_año.empty:
        return {"Error": f"No se encontraron datos para el año {año}"}

    # Seleccionar los top 3 desarrolladores con más recomendaciones para el año dado
    top_3_desarrolladores = df_año.nlargest(3, 'Recomendaciones')['developer'].tolist()

    # Construir la lista de resultados
    resultados = [{"Puesto {}: {}".format(i+1, desarrollador)} for i, desarrollador in enumerate(top_3_desarrolladores)]

    return resultados

# Retorna la ganancia y cantidad de películas de una productora dada
@app.get('/developer_reviews_analysis')
def developer_reviews_analysis(desarrolladora: str):
    
    # Filtrar datos para el desarrollador especificado
    df_desarrolladora = df_developer_reviews_analysis[df_developer_reviews_analysis['developer'] == desarrolladora]

    if df_desarrolladora.empty:
        return {"Error": f"No se encontraron datos para el desarrollador '{desarrolladora}'"}

    # Calcular el recuento de registros de reseñas categorizados con análisis de sentimiento positivo y negativo
    recuento_sentimiento = df_desarrolladora.groupby('sentiment')['Recuento de sentiment_analysis'].sum()

    # Construir el diccionario de resultados
    resultados = {desarrolladora: recuento_sentimiento.to_dict()}

    return resultados
# Retorna el éxito promedio y una lista con las películas de un director dado

@app.get('/recomendacion_juego')
def recomendacion_juego(id_producto: str):
    
    # Verificar si hay al menos una fila que coincida con el id_producto
    if not df_recomendacion_juego[df_recomendacion_juego['id'].astype(str) == id_producto].empty:
        # Obtener el título del juego de entrada
        titulo_juego = df_recomendacion_juego[df_recomendacion_juego['id'].astype(str) == id_producto]['title'].iloc[0]

        # Inicializar el vectorizador TF-IDF
        tfidf_vectorizer = TfidfVectorizer()

        # Crear la matriz TF-IDF para todos los títulos de juegos
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_recomendacion_juego['title'])

        # Calcular la similitud del coseno entre el título del juego de entrada y todos los demás títulos
        cosine_similarities = cosine_similarity(tfidf_matrix[df_recomendacion_juego.index[df_recomendacion_juego['id'].astype(str) == id_producto]], tfidf_matrix).flatten()

        # Obtener los índices de los juegos más similares (excluyendo el juego de entrada)
        indices_juegos_similares = cosine_similarities.argsort()[-6:-1][::-1]

        # Obtener la lista de juegos recomendados
        juegos_recomendados = df_recomendacion_juego.iloc[indices_juegos_similares]

        # Crear el mensaje de salida
        mensaje = f"Si te gusta '{titulo_juego}' te recomendamos los siguientes juegos:\n"
        for i, (_, titulo) in enumerate(juegos_recomendados[['id', 'title']].iterrows(), start=1):
            mensaje += f"{i}. id: {titulo['id']}\n   title: {titulo['title']}\n"

        return mensaje
    else:
        return "No se encontró ningún juego con ese id"