import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator

class GeoSearch:
    def __init__(self, model_path, data_path):
        # Инициализация объекта GeoSearch с указанными путями к модели и данным
        self.model = SentenceTransformer(model_path)  # Инициализация модели для векторизации текста
        self.df_full = pd.read_csv(data_path)  # Загрузка данных из CSV-файла в DataFrame
        self.translator = Translator()  # Инициализация объекта для перевода текста

    def get_similar(self, question, translate=False, num=1, search=100, names_only=False):
        # Метод для поиска семантически близких элементов
        
        # Приводим вопрос к нижнему регистру
        question = question.lower()
        
        if translate:
            # Если флаг translate установлен, переводим вопрос на английский язык
            query = self.translator.translate(question).text
        else:
            query = question
        
        # Векторизуем вопрос и нормализуем его
        query_embedding = self.model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        ).reshape(1, -1)
        
        embeddings_path = 'embeddings.pqt'
        if os.path.isfile(embeddings_path):
            # Если файл с векторными представлениями существует, загружаем его
            embeddings = pd.read_parquet(embeddings_path).values
        else:
            # В противном случае, выбрасываем исключение
            raise Exception("Embeddings file not found. Please create embeddings first.")
        
        # Выполняем поиск семантически близких элементов в корпусе embeddings
        res = util.semantic_search(query_embedding, embeddings, top_k=search)
        idx = [i['corpus_id'] for i in res[0]]  # Получаем индексы наиболее похожих элементов
        score = [i['score'] for i in res[0]]  # Получаем значения схожести
        
        if names_only:
            # Если флаг names_only установлен, возвращаем только имена наиболее похожих элементов
            return (
                self.df_full.loc[idx]
                .drop_duplicates(subset=['name', 'code'])
                .iloc[:num]
                .name.tolist()
            )
        else:
            # В противном случае, возвращаем информацию о наиболее похожих элементах
            return (
                self.df_full.loc[idx, ['name', 'code', 'region', 'country']]
                .assign(similarity=score)  # Добавляем столбец с схожестью
                .drop_duplicates(subset=['name', 'code'])
                .iloc[:num]
            )
