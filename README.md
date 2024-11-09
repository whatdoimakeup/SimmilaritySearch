# Docker Compose для сервиса поиска похожих по смыслу изображений 

Этот репозиторий содержит файл docker-compose.yml, который настраивает среду разработки с следующими сервисами:

- Weaviate: Открытый векторный поисковый движок.
- Multi2Vec-Clip: Модель для многомодальной векторизации.
- PostgreSQL: Реляционная база данных для хранения данных приложения.
- MinIO: Объектное хранилище, совместимое с Amazon S3.
- Django: Веб-фреймворк для создания веб-приложений.

## Предварительные требования

- Установленные Docker и Docker Compose на вашем компьютере.
- NVIDIA GPU и NVIDIA Container Toolkit (если вы используете GPU для Multi2Vec-Clip).

## Начало работы

1. Склонируйте этот репозиторий:

   
   `git clone <url-репозитория>
   cd <директория-репозитория>`
   

2. Соберите и запустите сервисы:

   
   `docker-compose up --build`
   

3. Доступ к сервисам:
   - Weaviate: http://localhost:8080
   - Консоль MinIO: http://localhost:9001
   - Django: http://localhost:8000

## Обзор сервисов

### Weaviate

- Образ: cr.weaviate.io/semitechnologies/weaviate:1.27.0
- Порты: 
  - 8080 (HTTP API)
  - 50051 (gRPC)
- Тома: Данные сохраняются в ./weaviate_data.
- Переменные окружения:
  - CLIP_INFERENCE_API: URL для сервиса Multi2Vec-Clip.
  - QUERY_DEFAULTS_LIMIT: Значение по умолчанию для лимита запросов.
  - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: Позволяет анонимный доступ.
  - DEFAULT_VECTORIZER_MODULE: Указывает, какой векторизатор использовать.

### Multi2Vec-Clip

- Образ: cr.weaviate.io/semitechnologies/multi2vec-clip:xlm-roberta-base-ViT-B-32-laion5b_s13b_b90k
- Переменные окружения:
  - ENABLE_CUDA: Включает использование CUDA для GPU.
  
### PostgreSQL

- Образ: postgres:14
- Порты: 
  - 5433 (сопоставлен с портом по умолчанию PostgreSQL 5432)
- Тома: 
  - Данные сохраняются в ./postgres_data.
  - Инициализационные SQL-скрипты могут быть добавлены в ./config/init.sql.
- Переменные окружения:
  - POSTGRES_PASSWORD: Пароль для пользователя PostgreSQL.

### MinIO

- Образ: minio/minio
- Порты: 
  - 9000 (API)
  - 9001 (Консоль)
- Тома: Данные сохраняются в ./minio_data.

### Django

- Контекст сборки: ./backend
- Порты: 
  - 8000 (Сервер разработки Django)
- Зависимости: Ожидает, пока PostgreSQL и Weaviate станут доступными, прежде чем запуститься.
- Переменные окружения:
  - PYTHONUNBUFFERED: Обеспечивает вывод логов в реальном времени.

## Проверка состояния

Каждый сервис включает проверки состояния, чтобы убедиться, что они работают корректно, прежде чем запустятся зависимые сервисы.