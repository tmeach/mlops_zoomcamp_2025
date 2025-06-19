FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Обновление pip
RUN pip install -U pip

# Установка pipenv
RUN pip install pipenv

# Копирование файлов Pipfile и Pipfile.lock в контейнер
COPY ["Pipfile", "Pipfile.lock", "./"]

# Установка зависимостей проекта
RUN pipenv install --system --deploy

# Копирование основного скрипта в контейнер
COPY ["main_workflow.py", "./"]


# Настройка командной строки для запуска скрипта с аргументами
CMD ["python", "main_workflow.py", "--year", "2023", "--month", "04"]