FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir python-dotenv langchain  openai install tiktoken langchain-community psycopg2-binary pgvector

# install postgresql client
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

EXPOSE 80

CMD ["python", "insert_data.py"]
