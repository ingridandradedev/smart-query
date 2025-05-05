# filepath: c:\Users\Dell\Desktop\smart-query\rag-pipeline\Dockerfile
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /rag-pipeline

# Copia todos os arquivos do projeto para o container
COPY . /rag-pipeline

# Define a variável de ambiente PYTHONPATH
ENV PYTHONPATH=/rag-pipeline

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta padrão do FastAPI
EXPOSE 8000

# Comando para iniciar o servidor FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
