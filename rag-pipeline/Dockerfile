# filepath: c:\Users\Dell\Desktop\smart-query\rag-pipeline\Dockerfile
# 1) Imagem base com Python 3.10 slim
FROM python:3.10-slim

# 2) Defina o diretório de trabalho
WORKDIR /app

# 1) Instala dependências
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 2) Copia o package Python
COPY rag_pipeline/ ./rag_pipeline/

# (opcional) copia .env se precisar em staging
COPY .env* ./

# 6) Adicione o diretório ao PYTHONPATH
ENV PYTHONPATH=/app

# 7) Exponha a porta 8000 (onde o FastAPI roda)
EXPOSE 8000

# 8) Defina o comando que inicia a aplicação
CMD ["uvicorn", "rag_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
