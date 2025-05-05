# filepath: c:\Users\Dell\Desktop\smart-query\rag-pipeline\Dockerfile
# 1) Imagem base com Python 3.10 slim
FROM python:3.10-slim

# 2) Defina o diretório de trabalho
WORKDIR /app

# 3) Copie o arquivo de dependências
COPY requirements.txt ./

# 4) Atualize pip e instale as dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copie todo o conteúdo do projeto para o container
COPY . .

# 6) Adicione o diretório ao PYTHONPATH
ENV PYTHONPATH=/app

# 7) Exponha a porta 8000 (onde o FastAPI roda)
EXPOSE 8000

# 8) Defina o comando que inicia a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
