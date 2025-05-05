# Usa imagem enxuta de Python
FROM python:3.10-slim

WORKDIR /app

# Copia dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY main.py .

# Porta do FastAPI/uvicorn
EXPOSE 8000

# Comando default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
