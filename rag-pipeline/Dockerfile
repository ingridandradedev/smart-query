# filepath: c:\Users\Dell\Desktop\smart-query\rag-pipeline\Dockerfile
# 1) Imagem base com Python 3.10 slim
FROM python:3.10-slim

# 2) Defina o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema (inclui libGL, libgthread, poppler e tesseract)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
      poppler-utils \
      tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

# 1) Instala dependências Python
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install unstructured-pytesseract

# define o OCR_AGENT para usar o pytesseract
ENV OCR_AGENT=unstructured_pytesseract

# 2) Copia o código da API
COPY rag_pipeline/ ./rag_pipeline/

# (opcional) copia .env se precisar em staging
COPY .env* ./

# 3) Ajusta PYTHONPATH e expõe porta
ENV PYTHONPATH=/app
EXPOSE 8000

# 4) Comando de inicialização
CMD ["uvicorn", "rag_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
