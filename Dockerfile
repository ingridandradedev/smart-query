# Dockerfile

# 1) Imagem base com Python 3.10 slim
FROM python:3.10-slim

# 2) Defina o diretório de trabalho
WORKDIR /app

# 3) Copie o pyproject.toml e (se houver) arquivos de lock/dependências
COPY pyproject.toml ./

# 4) Atualize pip e instale dependências (aqui faremos em 2 etapas: "pip install -r" caso precise,
#    ou se seu pyproject permite "pip install .", faremos após a cópia total)
RUN pip install --upgrade pip

# 5) Copie todo o conteúdo do diretório local para /app
COPY . .

# 6) Instale seu pacote (via pyproject). Se estiver configurado para "pip install .", vai funcionar.
#    Se você tiver um "setup.py" ou outra config, ajuste como precisar.
RUN pip install .

# 7) Exponha a porta 8000 (onde seu FastAPI roda)
EXPOSE 8000

# 8) Defina o comando que inicia sua aplicação
#    - Note que seu "main.py" está em "src/react_agent/main.py"
#    - Mas como "react_agent" é um pacote Python, podemos usar "react_agent.main:app"
CMD ["uvicorn", "react_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
