FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY streamlit_app.py ./streamlit_app.py

RUN mkdir -p /workspace/models /workspace/output

EXPOSE 8501

CMD ["python", "-m", "app.main", "--stage", "all"]
