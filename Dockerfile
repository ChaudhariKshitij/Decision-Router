FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Optional Jev SDK:  docker build --build-arg WITH_JEV=1 -t decision-router .
ARG WITH_JEV=0
COPY requirements-jev.txt .
RUN if [ "$WITH_JEV" = "1" ]; then pip install --no-cache-dir -r requirements-jev.txt; fi
COPY . .
RUN chmod +x docker-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
