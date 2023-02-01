# get docker images
FROM nikolaik/python-nodejs:python3.8-nodejs16-slim

# copy project
COPY . /final

# set work directory
WORKDIR /final/backend

# set environments
ENV PYTHONPATH=/final
ENV PYTHONBUFFERED=1

# install pip
RUN pip install pip==21.2.4
RUN pip install --upgrade pip

# install poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# expose port for FastAPI
EXPOSE 8000

# settings - poetry install, npm install, npm run build
RUN chmod +x settings.sh
RUN ./settings.sh

# run FastAPI backend server
CMD ["poetry", "run", "python", "__main__.py"]