FROM astral/uv:python3.13-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git bash

COPY . /app

CMD ["tail", "-f", "/dev/null"]