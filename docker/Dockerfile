# Dockerfile (Entwicklungs-Modus, nur Bibliotheken – Code wird per Volume gemountet)

# 1) Basis-Image mit Python 3.11
FROM python:3.11-slim

# 2) Build-Args für UID/GID (damit Dateien im Container Ihrem Host-User gehören)
ARG UID
ARG GID

# 3) System-Tools & IPopt-Dependencies installieren, bash-completion aktivieren, Host-User anlegen
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        git \
        nano \
        bash \
        procps \
        bash-completion \
        pkg-config \
        coinor-libipopt-dev \
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        gfortran \
        build-essential \
        python3.11-dev && \
    # bash-completion global aktivieren
    echo "if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi" >> /etc/bash.bashrc && \
    # Host-User anlegen
    groupadd -g ${GID} hostuser && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash hostuser && \
    echo 'hostuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    # Aufräumen
    rm -rf /var/lib/apt/lists/*

# 4) Weiterarbeiten als nicht-root User
USER hostuser

# 5) Arbeitsverzeichnis im Container
WORKDIR /workspace

# 6) requirements.txt kopieren (nur Abhängigkeiten)
COPY --chown=hostuser:hostuser requirements.txt .

# 7) Python-Pakete installieren
RUN pip install --no-cache-dir -r requirements.txt

# 8) Jupyter-Konfiguration (Login-Shell + interaktive Terminals)
RUN bash -lc "jupyter lab --generate-config && \
    echo \"c.ServerApp.terminado_settings = {'shell_command': ['/bin/bash','-l','-i']}\" >> ~/.jupyter/jupyter_server_config.py && \
    echo \"c.TerminalManager.shell_command = ['/bin/bash','-l','-i']\" >> ~/.jupyter/jupyter_server_config.py"

# 9) Port für Jupyter
EXPOSE 8888

# 10) Standard-Entrypoint: interaktives JupyterLab (ohne Token)
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"]

