# Dockerfile (Entwicklungs-Modus, nur Bibliotheken, Code per Volume)

# 1) Basis-Image mit Python
FROM python:3.10-slim

# 2) Build-Args für UID/GID
ARG UID
ARG GID

# 3) System‑Tools installieren, bash-completion einrichten und Host‑User anlegen
RUN apt-get update && \
    apt-get install -y \
      sudo \
      git \
      nano \
      bash \
      procps \
      bash-completion && \
    # bash-completion global aktivieren
    echo "if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi" \
      >> /etc/bash.bashrc && \
    # Host‑User anlegen
    groupadd -g ${GID} hostuser && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash hostuser && \
    echo "hostuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

# 4) Weiter als nicht-root User
USER hostuser

# 5) Arbeitsverzeichnis im Container
WORKDIR /workspace

# 6) Abhängigkeiten kopieren (nur requirements)
COPY --chown=hostuser:hostuser requirements.txt .

# 7) Python-Pakete installieren
RUN pip install --no-cache-dir -r requirements.txt

# ← UPDATED: nun mit beiden Einstellungen und -i für interaktiv
RUN bash -lc "jupyter lab --generate-config && \
    echo \"c.ServerApp.terminado_settings = {'shell_command': ['/bin/bash','-l','-i']}\" \
      >> ~/.jupyter/jupyter_server_config.py && \
    echo \"c.TerminalManager.shell_command = ['/bin/bash','-l','-i']\" \
      >> ~/.jupyter/jupyter_server_config.py"    

# 8) Port für Jupyter o.Ä. dokumentieren
EXPOSE 8888

# 9) Standard-Shell (Login‑Shell, damit /etc/bash.bashrc geladen wird)
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"]
