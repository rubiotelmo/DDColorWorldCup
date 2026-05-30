ARG DEBIAN_VERSION=trixie-20260518-slim

FROM debian:${DEBIAN_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base tools and native libraries used by the Python/OpenCV stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    less \
    locales \
    openssh-client \
    procps \
    sudo \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Locale
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Non-root user for VS Code
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Miniconda3 py313_26.3.2-2 was the latest Linux x86_64 installer on 2026-05-28.
ARG MINICONDA_VERSION=py313_26.3.2-2
ARG MINICONDA_INSTALLER=Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh
ARG MINICONDA_SHA256=2284bafb7863a23411b19874d216e237964d4b32dd9beb6807fa8b2d84570961
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}

ENV CONDA_DIR=/opt/miniconda
ENV CONDA_ENVS_PATH=/home/${USERNAME}/.conda/envs
ENV CONDA_PKGS_DIRS=/home/${USERNAME}/.conda/pkgs
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN mkdir -p "${CONDA_ENVS_PATH}" "${CONDA_PKGS_DIRS}" \
    && curl -fsSL "${MINICONDA_URL}" -o /tmp/miniconda.sh \
    && printf '%s  %s\n' "${MINICONDA_SHA256}" /tmp/miniconda.sh > /tmp/miniconda.sha256 \
    && sha256sum -c /tmp/miniconda.sha256 \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh /tmp/miniconda.sha256 \
    && "${CONDA_DIR}/bin/conda" config --system --set auto_activate_base false \
    && "${CONDA_DIR}/bin/conda" config --system --add envs_dirs "${CONDA_ENVS_PATH}" \
    && "${CONDA_DIR}/bin/conda" config --system --add pkgs_dirs "${CONDA_PKGS_DIRS}" \
    && "${CONDA_DIR}/bin/conda" clean -afy \
    && chown -R "${USERNAME}:${USERNAME}" "${CONDA_DIR}" /home/"${USERNAME}"/.conda

RUN printf '\n# Miniconda\nsource /opt/miniconda/etc/profile.d/conda.sh\nconda activate ddcolor 2>/dev/null || true\n' >> /home/"${USERNAME}"/.bashrc \
    && chown "${USERNAME}:${USERNAME}" /home/"${USERNAME}"/.bashrc

USER $USERNAME
WORKDIR /workspaces
