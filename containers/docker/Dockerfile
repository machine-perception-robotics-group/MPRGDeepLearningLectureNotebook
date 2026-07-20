FROM nvcr.io/nvidia/pytorch:26.03-py3 AS base
LABEL maintainer="Tsubasa Hirakawa <hirakawa@mprg.cs.chubu.ac.jp>"


# install sudo (for entrypoint script) and native libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# install computer vision python packages
# versions are pinned to match Google Colaboratory's environment (see ../pip_freeze_all.txt)
RUN pip install --no-cache-dir \
        opencv-python==5.0.0.93 \
        scikit-image==0.25.2 \
        scikit-learn==1.6.1 \
        pandas==2.2.2 \
        seaborn==0.13.2 \
        umap-learn==0.5.12


# release ports (jupyter, tensorboard)
EXPOSE 6006 8888


# user settings ===========================================
# add general user
ENV USER_NAME=dockeruser
ENV HOME=/home/${USER_NAME}

# add user to sudo group
WORKDIR /etc/sudoers.d
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}

# change permissions so that a general user can add user and group
# restore permission in entrypoint.sh
RUN chmod u+s /usr/sbin/useradd && chmod u+s /usr/sbin/groupadd

WORKDIR /
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
# =========================================================


# claude code stage ========================================
# base stage + Claude Code CLI (installed system-wide via apt, so it
# works regardless of which UID/GID entrypoint.sh creates at runtime)
FROM base AS claude

RUN install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://downloads.claude.ai/keys/claude-code.asc \
        -o /etc/apt/keyrings/claude-code.asc \
    && echo "deb [signed-by=/etc/apt/keyrings/claude-code.asc] https://downloads.claude.ai/claude-code/apt/stable stable main" \
        > /etc/apt/sources.list.d/claude-code.list \
    && apt-get update && apt-get install -y --no-install-recommends claude-code \
    && rm -rf /var/lib/apt/lists/*
# =========================================================
