#!/bin/bash -e

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# create group
if [ x"$GROUP_ID" != x"0" ]; then
    groupadd -g $GROUP_ID $USER_NAME
fi

# create user
if [ x"$USER_ID" != x"0" ]; then
    useradd -d /home/$USER_NAME -m -s /bin/bash -u $USER_ID -g $GROUP_ID $USER_NAME
fi

# /home/$USER_NAME may already exist (e.g. as a mount point created for a
# volume like /home/$USER_NAME/.claude), in which case useradd -m does not
# take ownership of it. Force ownership so the user can write under $HOME.
sudo chown $USER_ID:$GROUP_ID /home/$USER_NAME

# restore permissions
sudo chmod u-s /usr/sbin/useradd
sudo chmod u-s /usr/sbin/groupadd

# jupyter environment
echo ""
echo "Jupyter Settings =========================================="
mkdir -p ${HOME}/.jupyter
jupyter notebook --generate-config
{ \
    echo "c.NotebookApp.ip = '*'"; \
    echo "c.NotebookApp.open_browser = False"; \
    echo "c.NotebookApp.port = 8888"; \
} | tee -a ${HOME}/.jupyter/jupyter_notebook_config.py
echo " Jupyter Settings; done."
echo "==========================================================="
echo "To start jupyter server, please execute the following command:"
echo "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root > /dev/null 2>&1 &"
echo ""

exec "$@"
