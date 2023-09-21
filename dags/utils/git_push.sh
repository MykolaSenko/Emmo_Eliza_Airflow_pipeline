#!/bin/bash

# Set the path to the SSH key
AIRFLOW_SSH_KEY=~/.ssh/id_rsa

# Export the SSH key
export SSH_AUTH_SOCK=$(mktemp -u /tmp/ssh_auth_sock.XXXXXXXX)
export SSH_AGENT_PID=$(/usr/bin/ssh-agent -a $SSH_AUTH_SOCK)
/usr/bin/ssh-add "$AIRFLOW_SSH_KEY"


#cd /home/flyingpig/codes/becode_projects/Emmo_Eliza_Airflow_pipeline
# Push the changes to the remote Git repository
git push