#!/bin/bash

PASSWORD = "Docker!"

eval "$(ssh-agent)"

if [ -z "$first_flag" ]; then
  export first_flag=0  # Xに設定したい値をここに記述してください
  ssh-add
  chmod 600 /root/.ssh/*
  chmod 600 /root/.ssh/id_rsa
  echo "yes" "$PASSWORD"| ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -A web

else
  
  echo "PASSWORD" | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -A web

fi

