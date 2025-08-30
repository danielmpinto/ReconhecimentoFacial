#!/bin/bash
    PASSWORD="inspercomp"
    echo "$PASSWORD" |  sudo -S apt update -y
    echo "$PASSWORD" |  sudo -S apt install -y git python3-pip 
    pip install -r requirements.txt
    