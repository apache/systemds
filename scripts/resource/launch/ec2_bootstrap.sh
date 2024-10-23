#!/bin/bash

# setup the environment
sudo apt update
sudo apt install -y git
sudo apt install -y openjdk-11-jdk-headless
sudo apt install -y maven 
echo "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))" | sudo tee -a /etc/profile
source /etc/profile

echo "JDK 11 and MAVEN ware installed successfully"

# install SystemDS
# TODO: point to main repo install the end: git clone https://github.com/apache/systemds.git
git clone https://github.com/lachezar-n/systemds.git
cd systemds
# TODO: point to main repo install the end
git checkout resource-optimizer
sudo mvn install -Dmaven.test.skip=true
# export executable
echo "export SYSTEMDS_ROOT=$(pwd)" | sudo tee -a /etc/profile
echo "export PATH=$(pwd)/bin:$PATH" | sudo tee -a /etc/profile
source /etc/profile
# allow checking for installation finished
touch /tmp/systemds_installation_completed
echo "SystemDS installed and setup (singlenode) globally"

sudo snap install aws-cli --channel=v2/stable --classic
echo "AWS CLI installed successfully"
