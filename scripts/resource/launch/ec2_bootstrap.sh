#!/bin/bash

# setup the environment
sudo yum update
sudo yum install -y git
sudo yum install -y java-11-openjdk-devel
sudo yum install -y maven
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
echo "SystemDS installed and setup (single-node) globally"
# install ssm and cloudwatch agents by default (to allow collecting metrics)
sudo yum install amazon-ssm-agent -y
sudo yum install amazon-cloudwatch-agent -y
echo "Cloudwatch agent installed successfully"
# allow checking for installation finished
touch /tmp/systemds_installation_completed


