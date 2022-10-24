#/bin/bash
source parameters.sh

if [[ ! -d "python_venv" ]]; then
    echo "Creating Python Virtual Enviroment on $HOSTNAME"
    python3 -m venv python_venv
    source "python_venv/bin/activate"
    cd $SYSTEMDS_ROOT
    git pull >/dev/null 2>&1
    mvn clean package -P distribution >/dev/null 2>&1
    cd src/main/python
    pip install wheel >/dev/null 2>&1
    python create_python_dist.py >/dev/null 2>&1
    pip install . | grep "Successfully installed" &&
        echo "Installed Python Systemds Locally" || echo "Failed Installing Python Locally"
fi

## Install remotes
for index in ${!address[*]}; do
    if [ "${address[$index]}" != "localhost" ]; then
        echo "Installing for: ${address[$index]}"
        # Install SystemDS on system.
        ssh -T ${address[$index]} "
        mkdir -p github;
        cd github;
        if [[ ! -d 'systemds' ]]; then  git clone https://github.com/apache/systemds.git  > /dev/null 2>&1; fi;
        cd systemds;
        git reset --hard origin/master > /dev/null 2>&1;
        git pull > /dev/null 2>&1; 
        mvn clean package  -P distribution > /dev/null 2>&1;
        echo 'Installed Systemds on' \$HOSTNAME;
        cd \$HOME
        mkdir -p ${remoteDir}
        " &
    fi
done

wait
