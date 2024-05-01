pip3 install --upgrade pip
pip3 install \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    pandas \
    jupyterlab \
    ipywidgets \
    ipykernel \
    torch \
    torchvision \
    torchaudio \
    opencv-python \
    plyfile \
    lpips \
    pybind11 \
    viser
pip3 install -e ./submodules/diff-gaussian-rasterization/
pip3 install -e ./submodules/simple-knn/

USERNAME=$(whoami)
sudo chown -R $USERNAME:$USERNAME /home/vscode/.cache/torch