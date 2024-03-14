conda create -n lever python=3.8 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers
pip install scipy
pip install pandas
pip install cython
pip install scikit-learn
pip install sentence-transformers
pip install seaborn
pip install numpy==1.21.0
pip install numba==0.56
pip install git+https://github.com/kunaldahiya/pyxclib.git