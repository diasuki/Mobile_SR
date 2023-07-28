conda create -n sr_torch2.0 python=3.10 -y
conda activate sr_torch2.0
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib scikit-image scikit-learn -y
pip install -r requirements_pip.txt

# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y

conda create -n py38 python=3.8 -y
conda activate py38
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib scikit-image scikit-learn -y
pip install -r requirements_pip.txt
# Installing cupy (dependency for pwcnet)
conda install -y -c conda-forge cupy=7.8.0