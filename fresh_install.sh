set -e

sudo apt update -y
sudo apt upgrade


# Install anaconda
sudo apt install python3-pip
pip3 install beautifulsoup4 requests tqdm -y
python3 anaconda_downloader.py # Downloads anaconda to $HOME

conda_file=$(ls ~/ | grep Anaconda)
bash "~/$conda_file -b" # Force install


# Install htop
sudo apt-get install htop    # Usage: htop

# Install GpuStat
pip3 install gpustat    # Usage: gpustat -ucFP --watch --color;

# Install git
sudo apt install git


# Create new conda environment
env_name="test_env"
conda create -n $env_name python=3.7 -y
conda activate $env_name
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

