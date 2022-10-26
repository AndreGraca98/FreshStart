
sudo apt update -y

# Install anaconda
sudo apt install python3-pip -y
pip3 install beautifulsoup4 requests tqdm
python3 anaconda_downloader.py -y # Downloads anaconda to $HOME

conda_file=$(ls $HOME | grep Anaconda)
bash "$HOME/$conda_file" -b -u # Force install
rm "$HOME/$conda_file"

# Install htop
sudo apt-get install htop    # Usage: htop

# Install GpuStat
pip3 install gpustat    # Usage: gpustat -cuFP --watch --color;

# Install git
sudo apt install git


# Create new conda environment
env_name="my_env"
conda create -n $env_name python=3.7 tqdm -y
conda activate $env_name
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y



