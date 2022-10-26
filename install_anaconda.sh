
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
sudo apt install gpustat    # Usage: gpustat -cuFP --watch --color;

