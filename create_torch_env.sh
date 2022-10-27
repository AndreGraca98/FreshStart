
# Create new conda environment
env_name="my_env"
conda create -n $env_name python=3.7 -y
conda activate $env_name
conda install tqdm -y
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y


python -c "import torch
print('\nCan use cuda:', torch.cuda.is_available())
print('Devices available:', [f'cuda:{i}' for i in range(torch.cuda.device_count())])
"
