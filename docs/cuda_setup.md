

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin

sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb

sudo cp /var/cuda-repo-ubuntu2204-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update


sudo apt-get -y install cuda-toolkit-13-0



sudo apt-get install -y cuda-drivers

--

sudo apt remove --purge nvidia-*
sudo apt update
sudo apt install linux-headers-$(uname -r) dkms
sudo apt install --reinstall nvidia-driver-580
