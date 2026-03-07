# Cloud Alternatives for HW2 (Since Colab Doesn't Support Python 3.8)

Since Google Colab only offers Python 3.11/3.12 and `mujoco_py` requires Python 3.8-3.9, here are alternative cloud platforms:

## Option 1: Paperspace Gradient (Recommended - Easiest)

**Pros:** Free tier available, pre-configured ML environments, easy setup

**Steps:**
1. Go to https://www.paperspace.com/
2. Sign up (free tier available)
3. Create a Gradient Notebook
4. Choose a machine (free tier: CPU, or paid: GPU)
5. Select Python 3.8 environment
6. Upload your code or clone from GitHub
7. Run the setup commands from `setup.sh`

## Option 2: Lambda Labs (Good GPU Pricing)

**Pros:** Good GPU pricing, easy setup

**Steps:**
1. Go to https://lambdalabs.com/
2. Sign up
3. Launch an instance (choose Ubuntu 20.04 or 22.04)
4. SSH into the instance
5. Follow the `setup.sh` script from hw2/ac/

## Option 3: AWS EC2 (Most Flexible)

**Pros:** Full control, can use provided setup script

**Steps:**
1. Go to AWS Console → EC2
2. Launch instance:
   - AMI: Ubuntu 20.04 or 22.04 LTS
   - Instance type: t2.medium or better (t3.medium for GPU)
   - Storage: 20GB minimum
3. SSH into instance
4. Run setup:
   ```bash
   cd ~
   # Upload your hw2 folder or clone from GitHub
   cd hw2/ac
   bash setup.sh
   ```

## Option 4: Google Cloud Platform (GCP)

**Pros:** Similar to AWS, good integration with Colab

**Steps:**
1. Go to GCP Console
2. Create a Compute Engine VM
3. Choose Ubuntu 20.04
4. SSH in and follow setup.sh

## Quick Setup Script for Cloud Instances

Once you have SSH access to a Linux instance, run:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y unzip libglew-dev patchelf libegl1-mesa libgl1-mesa-glx libopengl0

# Install MuJoCo
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Install conda/miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc

# Create environment
cd ~/hw2/ac  # or wherever your code is
conda env create -f conda_env.yml
conda activate AC

# Install MetaWorld
pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@a98086ababc81560772e27e7f63fe5d120c4cc50
pip install "cython<3"

# Test
python -c "import mujoco_py; print('Success!')"
```

## Cost Estimates

- **Paperspace Free Tier:** $0 (CPU only, limited hours)
- **Paperspace Paid:** ~$0.50-1.00/hour (GPU)
- **Lambda Labs:** ~$0.50-1.50/hour (GPU)
- **AWS EC2 t2.medium:** ~$0.05/hour (CPU)
- **AWS EC2 t3.medium:** ~$0.04/hour (CPU)

For just testing code (not long training), a few hours should be sufficient and cost <$5.

## Recommendation

For your use case (testing code, not long training):
1. **Try Paperspace Gradient** - easiest setup, free tier available
2. **Or AWS EC2 t2.medium** - very cheap, use the provided setup.sh script

Both should work well for implementing and testing your Actor-Critic code.

