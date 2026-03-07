# Google Colab Setup Guide for HW2

## Step 1: Access Google Colab

1. Go to https://colab.research.google.com/
2. Sign in with your Google account
3. Click "New Notebook" or "File" → "New Notebook"

## Step 2: Upload Your Code

### Option A: Upload from Local (Quick Start)
1. In Colab, click the folder icon on the left sidebar
2. Click the upload icon (📤)
3. Upload your `hw2/ac/` folder (you can zip it first, then upload and unzip)

### Option B: Use Google Drive (Recommended for larger files)
1. Upload `hw2/` folder to Google Drive
2. In Colab, mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Navigate to your folder:
   ```python
   import os
   os.chdir('/content/drive/MyDrive/path/to/hw2/ac')
   ```

### Option C: Clone from GitHub (If you push to GitHub)
```python
!git clone https://github.com/yourusername/CS224R_2026Winter.git
%cd CS224R_2026Winter/hw2/ac
```

## Step 3: Install Dependencies

Run this in a Colab cell:

```python
# Install system dependencies
!apt-get update
!apt-get install -y libglew-dev patchelf libegl1-mesa libgl1-mesa-glx libopengl0

# Install MuJoCo 2.1.0
!mkdir -p ~/.mujoco
!wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O ~/.mujoco/mujoco210.tar.gz
!tar -xzf ~/.mujoco/mujoco210.tar.gz -C ~/.mujoco/
!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Install Python packages
!pip install mujoco_py==2.1.2.14
!pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@a98086ababc81560772e27e7f63fe5d120c4cc50
!pip install hydra-core==1.1.0 dm_control imageio==2.9.0 imageio-ffmpeg==0.4.4
!pip install termcolor==1.1.0 pandas==1.3.0 matplotlib==3.4.2 opencv-python==4.5.3.56
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install "cython<3"

# Set environment variables
import os
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/root/.mujoco/mujoco210/bin'
os.environ['MUJOCO_GL'] = 'egl'
```

## Step 4: Test Installation

```python
# Test mujoco_py
import mujoco_py
print("mujoco_py installed successfully!")

# Test metaworld
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
print("MetaWorld installed successfully!")
print(f"Available environments: {list(ALL_V2_ENVIRONMENTS.keys())[:5]}...")
```

## Step 5: Run Your Code

```python
# Navigate to your code directory
import os
os.chdir('/content/your/path/to/hw2/ac')  # Adjust path as needed

# Run training
!python train.py
```

## Troubleshooting

### If mujoco_py installation fails:
```python
# Try installing with specific flags
!MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 pip install mujoco_py==2.1.2.14 --no-cache-dir
```

### If you get GL errors:
```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Try osmesa instead of egl
```

### If you need to restart:
- Runtime → Restart runtime (after installing packages)

## Tips

1. **Save your notebook**: File → Save (saves to Google Drive automatically)
2. **Use GPU**: Runtime → Change runtime type → GPU (free tier has limited GPU hours)
3. **Download files**: Right-click files in the file browser to download
4. **Session timeout**: Colab sessions timeout after ~90 min of inactivity

