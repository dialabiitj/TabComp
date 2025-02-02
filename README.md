# VisToT

**Code for TabComp: A Dataset for Visual Table Reading Comprehension.**
Download Dataset: [Google Drive Link](https://drive.google.com/drive/folders/1qw1pa-9ggAlN05ko4MfnKJ4_BZZXo5tz)

## Requirements
- Use **python >= 3.8.0**. Conda recommended: [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)
- Use Pytorch >=1.7.0 for CUDA 11.0 : [https://pytorch.org/get-started/previous-versions/#linux-and-windows-20](https://pytorch.org/get-started/previous-versions/#linux-and-windows-20)
- Other requirements are listed in `requirements.txt`

**Setup the environment**
```bash
# create a new conda environment
conda create -n vt3 python=3.8.13

# activate environment
conda activate vt3

# install pytorch
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# install other dependencies
pip install -r requirements.txt
```
