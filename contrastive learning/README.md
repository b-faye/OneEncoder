# OneEncoder: A LightWeight Framework For Cross-Modal Representation Learning

## Abstract
Cross-modal alignment learning combines information from text, images, audio, and video to create unified models for tasks like visual question answering and audiovisual analysis. Current methods rely on large, modality-specific encoders trained on vast aligned datasets, which is costly, difficult, and time-consuming. To address this, we propose OneEncoder, a lightweight framework that progressively aligns four modalities. Initially aligning image and text, OneEncoder then integrates other modalities without retraining the entire system. Efficient and cost-effective, it performs well on small paired datasets, surpassing methods dependent on large datasets and specialized encoders.


<center><b>Step 1: Train only the UP to align image and text modalities</b></center><br>

![Step 1](images/step_1.png)

<center><b>Step 2: Freeze the UP and train only the Alignment Layer to integrate new modalities (e.g., audio, video)</b></center><br>

![Step 2](images/step_2.png)

## Installation
### Requirements
* Linux, CUDA >= 12.1
* Python >= 3.9

    We recommend you to use Anaconda to create a conda environment:

    ```bash
    conda create -n OneEncoder python=3.9 pip
    ```

    Then, activate the environment:
    ```bash
    conda activate OneEncoder
    ```

* Pytorch >= 2.1.1

    For example, if your CUDA version is 12.1, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=2.1.1 torchvision=0.16.1 cudatoolkit=12.1 -c pytorch
    ```
* Other requirements
    ```bash
    pip install -r requirements
    ```
## Datasets

All datasets are located in the `datasets` directory.


To train the Universal Projection (UP) module using the contrastive learning method, the following datasets are utilized:

- [COCO Captions](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption)
- [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
- [TextCaps](https://huggingface.co/datasets/lmms-lab/TextCaps)

### Audio Modality Integration

To integrate the audio modality, the following dataset is used:

- [ibriSpeech Speech Recognition Alignment (SRA)](https://huggingface.co/datasets/nguyenvulebinh/asr-alignment)

### Video Modality Integration

To integrate the video modality, the following dataset is used:

- [Microsoft Research Video to Text (MSR-VTT)](https://huggingface.co/datasets/AlexZigma/msr-vtt)

### X-ray Image Modality Integration

To integrate the X-ray image modality, the following dataset is used:

- [ROCO Dataset](https://www.kaggle.com/datasets/virajbagal/roco-dataset)

# Training and Validation
* Train the UP to align image and text
    ```bash
    cd "image and text"
     python3 text_image.py
    ```

* Integrate audio modality
    ```bash
    cd "text and audio"
     python3 text_audio.py
    ```

* Integrate video modality
    ```bash
    cd "text and video"
     python3 text_video.py
    ```

* Integrate x-ray image modality
    ```bash
    cd "text and x-ray image"
     python3 text_radio.py
    ```
    
# Demo

<center><b>Querying Results</b></center><br>

![Step 2](images/querying.png)
