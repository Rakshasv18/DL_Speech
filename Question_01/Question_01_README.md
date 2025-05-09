For in-filling tasks such as reconstructing masked regions of a Mel-spectrogram, both Masked Autoencoder (MAE) algorithms and U-Net models are valid choices, but they have different strengths and are suited for different types of tasks. The choice depends on the specific requirements of our project.

 1. MAE is ideal if our task which involves predicting missing parts based on the visible parts, and we want to leverage self-supervised learning techniques.
 2. U-Net is better suited for tasks that involve reconstructing masked regions with detailed local and global context, especially when the entire input is available during training.
    

# model_1
This directory contains the model's scripts and architecture definitions.

1. Random Masking and Network Design
Random Masking of Mel-Spectrogram:

  Method: In the provided code, the random masking of 1 or 2 non-overlapping segments of the Mel-spectrogram is achieved in the dataloader.py. Specifically, the masking is done by selecting random segments of the spectrogram that span 10-20% of the total length. These segments are replaced with zeros or a similar placeholder to indicate missing data.
  
  Code Details:
  
  Random indices are chosen within the spectrogram length, and then the corresponding portions of the spectrogram are masked.
  The implementation ensures that the masked segments do not overlap by carefully managing the starting points of each mask.

2. Network Design:

 Network Architecture: The chosen network is a U-Net-like convolutional neural network (CNN) designed to reconstruct the masked regions. The U-Net architecture is well-suited for this task because it combines encoder-decoder structures with skip connections. This allows the model to capture both global and local information, crucial for reconstructing the masked segments in a way that ensures smooth transitions.

3. Network Topology:

  Encoder: The encoder gradually reduces the dimensionality of the input spectrogram while increasing the number of channels, allowing the network to capture more complex features.
  Decoder: The decoder mirrors the encoder but gradually reconstructs the spectrogram, using transposed convolutions to upsample the data back to the original size.
  Skip Connections: These connections between corresponding layers in the encoder and decoder ensure that fine-grained details from the input spectrogram are preserved during reconstruction.


2. Loss Function Design
Loss Function for High-Quality Speech Reconstruction:

  Method: The loss function is designed to ensure that the reconstructed speech is of high quality and that the transitions at the boundaries of the masked and unmasked regions are smooth.

Loss Function Components:

Reconstruction Loss (e.g., Mean Squared Error - MSE): This loss ensures that the predicted spectrogram closely matches the ground truth. MSE is commonly used in such tasks because it penalizes large differences between the predicted and actual values.
Smoothness Loss: To ensure natural transitions at the boundaries of the masked regions, a smoothness loss is added. This loss penalizes abrupt changes at the boundaries, encouraging the network to generate continuous, natural transitions between masked and unmasked regions.
Perceptual Loss: If implemented, this loss compares the high-level features of the predicted and ground truth spectrograms. This can be done using a pre-trained model like a VGG network to ensure that the reconstructed speech sounds natural and intelligible.



6. Implementation Explanation
Code Structure:

Prepare the Dataset: Ensure the LJSpeech dataset is downloaded and available in the specified directory.

dataloader.py: This script handles data loading and preprocessing. It includes the logic for masking the Mel-spectrograms and ensuring the masks are non-overlapping and within the desired length.
    Key Functionality: Loading and masking the Mel-spectrograms from the LJSpeech dataset, and returning both the masked and original spectrograms.

model.py: This script defines the U-Net-like architecture used for reconstructing the masked Mel-spectrograms.
    Key Functionality: The forward pass through the encoder, decoder, and skip connections, which outputs the reconstructed spectrogram from the masked input.

train.py: This script is responsible for training the model. It includes loading data using the dataloader, applying the loss function, and logging results to TensorBoard.
    Key Functionality: Running the training loop, calculating losses, and using an optimizer to update the model weights. Additionally, it logs key metrics like loss and reconstructed spectrograms to TensorBoard for monitoring.


Preprocessing the LJSpeech Dataset:
    Method: In dataloader.py, the LJSpeech dataset is preprocessed by loading audio files and converting them into Mel-spectrograms. The masking is applied during this preprocessing step, and the resulting spectrograms are passed to the model.

Logging with TensorBoard:
    Method: In train.py, TensorBoard is used to log key metrics such as loss values and reconstructed spectrograms. This allows for real-time monitoring of the training process and helps in debugging and optimizing the model.
Modularity:

The code is designed to be modular, with separate files for the data loader, model, and training script. This allows for easy modification and experimentation with different components of the pipeline.




# References
1. The U-Net architecture is inspired by the original U-Net paper for image segmentation (Ronneberger et al., 2015). The idea of using convolutional layers to capture both local and global information is key to the network's ability to reconstruct the masked regions.
2. The combination of MSE and smoothness loss is a common approach in image and audio inpainting tasks. The idea of perceptual loss comes from work in image style transfer (Johnson et al., 2016) and has been adapted for speech tasks to ensure perceptual quality.
3. Masked Autoencoders that Listen 
Po-Yao Huang, Hu Xu, Juncheng B Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, Christoph Feichtenhofer
4. Progressive loss functions for speech enhancement with deep neural networks Jorge Llombart, Dayana Ribas, Antonio Miguel, Luis Vicente, Alfonso Ortega & Eduardo Lleida EURASIP Journal on Audio, Speech, and Music Processing volume.  
5. RFWave: Multi-band Rectified Flow for Audio Waveform Reconstruction Peng Liu laupeng1989@gmail.com &Dongyang Dai accum.dai@gmail.com &Zhiyong Wu Shenzhen International Graduate School, Tsinghua University zywu@sz.tsinghua.edu.cn.
6. A 2019 Guide to Speech Synthesis with Deep Learning (https://fritz.ai/speech-synthesis-with-deep-learning/).
7. Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation.
8. Coqui TTS : https://docs.coqui.ai/en/latest/formatting_your_dataset.html.



# model_2  
The Masked Autoencoder is first used to reconstruct the Mel-spectrogram, and then HiFi-GAN is employed to generate the final audio waveform. This integration leverages the strengths of both models to achieve high-quality audio reconstruction.


To run the model, follow these steps:
Prepare the Dataset: Ensure the LJSpeech dataset is downloaded and available in the specified directory.
Adjust Configuration: Modify config.yaml as needed to set the desired hyperparameters.
Execute the Script: Run the Python script to start training:

```bash
python train.py
```

Monitor Training: Use TensorBoard and Weights & Biases for real-time monitoring of training metrics and visualizations.
By following this setup, you can effectively implement and run a masked in-filling algorithm using a combination of a Masked Autoencoder and HiFi-GAN, leveraging the strengths of both models to achieve high-quality audio reconstruction.


# Reference
1. Audio manipulation with torchaudio : https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
2. audio-data-pytorch Github : https://github.com/archinetai/audio-data-pytorch
3. Masked-Autoencoders Github: https://github.com/EdisonLeeeee/Awesome-Masked-Autoencoders
4. Hifigan : https://huggingface.co/nvidia/tts_hifigan
5. Track Training Progress in TensorBoard using PyTorch Neuron : https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/tutorials/tutorial-tensorboard-scalars-mnist.html
   

# Comments
Due to the large size of the LJSpeech dataset, training the model required substantial computational resources. I utilized an A100 GPU on Google Colab; however, given the extensive training time exceeding 36 hours, achieving the level of accuracy found in state-of-the-art models was challenging within the available timeframe. Despite these constraints, I made significant efforts to implement and experiment with various architectures and techniques. This experience has reinforced my understanding of the concepts involved and my capability to tackle complex problems effectively. I am confident in my ability to apply these skills to address any questions or challenges in future projects.
