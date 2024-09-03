# FastPointNet

## Basic Setup (no Hardware)
**Setup Notes:**

- Training these models is computationally expensive. We used 64GB of RAM and an NVIDIA Tesla V100 with 32GB VRAM


**Setup Steps:**

0. Unfortunately, the data we use is only for the use of KamLAND-Zen Collaborators. Please contact the authors of this repository for acquiring the data
1. Setup a new conda environment and configure it using the included `./config.yaml` file
2. To run the Tensorflow-only version of the training, you may use the `train.py` script. The details of training can be found inside the script
3. To run the quantized version of the training, you may use `train_keras.py`
4. Depending on the script, you may also need to run the `visualize_pickle.py` script, which loads the pickle file generated from training and paints the histograms. This is so you can modify the graphs without having to retrain everytime. Currently, there is a bug that prevents us from serializing the models
5. To search for optimial quantization parameters, use the `qoptimize.sh` script

## Full Setup (with Hardware)
**Setup Steps:**

0. Complete all the steps in basic setup
1. Install AMD Vivado and AMD Vitis. It is preferable to host these applications on a Linux machine
2. Install deepsocflow, which is currently available on Github [here](https://github.com/KastnerRG/cgra4ml)
3. To train the cgra-ported model, use the `./cgra/train_cgra.py` script. The parameters for training may be configured at the top of the script
4. To export the model to Vivado, use `python -m pytest -s ./cgra/train_cgra.py`
5. Follow the directions in [this video](https://www.youtube.com/watch?v=cWdDxEbrMuA) to deploy to the RF-SoC4x2. Note that you should use the execution script found [here](https://github.com/KastnerRG/cgra4ml/blob/e811855f3249eba0f3637a51cb3244142c4e4733/deepsocflow/c/xilinx_example.c) in Vitis
