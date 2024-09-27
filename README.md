# Deep Neural Networks for YouTube Recommendations

## Overview

This repository provides a comprehensive implementation of a deep neural network-based recommendation system similar to YouTube's. The repo is organized to include the core Python implementation of the model and a Spark-based Scala solution for data generation and model serving.

### Repository Structure

1. **`py` Directory**: Contains the main Python code for model training and evaluation using TensorFlow.
   - **`deep_neural_networks_for_youtube_recommendations/`**: The central directory for Python-based code, divided into several sub-modules:
     - **`dnn/`**: Houses the core deep learning model (`dnn.py`) and supporting scripts for data preparation, training, and logging.
       - **Model Checkpoints (`ckpt/`)**: Stores TensorFlow checkpoints and metadata for model states at various training stages.
       - **Scripts for Data Handling & Training**:
         - **`data2tfrecords.py`**: Converts raw input data into TFRecords format for training.
         - **`read_tfrecords.py`**: Reads TFRecords for both training and evaluation.
         - **`tensor_board.py`**: Integrates TensorBoard for visual monitoring of model training metrics.
       - **Model Loading & Exporting**:
         - **`load_dnn_model.py`**: Facilitates loading of trained models for prediction tasks.
         - **`modelpath/`**: Stores the saved TensorFlow model.
       - **Training Data (`tfrecords/`)**: Includes separate TFRecord files for training and evaluation.
     - **`example/`**: Provides usage examples with Python scripts to demonstrate how to train and test the model (`example1.py`, `example2.py`).
     - **`reference/`**: Offers reference implementations for custom network layers and feature engineering in TensorFlow.
     - **`tfrecords_methods/`**: Contains additional scripts for handling TFRecords, including data conversion and reading methods.

2. **`spark` Directory**: Provides the Scala-based code for using Spark to generate data and make predictions using the TensorFlow model.
   - **`explore/`**: Contains data generation and model inference scripts:
     - **Data Generation**:
       - **`MakeDataOne.scala`** & **`MakeDataTwo.scala`**: Generate synthetic datasets and store them as TFRecords in HDFS.
     - **Model Prediction**:
       - **`PredictUserVectorMakeDataOne.scala`**, **`PredictUserVectorMakeDataTwo.scala`**: Load and use the TensorFlow model to predict user vectors.
       - **`ItemEmbeddingMakeDataOne.scala`**, **`ItemEmbeddingMakeDataTwo.scala`**: Predict item embeddings using the same model.
   - **`sparkapplication/`**: Base Spark classes for local and online model execution.
   - **`vars/`**: Contains configuration files (`vars.prod.properties`, `vars.sit.properties`) for different environments.
   - **`example/`**: Demonstrates basic Spark operations using example Scala scripts.

3. **`paper` Directory**: Contains the original research paper "Deep Neural Networks for YouTube Recommendations," which provides background and theoretical context for the model implemented in this repository.

---

## How to Use

- **Python Model Training & Evaluation**: The core TensorFlow-based model can be trained and evaluated using the provided Python scripts. Data can be prepared in TFRecords format, and TensorBoard is used for visualization.
- **Spark-Based Data Generation & Prediction**: The Spark/Scala code allows you to generate data at scale, load TensorFlow models, and run predictions efficiently, leveraging distributed processing.

## Conclusion

This repository provides a complete toolkit for developing and deploying a deep neural network-based recommendation system, with capabilities for both Python-based training and Spark-based data handling. The implementation offers flexibility and scalability for recommendation tasks in YouTube-style environments. The supporting paper gives further insight into the model's background and theory.
