# Custom NER (Named Entity Recognition) Model with RNN and BiLSTM Architecture

## Problem Statement

### Introduction
<p> NER (Named Entity Recognition) is an important application of NLP (Natural Language Processing), which is used for the identification of different entites like people, geographic locations, dates, etc. 
It is becoming popular in the medical community because of it's vast capabilities in identifying and extracting disease-like conditions, various symptoms, and pharmaceutical information in various information 
sources like clinical records, scientific literature, and so on. </p>
<p> NER models can be configured in various ways: </p>
<ul>
  <li><b>Using Machine Learning algorithms:</b> This includes using algorithms like "Decision Trees" and so on.</li>
  <li><b>Using Deep Learning Methods: </b> This includes using RNNs (Recurrent Neural Networks) and transformers for handling various dependencies within the training data.</li>
  <li><b>A mixture of both of these methods:</b> In order to meet the requirements of computational power, accuracy and efficiency, a mixture of both of these approaches might be used in order to balance these factors.</li>
</ul>

### Dataset Used
<p>The NCBI disease corpus is utilized for training and testing the NER model. This corpus contains annotated sentences with disease names, which serve as a research resource for the biomedical natural language processing community</p>
<p>The Dataset files used are:</p>
<ul>
  <li><b> Training Dataset: </b> <i>NCBItrainset_corpus.txt</i></li>
  <li><b> Testing Dataset: </b> <i>NCBItestset_corpus.txt</i></li>
</ul>

### Objective
<p> The main objective of this project is to create a custom NER model, which is powerful and robust enough to handle vast amounts of annotated biomedical texts. Thus, along with PyTorch and Cuda, a BiLSTM 
  architecture is used for increasing accuracy. </p>

## Procedure
<ol>
  <h3><li>Data Pre-processing</li></h3>
  <ul>
    <li><b>Dataset Parsing:</b><p>The training dataset files are parsed and the sentences and annotations are read, formatted and stored.</p></li>
    <li><b>Data Tagging:</b><p>The sentences are broken down into tokens, which are then tagged on the basis of a custom IO tagging scheme and stored in a separate text file.</p></li>
  </ul>

  <h3><li>Data Encoding</li></h3>
  <ul>
    <li><b>Label Encoding:</b><p>The words are encoded using the "Label Encoder" from "sklearn" library, with a special "UNK" (Unknown) tag to handle vocabulary during testing.</p></li>
    <li><b>PyTorch Compability:</b><p>A custom "NERDataset" class is used so that it is compatible with "DataLoader" from PyTorch.</p></li>
  </ul>

  <h3><li>Defining the Model and Parameters</li></h3>
  <ul>
    <li><b>NERModel:</b><p>An LSTM-based RNN (Recurrent Neural Network) model is defined, consisting of an embedding layer, an LSTM layer, and a fully connected layer for classification.</p></li>
  </ul>

  <h3><li>Training the Model</li></h3>
  <ul>
    <li><b>Training Procedure:</b><p>The model is trained using the AdamW optimizer, and uses the CrossEntropy loss function, and is run only for a few epochs. The model is then saved as "NER_Model.pth" in the main file directory.</p></li>
  </ul>
</ol>
