# Custom NER (Named Entity Recognition) Model with BiLSTM Architecture

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
