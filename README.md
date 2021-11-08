# GlobalTimeSeriesCoherenceMatrices

## Description
Code for the Paper Constructing Global Coherence Representations:Identifying Interpretability and Coherences of Transformer Attention in Time Series Data. 
It is about creating coherence matrices which represent the attention from each symbol to each other symbol to enhance the understanding of the global class representation. It can be used as a classification model while providing a nice visualization and boost to the interpretability of the global classes <br>


The project contains two Jupiter notebooks which provide the model from the publication and also includes the weights for the published results (in the "saves"-folder for 500 and 15 epochs). The saved weights need in the saved folder without a nested structure to be loaded. 

- GlobalTransformerInterpretation-ForReproducibility.ipynb: Contains the model from the paper for reproducibility.
- GlobalTransformerInterpretation-WithFix.ipynb: Contains a fix; results are not compatible with the publication.

The code was tested on four datasets (linked below and included in the repository) and trains with a 5 fold cross-validation. Each fold trains 2 models and 10 coherence representations:

Models:
- Normal input data
- Symbolic data (SAX)

Coherence Representations:
- Full Coherence Attention Matrices based on sum
- Full Coherence Attention Matrices based on relative average
- Column  Reduced  Coherence  Attention  Matrices based on sum
- Column  Reduced  Coherence  Attention  Matrices based on relative average
- Global Trend Matrix based on max of sum 
- Global Trend Matrix based on max of relative average
- Global Trend Matrix based on average of sum 
- Global Trend Matrix based on average of relative average
- Global Trend Matrix based on median of sum 
- Global Trend Matrix based on median of relative average


At the end of the notebook the coherence matrices can be analyzed with the given visualizations.

## Dependencies
A list of all needed dependencies (other versions can work but are not guaranteed to do so):

python=3.7.3<br>
tensorflow==2.2.0<br>
tensorflow_addons==0.11.2<br>
tensorflow_probability==0.7.0<br>
seaborn==0.10.1<br>
scipy==1.4.1<br>
scikit-learn==0.23.2<br>
pyts==0.11.0<br>
pandas==1.0.0<br>
numpy==1.18.5<br>
matplotlib==3.3.1<br>



## Cite and publications
This code represents the used model for the following publication: <br>
https://ieeexplore.ieee.org/abstract/document/9564126 <br>

If you use, build upon this work or if it helped in any other way, please cite the linked publication.


## Datasets

Included datasets are:

http://www.timeseriesclassification.com/description.php?Dataset=SyntheticControl <br>
http://www.timeseriesclassification.com/description.php?Dataset=ECG5000 <br>
http://www.timeseriesclassification.com/description.php?Dataset=Plane <br>
http://www.timeseriesclassification.com/description.php?Dataset=PowerCons

