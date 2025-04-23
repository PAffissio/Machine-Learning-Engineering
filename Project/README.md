Patrick Affissio
Machine Learning in Truss analysis 
This project seeks to improve on current methods of truss analysis (Closed form analysis, FEM) using a graph based neural network to predict deflections of a 10 bar truss under load.
Custom datasets were generated for use in this model and are located in the repo.
In order to run the model, all that is needed are the dataset files (.npy) and the trussGNN.ipynb file. Open the jupiter notebook and run all sections to train and evaluate the model.
This model performed quite well, with a mean absolute error on the order of 1e-4. Some improvements can be made in the normalization of the data in order to avoid vanishing gradients disproportionately small deformations
