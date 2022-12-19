# Work 3: Lazy learning exercise

KNN (K-Nearest Neighbors) is a supervised machine learning algorithm used for classification and regression. The algorithm works by finding the 'K' nearest neighbors of a given data point and using those neighbors to predict the class of the data point. KNN is a non-parametric and lazy learning algorithm that does not make any assumptions about the underlying data distribution. In this project it has been implemented KNN with several reduction algorithms. To validate the model, several datasets are used, namely; Adult, Pen-Based and Hypothyroid.

### Installation 

This repository was written and tested for python 3.7. To install the virtual environment and the required libraries on windows, run:

```bash
python -m venv group5_work3
group5_work3\Scripts\activate.bat
pip install -r requirements.txt
```

On Unix or MacOS, run:

```bash
python -m venv group5_work3
source group5_work3/bin/activate
pip install -r requirements.txt
```

Distance metrics were implemented using pytorch. If you want to use the GPU, you can install the pytorch library supporting GPU as follows:

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

If the machine does not have a GPU, distances will be computed by using the CPU.

**Note:** You would require at least 4GB of RAM in the GPU for the *adult* dataset. If you do not want to use pytorch to compute the distance matrices, you can use scipy by setting the parameter --gpu to "no" when calling to main.py

### Usage

You can run the experiments with:

```python
python main.py --dataset <dataset>
               --run_experiments <run_experiments>
               --reduction <reduction>
               --visualize_results <visualize_results>
               --gpu <gpu>
```
Specifying the parameters  according to the following table:

| Parameter             | Description                                                                                                                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **dataset**           | Dataset to use. If set to 'pen-based', the Pen-Based dataset is used. If set to 'hyp', the Hypothyroid dataset will be used. If set to 'adult', the adult dataset will be used.                                   |
| **run_experiments**   | If set to 'yes', 10-fold cross-validation experiments are performed over a set of hyperparameters. Results are saved in a csv file in the results folder for each dataset.                                        |
| **reduction**         | If set to 'yes', experiments are performed for different reduction algorithms using the best hyperparameters found for each dataset which are defined in main.py.                                              |
| **visualize_results** | If set to 'yes', different plots will be generated and saved in the results folder inside the corresponding dataset's folder. In order to generate the plots, the results csv files must be in the results folder. |
| **pytorch**           | If set to 'no', pytorch is not used to compute the distance matrices during experiments and scipy cdist function is used instead.                                                                                 |

For each experiment, the code will save the distances matrices for each fold and each combination of distance-weight which will be reused in other executions so that executions are computed faster and the same computation is not performed twice. Also, execution times are saved the first time a distance matrix is computed, and those times are added to the execution time for those executions that use a precomputed distance matrix.

### Examples

The following execution would run each reduction algorithm with the best hyperparameters found which are defined in the script and it will return the results in the console.

```python
python main.py --dataset hyp --run_experiments yes --reduction yes --visualize_results yes --gpu no
```

This other execution would run different experiments for a combination of hyperparameters defined in the code. Distances will be computed using scipy as the parameter *gpu* is set to 'no'.

```python
python main.py --dataset hyp --run_experiments yes --reduction no --visualize_results yes --gpu no
```