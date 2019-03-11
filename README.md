# Paper-Implementation-Matrix-Factorization-Recommender-Systems-Netflix
[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/harshraj11584/Paper-Implementation-Matrix-Factorization-Recommender-Systems-Netflix/blob/master/LICENSE) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
## IEEE paper **"Matrix Factorization Techniques for Recommender Systems"** 
### - Yehuda Koren, Robert Bell, Chris Volinsky   
### Python 3.6

Links to original paper published by IEEE Computer Society : [[1]](https://ieeexplore.ieee.org/document/5197422), [[2]](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) 

Link to Netflix Dataset Used : [[1]](https://www.kaggle.com/netflix-inc/netflix-prize-data)

### Files 

1) **Presentation.pdf** : Explains the paper. Was written in Latex Beamer, tex code is in _presentation.tex_     

2) **recommender_final.py** : The final recommender. Includes biases and regularization. Requires **mf.py** to be imported to run. Use directly on any dataset by changing line 19 in **recommender_final.py**.   

3) **recommender_final_toy_dataset.py** shows how exactly Matrix Factorization Techniques work by considering a 5x5 toy dataset.   

4) The **.ipynb_** files include visualizations of RMSE decreasing with iterations when fitting on the training dataset. All **.ipynb** files are standalone and do not require importing **mf.py**    

5) **feasible_data_n.txt** : Files with only the first n datapoints from whole dataset. Used for Testing.

5) **Training** and **Testing Data** :  
Not given separately. Program randomly separates k% of data as Test data, trains on remaining, then tests on the k% values. Default k=20, can be changed on line 154.


### Error Analysis

![img](https://github.com/harshraj11584/Paper-Implementation-Matrix-Factorization-Recommender-Systems-Netflix/blob/master/Error%20Chart.png)
