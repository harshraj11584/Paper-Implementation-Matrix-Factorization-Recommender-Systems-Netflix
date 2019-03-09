# Paper-Implementation-Matrix-Factorization-Recommender-Systems-Netflix
## IEEE paper **"Matrix Factorization Techniques for Recommender Systems"** 
### - Yehuda Koren, Robert Bell, Chris Volinsky   
### Python 3.6

Links to original paper published by IEEE Computer Society : [[1]](https://ieeexplore.ieee.org/document/5197422), [[2]](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) 

Link to Netflix Dataset Used : [[1]](https://www.kaggle.com/netflix-inc/netflix-prize-data)

### Instructions 

1) _Presentation.pdf_ explains the paper. Was written in Latex Beamer, tex code is in _presentation.tex_     

2) _recommender_final.py_ is the final recommender, and requires _mf.py_ to be imported to run. Use directly on any dataset by changing line 19 in _recommender_final.py_.   

3) _recommender_final_toy_dataset.py_ shows how exactly Matrix Factorization Techniques work by considering a 5x5 toy dataset.   

4) The _*.ipynb_ files include visualizations of RMSE decreasing with iterations when fitting on the training dataset. All _*.ipynb_ files are standalone and do not require importing _mf.py_

5) Train and Test Data not given separately. Program randomly separates k% of data as Test data, trains on remaining, then tests on the k% values. Default k=20, can be changed on line 154.
