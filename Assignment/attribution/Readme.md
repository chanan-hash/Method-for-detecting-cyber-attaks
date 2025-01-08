## APP & Attribution classification

### Running the code
For runing the code you need to check you have at least *python version* of `3.11`.
We ran our code on *python version* `3.12.7`.   
In addition check you have the libreries that needed. If not you can install them by running this commands:
```bash
pip install pandas
pip install numpy
pip install sklearn
pip install matplotlib
pip install seaborn
``` 
Another way, in the notebook there is all the installation needed, just take down the comment sign and run the block.


---
### APP section running
make sure you have the scripts (`"Train model.py"` and `"Load model.py"`) and the `"train.csv"`, `"test.csv"` and `"val_without_labels.csv"` all in the same folder.

run in cmd: (if using pycharm for exmaple, can run the scripts from there too)

python `"Train model.py"`

python `"Load model.py"`


at the end you should a file called `prediction.csv"`, this is the prediction file for the APP.


---
### Attribution section running
make sure you have the `Test_inference_Attribution.ipynb`  and the `"train.csv"`, `"test.csv"` and `"val_without_labels.csv"` all in the same folder.

***Note:*** we have 2 `train.csv`, `test.csv` and `"val_without_labels.csv"`, one for the APP and one for the attribution, they both with the same name, so before running check for each section that this is the suitable dataset


---
### Dataset path
Please check that you have the datasets in the same folder/directory of the code.  
If not you can or update the path to the right directory, or add the datasets to the same folder that the code is in.


---
### Output
In the end the output will be a new csv file that with only one column of the predict label/attribute.