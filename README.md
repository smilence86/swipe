Score a girl's picture with cnn model, swipe left or right in tantan app automatically.

# How to use:
  
1、Recommend to use venv environment (pip3 install virtualenv)
```
$ git clone https://github.com/smilence86/swipe.git
$ cd ./swipe
$ virtualenv -p python3.5 .venv
$ source .venv/bin/activate
```
  
2、Install requirements
```
$ pip3 install --no-cache-dir -r requirements.txt -i https://pypi.doubanio.com/simple/
```
  
3、Run script
```
$ python3 train_model.py
```
  
4、Exit venv environment
```
$ deactivate
```
  
  
  
If can't run analysis.py, install requirements to resolve:
```
$ sudo apt-get install tcl-dev tk-dev python-tk python3-tk
```