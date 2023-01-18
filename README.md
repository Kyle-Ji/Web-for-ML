# Web-for-ML

Web For ML (Machine Learning) is a project showing the possibility and performance of running pretained machine learning modeuls with [ONNX Runtime](https://github.com/microsoft/onnxruntime). 

The project focues on measuring the performance including time and memory usage of [ONNX Runtime Node.js binding](https://onnxruntime.ai/docs/get-started/with-javascript.html#onnx-runtime-nodejs-binding), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html#onnx-runtime-web), [ONNX Runtime for React Native](https://onnxruntime.ai/docs/get-started/with-javascript.html#onnx-runtime-for-react-native).

## Installation

1. [Node.js](https://nodejs.org/en/)

Then 
```Shell
# install latest release version
npm install onnxruntime-node
```

2. Python packages
```Shell
# Using a new conda environment:
conda create -n mlweb python
conda activate mlweb
```

Install required packages
```
pip install onnxruntime-gpu torch skl2onnx torchvision torchtext torchdata
```

**! Warning !**: Do not install `tensorflow` and `tf2onnx` together with above packages. They have package version conflicts with each other.

Test the enviroment

```Shell
cd onnx-in-python
python pytorch_cv.py
python python_nlp.py
python skl_cv.py
```


## References

1. [ONNX Model Zoo](https://github.com/onnx/models)
2. [Model visualization](https://github.com/lutzroeder/Netron)
3. [How to Run Machine-Learning Models in the Browser using ONNX](https://hackernoon.com/how-to-run-machine-learning-models-in-the-browser-using-onnx)
4. [JavaScript Tutorial](https://zh.javascript.info/intro)
5. [JavaScript 语言入门教程](https://wangdoc.com/javascript/)
6. [ONNX Web Tutorial](https://rekoil.io/blog/onnxruntime-web-tutorial/)
7. [Learn Git Branching](https://learngitbranching.js.org/)
8. [Top 50+ Linux Commands You MUST Know](https://www.digitalocean.com/community/tutorials/linux-commands)
