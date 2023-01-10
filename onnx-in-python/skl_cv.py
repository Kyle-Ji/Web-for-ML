from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
clr = LogisticRegression()
clr.fit(X_train, y_train)
print(clr)

# LogisticRegression()


from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())



from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())



import numpy
import onnxruntime as rt

sess = rt.InferenceSession("logreg_iris.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

# OUTPUT:
#  [0 1 0 0 1 2 2 0 0 2 1 0 2 2 1 1 2 2 2 0 2 2 1 2 1 1 1 0 2 1 1 1 1 0 1 0 0
#   1]