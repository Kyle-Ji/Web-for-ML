# may put this function in another utility file
def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

# replace `import tensorflow as tf` with this line
# or insert this line at the beginning of the `__init__.py` of a package that depends on tensorflow
tf = import_tensorflow()    

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import os
import requests
import tf2onnx
import onnxruntime as rt
from os import path
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import onnxruntime

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

img_path = 'ade20k.jpg'
if not path.exists(img_path):
    image_url = "https://github.com/onnx/tensorflow-onnx/raw/main/tests/ade20k.jpg"
    image_data = requests.get(image_url).content
    with open(img_path, 'wb') as img:
        img.write(image_data)

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = ResNet50(weights='imagenet')

preds = model.predict(x)
print('Keras Predicted:', decode_predictions(preds, top=3)[0])
model.save(os.path.join("/tmp", model.name))


spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(output_names, {"input": x})

print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])

# make sure ONNX and keras have the same results
np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-4)