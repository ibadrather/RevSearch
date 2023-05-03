"""
# Example usage
preprocessed_image = preprocess_image("path/to/your/image.jpg", (224, 224))
output = infer_onnx_model("path/to/your/model.onnx", preprocessed_image)
"""

import numpy as np
import onnxruntime as ort


def infer_onnx_model(model_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Perform inference on an ONNX model.

    Args:
        model_path (str): The path to the ONNX model file.
        input_data (numpy.ndarray): The preprocessed image as a NumPy array.

    Returns:
        numpy.ndarray: The output of the ONNX model as a NumPy array.
    """
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    prediction = sess.run([output_name], {input_name: input_data})

    return np.array(prediction)


def load_onnx_model(model_path: str):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name
