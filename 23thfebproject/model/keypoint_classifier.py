import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
            self,
            model_path='model/keypointclassifier.tflite'  # Update the model path to point to your TFLite model
    ):
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
            self,
            input_data,
    ):
        # Ensure input_data is in the right format (numpy array)
        input_data = np.array(input_data, dtype=np.float32)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        result = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Assuming your model outputs probabilities, use argmax to get the predicted class index
        result_index = np.argmax(result)

        return result_index
