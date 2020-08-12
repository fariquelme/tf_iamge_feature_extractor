import tensorflow as tf
import tensorflow.keras.applications as tf_apps
from inspect import getmembers, isfunction
from tensorflow import keras as k
import numpy as np
from PIL import Image
from glob import glob
import pathlib
import math
from tqdm import tqdm
import sys

# TODO: Add image augmentation
class ImageFeatureExtractor:
    """
    Description:
    -----------
    Feature Extractor for images using common deep learning image clasification models

    Usage: 
    ------
    fex = ImageFeatureExtractor(batch_size=10)
    fex.list_models()
    model = fex.get_model('DenseNet121')
    fex.extract_features(model, 'path/to/images_dir/', 'path/to/features_dir/', int_n_image_batches)

    Attributes
    ----------
    - batch_size : int
        > Number of images to fit in memory by iteration
    """
    
    # Get available tensorflow.keras.applications
    models = {f[0]:f[1] for f in getmembers(tf_apps) if isfunction(f[1])}
    
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
    
    def list_models(self):
        models = ' Available Models '.center(30, '=')
        models = models + ''.join([f'\n> {model_name}' for model_name in self.models.keys()])
        print(models)
    
    def get_model(self, model_name, include_top=True, weights='imagenet', classes=1000, **kwargs):
        return self.models[model_name](include_top=include_top, weights=weights, classes=classes, **kwargs)
    
    def get_layer_specs(self, model):
        features_layer_name = None
        features_layer_shape = None
        input_layer_shape = None
        # Reverse layers order to get first output that is not max pool and has shape 4
        for layer in model.layers[::-1]:
            if layer.__class__.__name__ != 'MaxPooling2D' and len(layer.output.shape) == 4:
                if features_layer_name is None:
                    features_layer_name = layer.name
                    features_layer_shape = layer.output.shape[1:]
            if isinstance(layer,tf.python.keras.engine.input_layer.InputLayer):
                input_layer_shape = layer.output.shape[1:]
            
        return features_layer_name, features_layer_shape, input_layer_shape
    
             
    def model_summary(self, model, weights='imagenet', classes=1000, **kwargs):
        if isinstance(model, str):
            model=self.get_model(model_name, weights=weights, classes=classes, **kwargs)
        model.summary()
        return k.utils.plot_model(model, to_file='model_graph.png', show_shapes=True, show_layer_names=True, dpi=1000)

    def preprocess_image(self, img, shape, normalize=True, dtype=np.float32):
        # Reshapes image to network input size, and normalizes it
        img = img.resize(shape, Image.BILINEAR)
        img = np.array(img, dtype=dtype)
        if normalize:
            img = img/255
        return img

    def load_image(self, path) :
        img = Image.open(path)
        return img

    def get_image_paths(self, images_path_prefix, extension='jpg'):
        return glob(images_path_prefix + f'*.{extension}')
    
    def  extract_features(self,
                          model,
                          images_path_prefix,
                          features_save_path,
                          features_layer_name = None,
                          features_layer_shape = None,
                          input_layer_shape = None,
                          normalize=True):
        
        # Set model if string is given
        if isinstance(model, str):
            model = self.get_model(model)
    
        print(f'[{model.name}]: Processing images from {images_path_prefix}')
        k.backend.clear_session()
        
        # Get network input size for the imag, features layer name and shape
        if not (features_layer_name and features_layer_shape and input_layer_shape):
            features_layer_name, features_layer_shape, input_layer_shape = self.get_layer_specs(model)
        
        # Set output layer to features layer
        layers_dict = {layer.name: layer.output for layer in model.layers} # all layer outputs
        output_layer = [layers_dict[features_layer_name]]
        
        # Define cut-off model
        extractor = k.Model(model.inputs, output_layer)

        # Load image paths
        image_paths = self.get_image_paths(images_path_prefix)
        
        # Split image paths into n batches
        batch_paths = [image_paths[i:i + self.batch_size] for i in range(0, len(image_paths), self.batch_size)]  
        
        # Create features save directory
        save_path_prefix = f'{features_save_path}/{model.name}_{features_layer_name}/'
        pathlib.Path(save_path_prefix).mkdir(parents=True, exist_ok=True)
        
        # Process images by batch
        for i, batch in enumerate(tqdm(batch_paths)):
            
            # Create input array
            X = np.zeros((len(batch),) + input_layer_shape)
            for j, file_path in enumerate(batch):
                img = self.load_image(file_path)
                
                # Normalize image and resize
                img = self.preprocess_image(img, input_layer_shape[:-1], normalize=normalize)
                X[j,:] = img
                
            # Extract features
            features = extractor.predict(X, batch_size=self.batch_size)
            
            # Save image features
            for j, path in enumerate(batch):
                filename = path.split('/')[-1].split('.')[0]
                np.savez_compressed(f'{save_path_prefix}/{filename}.npz', embedding=features[j])

