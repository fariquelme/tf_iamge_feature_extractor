import tensorflow as tf
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.nasnet import NASNetLarge 
from tensorflow.python.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.xception import Xception
from tensorflow import keras as k
import numpy as np
from PIL import Image
from glob import glob
import pathlib


# TODO: Add image augmentation
# TODO: Add automatic creation of features directory recursively
# TODO: Add support for multiple image file extensions
class ImageFeatureExtractor:

    def __init__(self, batch_size=10):
        """
        Description:
        -----------
        Feature Extractor for images using common deep learning image clasification models

        Usage: 
        ------
        fex = FeatureExtractor(batch_size=10)
        fex.extract_features('ResNet152V2', 'path/to/images_dir/', 'path/to/features_dir/', int_n_image_batches)

        Attributes
        ----------
        - batch_size : int
            > Number of images to fit in memory
        """

        self.batch_size = batch_size

        self.img_shapes = {
                        'DenseNet201' : (224,224),
                        'ResNet152V2' : (224,224),
                        'InceptionV3' : (299,299),
                        'NASNetLarge' : (331,331),
                        'VGG19' : (224,224),
                        'Xception' : (299,299)
                     }
        
        self.models_output_layers = {
                        'DenseNet201' : 'relu',
                        'ResNet152V2' : 'post_relu',
                        'InceptionV3' : 'mixed10',
                        'NASNetLarge' : 'activation_259',
                        'VGG19' : 'block5_conv4',
                        'Xception' : 'block14_sepconv2_act'
                     }
            
        self.fmodels = [  
                    'DenseNet201',
                    'ResNet152V2',
                    'InceptionV3',
                    'NASNetLarge',
                    'VGG19',
                    'Xception'
                 ]
    def preprocess_image(self, img, shape, normalize=True):
        img = img.resize(shape, Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        if normalize:
            img = img/255
        return img

    def load_image(self, path) :
        img = Image.open(path)
        return img

    def model_summary(self, model):
        model.summary()
        return k.utils.plot_model(model, to_file='model_graph.png', show_shapes=True, show_layer_names=True, dpi=1000)
    
    def  extract_features(self, model_name, images_path_prefix, features_save_path, n_batches):
        print(f'Processing: {images_path_prefix} ({model_name})')
        k.backend.clear_session()
        model=globals()[model_name](include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        layers_dict = {layer.name: layer.output for layer in model.layers} # all layer outputs
        output_layer = [layers_dict[self.models_output_layers[model_name]]]

        batch_paths = np.array_split(self.get_image_paths(images_path_prefix), n_batches)
        save_path_prefix = f'{features_save_path}/{model_name}/'
        pathlib.Path(save_path_prefix).mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(batch_paths):
            print(f'\tbatch {i}/{len(batch_paths)}')
            x = np.array([ self.preprocess_image(self.load_image(file_path), self.img_shapes[model_name], normalize=True) for file_path in batch])
            extract = k.Model(model.inputs, output_layer)
            features = extract.predict(x, batch_size=self.batch_size)
            for j, path in enumerate(batch):
                filename = path.split('/')[-1].split('.')[0]
                np.savez_compressed(f'{save_path_prefix}/{filename}.npz')
            print(batch[i], features[i].shape)
            
        
    def get_image_paths(self, images_path_prefix):
        return glob(images_path_prefix + '*.jpg')

