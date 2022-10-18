import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist


class Network:
    def __init__(self, **kwargs):
        self.training_datasets = list(kwargs['training_data'])
        self.testing_datasets = list(kwargs['testing_data'])
        self.metrics = kwargs['metrics']
        self.loss = kwargs['loss']
        self.validation_split = kwargs['validation_split']
        self.input_shape = kwargs['input_shape']

        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.classes = kwargs['classes']
        self.normal_activation_function = kwargs['activation']
        self.final_activation_function = kwargs['output_layer']['activation']

        self.history = None

        self.shape_date()
        self.model = Sequential()
        self.build_model(**kwargs)
        self.model.compile(loss=self.loss, metrics=self.metrics, **kwargs.get('compile', {}))

    def run(self):
        self.history = self.model.fit(
            *self.training_datasets,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split
        )

    def evaluate(self):
        return self.model.evaluate(*self.testing_datasets)

    @staticmethod
    def expand_filters(filters):
        for num, layer in enumerate(filters):
            if isinstance(layer, int):
                filters[num] = (layer, layer)
        return filters

    def build_model(self, **kwargs):
        layers = zip(kwargs['filters'], self.expand_filters(kwargs['kernel']))
        for num, (nodes, filters) in enumerate(layers):
            self.model.add(
                tf.keras.layers.MaxPool2D(
                    **({} if filters == (0, 0) else {'strides': filters})  # should spread strides out into method unless it isn't specified
                ) if nodes == 0 else
                tf.keras.layers.Conv2D(
                    nodes,
                    filters,
                    activation=self.normal_activation_function,
                    **({'input_shape': self.input_shape} if num == 0 else {})
                )
            )
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.classes, activation=self.final_activation_function))

    def shape_date(self):
        self.training_datasets[0] = self.training_datasets[0].reshape(self.training_datasets[0].shape[0], self.training_datasets[0].shape[1], self.training_datasets[0].shape[2], 1) / 255
        self.testing_datasets[0] = self.testing_datasets[0].reshape(self.testing_datasets[0].shape[0], self.testing_datasets[0].shape[1], self.testing_datasets[0].shape[2], 1) / 255

        self.training_datasets[1] = tf.one_hot(self.training_datasets[1].astype(np.int32), depth=10)
        self.testing_datasets[1] = tf.one_hot(self.testing_datasets[1].astype(np.int32), depth=10)

    def get_history(self, *args):
        return { arg: self.history.history[arg] for arg in args } if args else self.history.history


def __main__():
    training, testing = mnist.load_data()
    n = Network(**{
        'input_shape': (28, 28, 1),
        'batch_size': 1000,
        'classes': 10,
        'epochs': 4,
        'validation_split': 0.1,
        'metrics': ['acc'],
        'loss': 'categorical_crossentropy',
        'optimizer': tf.keras.optimizers.RMSprop(),

        'training_data': training,  # training dataset
        'testing_data': testing,  # testing dataset

        'filters': [32, 0, 64],  # this will act as the number of filters per layer inside of the Conv2D. If 0 it will switch from Conv2D to MaxPool2D.
        'kernel': [5, 0, 3],  # if an integer is given it will expand from number to (number, number), If layers is 0 at this point, it will instead turn into the number for strides.
        'activation': 'softmax',  # activation function. Whatever is given here is handed over as is.

        'output_layer': {
            'activation': 'softmax',  # this is only for the final layer of the Network. (Uses a Dense layer and doesn't change)
        },
    })

    n.run()  # This does training
    a = n.evaluate()  # This does evaluation
    print(n.get_history('acc', 'loss'))


if __name__ == '__main__':
    __main__()
