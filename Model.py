import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.
from random import randint
from official.nlp import optimization  # to create AdamW optimizer
from keras.models import load_model

class BERT():
    
  def __init__(self, n_classes, epochs = 15):
    """This function initilaizes the model class

      Args:
          n_classes ([int]): [number of categories (covid-19, party related ... etc)]
          epochs (int, optional): [number of training epochs for training]. Defaults to 15.
    """
    self.n_classes = n_classes
    self.build_classifier_model()
    self.epochs = epochs
  def build_classifier_model(self):
    """[This function builds the model, it includes the all the ANN layers and preprocessing]
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2', trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.0)(net)
    net = tf.keras.layers.Dense(self.n_classes, activation=tf.keras.activations.softmax, name='classifier')(net)
    self.model = tf.keras.Model(text_input, net)
  def save_model(self, save_path='model.h5'):
    """[fucntion to save the model]

    Args:
        save_path (str, optional): [path to save the model must be in form somepath/ModelName.h5]. Defaults to 'model.h5'.
    """

    self.model.save(save_path)

  def load_model(self, model_path='model.h5'):
    """[function to load the trained model]

    Args:
        model_path (str, optional): [path to load the model from, must be in form somepath/ModelName.h5]. Defaults to 'model.h5'.
    """
    steps_per_epoch = 1500
    num_train_steps = steps_per_epoch * self.epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    self.model = load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer,'AdamWeightDecay':optimizer})
    
  def train(self, TrainGen, ValGen, class_weight=None):
    """[summary]

      Args:
          TrainGen ([type]): [Data loader for training samples]
          ValGen ([type]): [Data loader for validation samples]
          class_weight ([dictionary], optional): [a dictionary to assign higher weights for classes that occurs
          less frequently, can be constructed from TrainGen.class_weights()]. Defaults to None.
    """
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.categorical_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    steps_per_epoch = len(TrainGen)
    num_train_steps = steps_per_epoch * self.epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    self.model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
    
    self.history = self.model.fit(x=TrainGen,
                               validation_data=ValGen,
                               epochs=self.epochs,
                               class_weight=class_weight)
