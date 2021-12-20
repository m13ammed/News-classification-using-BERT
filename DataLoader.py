from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import class_weight
import pandas as pd
import tensorflow as tf

from math import ceil
class Dataset(Sequence):

  def __init__(self, DataPath, BatchSize = 48, mode = "train",seed = 13):
    """[initializing of data loader class]

      Args:
          DataPath ([type]): [path to data must be in form SomePath/name.json]
          BatchSize (int, optional): [batch size]. Defaults to 48.
          mode (str, optional): [either train,val, or test to make either a training data loader, validation, or testing]. Defaults to "train".
          seed (int, optional): [description]. Defaults to 13.
    """
    self.DataPath = DataPath
    self.BatchSize = BatchSize 
    self.mode = mode
    self.seed = seed
    self.load_split_json()
    self.n_classes = len(self.ClsIdxDic)
    self.num_samples = self.X.shape[0]
    self.idxList = [i for i in range(0,self.num_samples)]

  def __getitem__(self, index):
    """[function to return s single batch ]

      Args:
          index ([type]): [index of returned batch]

      Returns:
          [type]: [the title, a one hot encoded vector representing the class of the title]
    """
    start = index*self.BatchSize
    ending = index*self.BatchSize + self.BatchSize
    if ending >= len(self.idxList):
      ending = len(self.idxList) 
    tempList = self.idxList[start:ending]
    X_batch = self.X[tempList]
    Y_batch = self.Y[tempList]

    return X_batch, tf.one_hot(Y_batch, self.n_classes)
  
  def __len__(self):
    """[number of bacthes in the dataloader]

    """
    return int(ceil(len(self.idxList) / self.BatchSize))

  def __on_eopch_end__(self):
    """[function called at the end of each training epoch]
    """
    np.random.seed(self.seed)
    np.random.shuffle(self.idxList)
    print("shuffle done!")

  def load_split_json(self):
    """[fucntion to load the json file containing the data, and split into train, test, and validation]
    """
    DF = pd.read_json(self.DataPath)
    X = DF["Title"]
    Y = DF["Practice Area"]
    self.ClsIdxDic={}
    self.cls_names = list(Y.unique())
    self.cls_names.sort()
    self.n_classes = len(self.cls_names)

    for i,name in enumerate(self.cls_names):
      self.ClsIdxDic.update({name:i})

    self.inv_ClsIdxDic = {v: k for k, v in self.ClsIdxDic.items()}

    X_train, X_temp, Y_train, Y_temp = train_test_split(X,Y, test_size = 0.3, random_state = self.seed, stratify = Y)
    if self.mode != "train":
      X_val, X_test, Y_val, Y_test = train_test_split(X_temp,Y_temp, test_size = 0.5, random_state = self.seed, stratify = Y_temp)
      if self.mode == "val":
        self.X = X_val.to_numpy().reshape(-1)
        self.Y = np.array(list(map(lambda x: self.ClsIdxDic[x],Y_val))).reshape(-1)
      elif self.mode == "test":
        self.X = X_test.to_numpy().reshape(-1)
        self.Y = np.array(list(map(lambda x: self.ClsIdxDic[x],Y_test))).reshape(-1)
      else: 
        print("please use a valid mode options are [train, val, test]")
    else:
        self.X = X_train.to_numpy().reshape(-1)
        self.Y = np.array(list(map(lambda x: self.ClsIdxDic[x],Y_train))).reshape(-1)

  def class_weights(self):
    """[funciton to generate weights assign higher weights for classes that occurs
          less frequently]

      Returns:
          [dictionary]: [dictionary of of weight assigned for each class]
    """

    class_weights = class_weight.compute_class_weight("balanced",list(self.ClsIdxDic.values()), self.Y)
    return dict(zip(list(self.ClsIdxDic.values()),class_weights))

  def get_true(self):
    """

      Returns:
          [np.array]: [true lables in form of [0,1,2,3] (not hot encoded)]
    """
    return self.Y
