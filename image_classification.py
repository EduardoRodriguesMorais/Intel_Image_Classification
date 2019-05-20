import numpy as np 
import pandas as pd 
from pathlib import Path

from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import os
    
data_folder = Path("../input")
data_folder.ls()

#Data augmentation
transforms = get_transforms(do_flip = True, 
                            flip_vert = True, 
                            max_rotate = 100.0, 
                            max_zoom = 1.5, 
                            max_lighting = 0.2, 
                            max_warp = 0.2, 
                            p_affine = 0.75, 
                            p_lighting = 0.75)

#Data Bunch
data = (ImageList.from_folder(path=data_folder)
        .split_by_folder('seg_train', 'seg_test')
        .label_from_folder()
        .add_test_folder('seg_pred')
        .transform(transforms, size=224)
        .databunch(path='.', bs=32)
        .normalize(imagenet_stats)
       )

data.classes

data.show_batch(rows = 6)

#Model CNN Learner
learn = cnn_learner(data, models.resnet50, metrics = [accuracy], model_dir = '/tmp/model/')

#CallBacks
reduce_lr = ReduceLROnPlateauCallback( learn, patience = 5, factor = 0.2, monitor = 'accuracy')
early_stopping = EarlyStoppingCallback( learn, patience = 10, monitor = 'accuracy')
save_model = SaveModelCallback( learn, monitor = 'accuracy', every = 'improvement')
callbacks = [reduce_lr, early_stopping, save_model]

#Unfreeze layers
learn.unfreeze()

#Find best initial learn rate 
learn.lr_find()

learn.recorder.plot(suggestion = True)

min_grad_lr = learn.recorder.min_grad_lr

#Performs model training
learn.fit_one_cycle(50, min_grad_lr, callbacks = callbacks, wd = 1e-3)

learn.save('model')

learn.recorder.plot_losses()

learn.recorder.plot_metrics()

#Cria um interpretador de classificação
interp = ClassificationInterpretation.from_learner(learn)

predictions, y, loss = learn.get_preds(with_loss = True)

#Apresenta Accuracy
acc = accuracy(predictions, y)
print('-> Accuracy: ', acc)

#Apresenta matriz de confusão
interp.plot_confusion_matrix()