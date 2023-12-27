### Часть 1:
- реализовать класс набора данных OxfordIIITPet
- реализовать вычисление метрик accuracy, balanced accuracy
- реализовать класс обучения Trainer (сохранение и выгрузку весов модели, логирование значений целевой функции и метрик на каждом шаге обучения, реализовать функции обучения на train set и evaluation на test set, обучение проводить на gpu)
- реализовать логирование в neptune.ai в классе Trainer
- подобрать препроцессинг данных на основе следующих преобразований: 
    - Обучающие данные (**пункт d не изменять**):
      1. RandomResizedCrop(224); 
      2. RandomHorizontalFlip(). 
      3. ColorJitter: scale hue, saturation, and brightness with coefficients uniformly drawn from [0.6, 1.4].
      4. Normalize(mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]) 

    - Тестовые данные (**не изменять**):
      1. Resize(256)
      2. CenterCrop(224)
      3. Normalize(mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225])
- визуализация батча до и после применения аугментаций (функция show_batch)

### Часть 2:
- реализовать функцию conv_block (models/blocks/vgg16_blocks.py)
- реализовать функцию classifier_block (models/blocks/vgg16_blocks.py)
- реализовать нейронную сеть VGG-16, используя функции conv_block и classifier_block
- обучить VGG-16 на наборе данных OxfordIIITPet

![VGG-16](https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network-1.jpg)

### Часть 3:
- реализовать классы блоков для ResNet-50 (models/blocks/resnet_blocks.py)
- реализовать нейронную сеть ResNet-50, используя предыдущий пункт (`notebooks/ResNet50-architecture.pdf`)
- добавить в ResNet-50 инициализацию весов (функция `_init_weights` в resnet50.py)
- в классе Trainer добавить возможность выбрать модель для обучения согласно cfg.model_name в train_cfg.py
- обучить базовую модель ResNet-50 на наборе данных OxfordIIITPet

### Часть 4 ([Bag of Tricks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)):
- добавить модификации ResNet-B,C,D (`notebooks/ResNet50-model_tweaks.pdf`)
- добавить label smoothing, linear scaling learning rate, learning rate warmup, cosine learning rate decay, no bias decay (функция `weight_decay_params` в resnet50.py), zero gamma
- обучить ResNet-50, используя все предложенные трюки

## Convolutional neural network:

Для этого задания используем библиотеку PyTorch, архитектуры, класс набора данных, метрики, процесс обучения реализуем самостоятельно. 
Можно использовать готовые решения PyTorch для label smoothing, learning rate scheduler, визуализации батчей 

### Набор данных OxfordIIITPet:
1) реализовать класс набора данных OxfordIIITPet
2) подобрать препроцессинг данных на обучающей выборке
3) визуализировать батч изображений до и после применения аугментаций (utils/visualization.py функция show_batch)

### VGG-16:
1) реализовать функцию conv_block (models/blocks/vgg16_blocks.py)
2) реализовать функцию classifier_block (models/blocks/vgg16_blocks.py)
3) реализовать нейронную сеть VGG-16, используя функции conv_block и classifier_block

### ResNet-50:
1) реализовать классы блоков для ResNet-50 (models/blocks/resnet_blocks.py)
2) реализовать нейронную сеть ResNet-50, используя предыдущий пункт (`notebooks/ResNet50-architecture.pdf`)
3) добавить в ResNet-50 инициализацию весов (функция `_init_weights` в resnet50.py)
4) добавить модификации ResNet-B,C,D (`notebooks/ResNet50-model_tweaks.pdf`)
5) добавить label smoothing, linear scaling learning rate, learning rate warmup, cosine learning rate decay, no bia decay, zero gamma


### Обучение моделей:
1) реализация класса обучения Trainer
2) добавить логирование в neptune.ai в классе Trainer (класс logs/Logger.py)
3) реализация функций подсчета метрик accuracy, balanced accuracy (utils/metrics.py) без использования сторонних библиотек (numpy, torch - можно)
4) во время обучения логировать значение целевой функции, accuracy и lr на обучающей выборке на каждом шаге
5) после окончания каждой эпохи обучения посчитать значения целевой функции, accuracy, balanced accuracy на тестовых данных, полученные значения логировать в neptune.ai
6) сохранить модель с лучшим значением accuracy на тестовой выборке (класс Trainer метод save_model)

### Этапы:
1) обучить VGG-16 на наборе данных OxfordIIITPet (часть 2): SGD, lr=1e-3, momentum=0.9, weight_decay=5e-4, bs=64, max_epoch=100
2) обучить базовую модель ResNet-50 на наборе данных OxfordIIITPet (часть 3): lr=0.001, bs=64, max_epoch=100
3) обучить модель ResNet-50 c модификациями B,C,D на наборе данных OxfordIIITPet, используя предложенные трюки (часть 4)
4) сравнить модели, обученные в предыдущих пунктах (можно добавить README.md/pdf/txt к репозиторию с описанием наблюдений)

### Общие требования к обучению моделей:
1. Все параметры обучения должны настраиваться в конфигах 
2. Визуализация батча до и после применения аугментаций (utils/visualization.py функция show_batch)
3. Каждый эксперимент и его параметры должны быть залогированы в neptune.ai (с понятным название эксперимента)
4. Во время обучения логировать значение целевой функции, accuracy, balanced accuracy и lr на обучающей выборке на каждом шаге
5. После окончания каждой эпохи обучения посчитать значения целевой функции, accuracy, balanced accuracy на тестовых данных, полученные значения логировать в neptune.ai
6. Сохранить модель с лучшим значением метрик на тестовой выборке (класс Trainer метод save_model)
7. Все модели обучать до насыщения accuracy на тестовой выборке
8. Обучать на GPU (например, [kaggle](https://www.kaggle.com/)/[colab](https://colab.research.google.com/)), пример notebook - `notebooks/kaggle_notebook.ipynb` 
9. Во время обучения переводить модель в режим model.train(), во время evaluation - model.eval()



### Dead line - 30 ноября

### Полезные ссылки
- [набор данных OxfordIIITPet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [набор данных OxfordIIITPet на kaggle](https://www.kaggle.com/datasets/polinakhanzhina/oxford-iiit-pet)
- [статья по VGG](https://arxiv.org/pdf/1409.1556.pdf)
- [документация PyTorch по сверточному слою](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [пример класса Trainer](https://github.com/pkhanzhina/mllib_f2023_mlp/blob/master/executors/mlp_trainer.py)
- [аугментации torchvision](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)
- [пример визуализации батча](https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py)
- [статья по ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [torchsummary](https://pypi.org/project/torch-summary/)
- [Bag of Tricks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
- [learning rate schedulers in PyTorch](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
