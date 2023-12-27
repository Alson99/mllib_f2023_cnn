# TODO: Реализуйте класс для обучения моделей, минимальный набор функций:
#  1. Подготовка обучающих и тестовых данных
#  2. Подготовка модели, оптимайзера, целевой функции
#  3. Обучение модели на обучающих данных
#  4. Эвалюэйшен модели на тестовых данных, для оценки точности можно рассмотреть accuracy, balanced accuracy
#  5. Сохранение и загрузка весов модели
#  6. Добавить возможность обучать на gpu
#  За основу данного класса можно взять https://github.com/pkhanzhina/mllib_f2023_mlp/blob/master/executors/mlp_trainer.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.oxford_pet_dataset import OxfordIIITPet
from logs.Logger import Logger
from models.vgg16 import VGG16
from models.resnet50 import ResNet50
from utils.metrics import accuracy, balanced_accuracy
from utils.visualization import show_batch
from utils.utils import set_seed


class Trainer:
    def __init__(self, cfg):
        self.test_loader = None
        self.train_loader = None
        set_seed(cfg.seed)

        self.cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')
        self.logger = Logger(log_dir=cfg.log_dir)

        self.model = None  # Initialize your model (VGG16, ResNet50, etc.)
        self.model.to(self.device)
        self.optimizer = None  # Initialize your optimizer (e.g., Adam)
        self.criterion = None  # Initialize your loss function (e.g., CrossEntropyLoss)

    def prepare_data(self):
        train_dataset = OxfordIIITPet(self.cfg, dataset_type='train', transform=None)
        test_dataset = OxfordIIITPet(self.cfg, dataset_type='test', transform=None)

        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

            test_accuracy, test_balanced_accuracy = self.evaluate()

    def evaluate(self):

        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

        # Calculate and return accuracy and balanced accuracy
        return accuracy, balanced_accuracy

    def save_weights(self, save_path):
        # Save the model weights to the specified path
        torch.save(self.model.state_dict(), save_path)

    def load_weights(self, load_path):
        # Load the model weights from the specified path
        self.model.load_state_dict(torch.load(load_path))

