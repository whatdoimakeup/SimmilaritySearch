import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoProcessor

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from PIL import Image

class ModelWrapper:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Инициализирует объект ModelWrapper, загружая модель и процессор.

        Параметры:
            model_name (str): Название предобученной модели для загрузки.
            device (str): Устройство, на котором будет работать модель ("cuda" или "cpu").
        """
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.__is_train = False
    
    def __call__(self, images, texts, do_rescale = False):
        """
        Обрабатывает изображения и тексты, возвращая их эмбеддинги.

        Параметры:
            images: Список или батч изображений.
            texts: Список или батч текстов.

        Возвращает:
            torch.Tensor: Тензор, содержащий результат работы.
        """
        # Обрабатываем изображения и тексты одновременно через процессор
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, do_rescale=do_rescale).to(self.device)
        
        # Получаем эмбеддинги из модели
        outputs = self.model(**inputs)
        
        return outputs

    def train(self):
        """
        Переводит модель в режим обучения.
        """
        self.model.train()
        self.__is_train = True

    def eval(self):
        """
        Переводит модель в режим оценки (inference).
        """
        self.model.eval()
        self.__is_train = False

    def freeze_layers(self, n: int):
        """
        Замораживает слои до указанного уровня.
        
        Параметры:
            n (int): Номер последнего слоя для заморозки (начиная с 0).
        """
        # Проверяем, что n находится в допустимом диапазоне
        total_layers = len(list(self.model.parameters()))
        if n >= total_layers:
            raise ValueError(f"Число слоев для заморозки превышает общее количество ({total_layers}) слоев.")
        
        # Замораживаем слои до n-го включительно
        for i, param in enumerate(self.model.parameters()):
            if i <= n:
                param.requires_grad = False
            else:
                break
        # print(f"Заморожены слои от 0 до {n}")

    def unfreeze_all_layers(self):
        """Размораживает все слои модели."""
        for param in self.model.parameters():
            param.requires_grad = True
        # print("Все слои разморожены")

    def encode_text(self, texts):
        """
        Кодирует текст и возвращает текстовые признаки (фичи).

        Параметры:
            texts (list[str]): Список текстов для кодирования.

        Возвращает:
            torch.Tensor: Тензор с текстовыми признаками, полученными из модели.
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        return self.model.get_text_features(**inputs)

    def encode_image(self, images):
        """
        Кодирует изображения и возвращает визуальные признаки (фичи).

        Параметры:
            images (list[Image] или torch.Tensor): Список изображений или тензор изображений для кодирования.

        Возвращает:
            torch.Tensor: Тензор с признаками изображений, полученными из модели.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs)

    def get_similarity(self, images, texts):
        """
        Вычисляет косинусное сходство между фичами изображений и текстов.

        Параметры:
            images (list[Image] или torch.Tensor): Список изображений или тензор изображений для кодирования.
            texts (list[str]): Список текстов для кодирования.

        Возвращает:
            torch.Tensor: Матрица сходства (размером `len(texts) x len(images)`), 
            где каждое значение представляет собой косинусное сходство между изображением и текстом.
        """
         
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = torch.matmul(text_features, image_features.T)
        return similarity
    

    def get_image_similarity(self, image_batch):
        """
        Вычисляет косинусное сходство между всеми изображениями в батче.

        Параметры:
            image_batch (list[Image] или torch.Tensor): Список изображений или тензор изображений для кодирования.

        Возвращает:
            torch.Tensor: Квадратная матрица сходства (размером `len(image_batch) x len(image_batch)`), 
            где каждое значение представляет собой косинусное сходство между парами изображений.
        """
        image_features = self.encode_image(image_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity_matrix = torch.matmul(image_features, image_features.T)
    
        return similarity_matrix
    
from sklearn.metrics import f1_score, accuracy_score
def calculate_metrics(logits, ground_truth):
    """
    Считает метрики: точность, f1.
    """
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1    

class Trainer:
    def __init__(self, model_wrapper, train_dataloader, val_dataloader, optimizer, scheduler = None):
        """
        Класс Trainer отвечает за управление процессом обучения модели, в том числе за обучение, валидацию и 
        настройку параметров обучения с использованием оптимизатора и планировщика.

        Атрибуты:
            model_wrapper: Оболочка модели, включающая модель, оптимизацию и метрики.
            train_dataloader: DataLoader для обучения, предоставляющий данные для тренировки.
            val_dataloader: DataLoader для валидации, предоставляющий данные для оценки.
            optimizer: Оптимизатор, используемый для обновления весов модели.
            scheduler: Планировщик, управляющий изменением скорости обучения (необязательный параметр).
            loss_fn: Функция потерь, используемая для обучения (по умолчанию ContrastiveLoss).
            train_losses: Список для хранения значений потерь на обучающем наборе на протяжении обучения.
            val_losses: Список для хранения значений потерь на валидационном наборе на протяжении обучения.
            f1_train: Список значений F1-метрики для каждого этапа обучения.
            acc_train: Список значений точности для каждого этапа обучения.
            f1_val: Список значений F1-метрики для каждого этапа валидации.
            acc_val: Список значений точности для каждого этапа валидации.
        """
        self.model_wrapper = model_wrapper
        self.model_wrapper.model.to(torch.float32)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = ContrastiveLoss()
        self.train_losses = []
        self.val_losses = []
        self.f1_train = []
        self.acc_train = []
        self.f1_val = []
        self.acc_val = []

    def train_epoch(self):
        """
        Выполняет один полный цикл обучения модели на тренировочном наборе данных.

        Returns:
            avg_loss (float): Среднее значение функции потерь по всем батчам за текущую эпоху.
        """
        self.model_wrapper.train()
        epoch_loss = 0
        epoch_f1 = 0
        epoch_acc = 0
        for batch_images, batch_labels in tqdm(self.train_dataloader, desc="Training"):

            output = self.model_wrapper(batch_images, batch_labels)
            logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text
            # Получаем фичи изображений и текстовые фичи классов
            current_batch_size = batch_images.size(0)
            ground_truth = torch.arange(current_batch_size, device=self.model_wrapper.device)
            acc, f1 = calculate_metrics(logits_per_image, ground_truth)
            epoch_acc += acc
            epoch_f1 += f1

            loss = self.loss_fn(logits_per_image, logits_per_text)
            epoch_loss += loss.item()

            # Обновляем параметры
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        avg_loss = epoch_loss / len(self.train_dataloader)
        avg_f1 = epoch_f1 / len(self.train_dataloader)
        avg_acc = epoch_acc / len(self.train_dataloader)
        self.train_losses.append(avg_loss)
        self.f1_train.append(avg_f1)
        self.acc_train.append(avg_acc)
        return avg_loss

    def val_epoch(self):
        """
        Выполняет один полный цикл валидации модели на валидационном наборе данных.

        Returns:
            avg_loss (float): Среднее значение функции потерь по всем батчам на валидационном наборе за текущую эпоху.
        """
        self.model_wrapper.eval()
        epoch_loss = 0
        epoch_f1 = 0
        epoch_acc = 0
        with torch.no_grad():
            for batch_images, batch_labels in tqdm(self.val_dataloader, desc="Validating"):
                
                # Получаем фичи изображений и текстовые фичи классов
                output = self.model_wrapper(batch_images, batch_labels)
                logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text

                current_batch_size = batch_images.size(0)
                ground_truth = torch.arange(current_batch_size, device=self.model_wrapper.device)
                acc, f1 = calculate_metrics(logits_per_image, ground_truth)
                epoch_acc += acc
                epoch_f1 += f1

                loss = self.loss_fn(logits_per_image, logits_per_text)
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.val_dataloader)
        avg_f1 = epoch_f1 / len(self.val_dataloader)
        avg_acc = epoch_acc / len(self.val_dataloader)
        self.val_losses.append(avg_loss)
        self.f1_val.append(avg_f1)
        self.acc_val.append(avg_acc)
        return avg_loss

    def fit(self, epochs):
        """
        Запускает процесс обучения и валидации модели на протяжении заданного количества эпох.

        Параметры:
            epochs (int): Количество эпох, на протяжении которых будет производиться обучение модели.

        Описание:
            - Выполняет обучение и валидацию на каждой эпохе.
            - Сохраняет потери и метрики для каждой эпохи.
            - Опционально использует планировщик (scheduler) для корректировки learning rate после каждой эпохи.
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
                model_path = f"./clip_ft_{epoch+1}"
                self.model_wrapper.model.save_pretrained(model_path)
                self.model_wrapper.processor.save_pretrained(model_path)

    def plot_losses(self):
        """ 
        Отрисовка графиков функций потерь.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss over Epochs")
        plt.show()

    def plot_f1(self):
        """ 
        Отрисовка графиков метрики f1.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.f1_train, label="Train f1")
        plt.plot(self.f1_val, label="Validation f1")
        plt.xlabel("Epochs")
        plt.ylabel("F1")
        plt.legend()
        plt.title("Training and Validation f1 over Epochs")
        plt.show()

    def plot_acc(self):
        """ 
        Отрисовка графиков точности.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.acc_train, label="Train accuracy")
        plt.plot(self.acc_val, label="Validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation accuracy over Epochs")
        plt.show()


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, smoothing=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.smoothing = smoothing

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Apply label smoothing
        N = logits.size(0)
        smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
        smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate loss manually using log-softmax and smoothed labels
        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs).sum(dim=1).mean()

        return (loss_img + loss_txt) / 2
    

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, is_train = True):
        """
        Инициализирует датасет с изображениями и метками, основанными на названии папки.

        Параметры:
            image_paths (list[str]): Список путей до изображений.
            transform (callable, optional): Преобразования для применения к изображениям.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.labels = [self._get_label_from_path(path) for path in image_paths]
        self._is_train = is_train

    def _get_label_from_path(self, path):
        """
        Извлекает метку на основе названия папки в пути.

        Параметры:
            path (str): Путь до изображения.

        Возвращает:
            str: Метка, извлеченная из названия папки.
        """
        return os.path.basename(os.path.dirname(path))

    def __len__(self):
        """
        Возвращает количество изображений в датасете.

        Возвращает:
            int: Количество изображений.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Загружает и возвращает изображение и его метку.

        Параметры:
            idx (int): Индекс изображения.

        Возвращает:
            tuple: Кортеж, содержащий изображение (после преобразования) и метку.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        text = f"Image of {label}"
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        if self._is_train:
            return image, image_path
        return image, text