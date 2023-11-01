import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor

class VQADataset(Dataset):
    def __init__(self, questions, annotations, processor, config,
                 filename_to_id, id_to_filename):
        self.questions = questions
        self.annotations = annotations
        self.processor = processor
        self.config = config
        self.id_to_filename = id_to_filename
        self.filename_to_id = filename_to_id

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        text = questions['question']
        image = Image.open(self.id_to_filename[annotation['image_id']])

        encoding = self.processor(image, text, padding='max_length', truncation=True, return_tensors='pt')
        labels = annotation['labels']
        scores = annotation['scores']
        targets = torch.zeros(len(self.config.id2label))
        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets
        return encoding