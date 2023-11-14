import re
import json
import torch
import argparse
import logging
import os

from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from transformers import ViltForQuestionAnswering, ViltProcessor, ViltConfig
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from vilt_finetuning_script.objects.vqa_dataset_obj import VQADataset
from vilt_finetuning_script.objects.logger_obj import loggerObj
from vilt_finetuning_script.utils.util_functions import Utilities_helper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question-path', type=str, help='Path to the question json file')
    parser.add_argument('--annotation-path', type=str, help='Path to the annotation json file')
    parser.add_argument('--image-root', type=str, help='Root path to the image files')
    parser.add_argument('--pretrained-model', type=str, default="dandelin/vilt-b32-mlm",
                        help='Pretrained model to be used')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs for training')

    parser.add_argument('--model-save-dir', type=str, help='Path to where the model will be saved')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils_helper = Utilities_helper()

    logger_ref = loggerObj(logs_subdir=args.model_save_dir,
                           log_file_name="log",
                           utils_helper=utils_helper,
                           log_level=logging.DEBUG)
    logger_ref.setup_logger()
    logging.info("Successfully setup logger...")
    logging.info("Passed arguments -->>")
    logging.info('\n  -  ' + '\n  -  '.join(f'{k}={v}' for k, v in vars(args).items()))

    vqa = VQA(args.question_path, args.annotation_path, args.image_root,
              args.pretrained_model, args.batch_size, args.lr, args.num_epochs, device, args.model_save_dir)
    questions, annotations, filename_to_id, id_to_filename = vqa.load_data()
    dataset, config, processor = vqa.prepare_data(questions, annotations, filename_to_id, id_to_filename)
    vqa.train(dataset, config, processor)


class VQA:
    def __init__(self, question_path, annotation_path, image_root,
                 pretrained_model, batch_size, lr, num_epochs, device, model_save_dir):
        self.question_path = question_path
        self.annotation_path = annotation_path
        self.image_root = image_root
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.model_save_dir = model_save_dir

    def load_data(self):
        logging.info("Reading data...")

        #---Create necessary file-name maps---
        filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")
        def id_from_filename(filename: str) -> Optional[int]:
            match = filename_re.fullmatch(filename)
            if match is None:
                return None
            return int(match.group(1))
        file_names = [f for f in tqdm(listdir(self.image_root)) if isfile(join(self.image_root, f))]
        filename_to_id = {self.image_root + "/" + file: id_from_filename(file) for file in file_names}
        id_to_filename = {v: k for k, v in filename_to_id.items()}
        #-------------------------------------

        with open(self.question_path) as file:
            data_questions = json.load(file)

        with open(self.annotation_path) as file:
            data_annotations = json.load(file)

        return data_questions['questions'], data_annotations['annotations'], filename_to_id, id_to_filename

    def prepare_data(self, questions, annotations, filename_to_id, id_to_filename):
        config = ViltConfig.from_pretrained(self.pretrained_model)
        processor = ViltProcessor.from_pretrained(self.pretrained_model)

        def get_score(count):
            return min(1.0, count / 3)

        for annotation in tqdm(annotations):
            answers = annotation['answers']
            answer_count = {}
            for answer in answers:
                answer_ = answer["answer"]
                answer_count[answer_] = answer_count.get(answer_, 0) + 1
            labels = []
            scores = []
            for answer in answer_count:
                if answer not in list(config.label2id.keys()):
                    continue
                labels.append(config.label2id[answer])
                score = get_score(answer_count[answer])
                scores.append(score)
            annotation['labels'] = labels
            annotation['scores'] = scores

        return VQADataset(questions, annotations, processor, config, filename_to_id, id_to_filename),\
               config, processor

    def train(self, dataset, config, model_save_dir):
        model = ViltForQuestionAnswering.from_pretrained(self.pretrained_model,
                                                         id2label=config.id2label,
                                                         label2id=config.label2id)

        # Use all available GPUs, if we are on a CUDA machine, otherwise use CPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

        model = model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        train_dataloader = DataLoader(dataset, collate_fn=self.collate_fn,
                                      batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            print(f"Epoch: {epoch}")
            for i, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                print("Loss:", loss.item())
                loss.backward()
                optimizer.step()

                # Save the model every 2 iterations
                if i % 2 == 0:
                    torch.save(model.state_dict(),
                               os.path.join(model_save_dir, f'model_{epoch}_{i}.pth'))

    @staticmethod
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        tokenizer = batch[0]["processor"].tokenizer

        batch_encoding = tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask}, return_tensors="pt")
        labels = torch.stack(labels, dim=0)

        return {"input_ids": batch_encoding["input_ids"],
                "attention_mask": batch_encoding["attention_mask"],
                "pixel_values": pixel_values,
                "labels": labels}

if __name__ == "__main__":
    main()
