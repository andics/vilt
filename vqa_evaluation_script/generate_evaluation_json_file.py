import json
import re
import torch
import argparse

from typing import Optional
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader

class VQAv2ModelEvaluator:
    '''
    The following class can be used for generating a .json file of the VQA structure.
    '''
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for complete resolution-bin evaluation pipeline')
        parser.add_argument('-qf', '--question-file', nargs='?',
                            type=str,
                            default = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Datasets/vqa_var_10d/v2_OpenEnded_mscoco_val2014_questions.json",
                            required = False,
                            help='The path to the file which contains the questions (json). Either test or val')
        parser.add_argument('-if', '--images-folder', nargs='?',
                            type=str,
                            default = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Datasets/vqa_var_10d/val2014",
                            required = False,
                            help='A path to the folder containing the images for the particular question file')
        parser.add_argument('-rsp', '--results-save-path', nargs='?',
                            type=str,
                            default = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/hugging_fc/eval_results/var_results.json",
                            required = False,
                            help='Path to the location in which to save the results .json file, in VQA format')

        self.args = parser.parse_args()

        self.question_json_file_path = self.args.question_file
        self.test_images_folder = self.args.images_folder
        self.results_json_file_path = self.args.results_save_path


    def read_questions(self):
        self.question_json_content = json.load(open(self.question_json_file_path))
        self.questions = self.question_json_content['questions']


    def read_images(self):
        file_names = [f for f in tqdm(listdir(self.test_images_folder)) if isfile(join(self.test_images_folder, f))]
        filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")
        def id_from_filename(filename: str) -> Optional[int]:
            match = filename_re.fullmatch(filename)
            if match is None:
                return None
            return int(match.group(1))

        filename_to_id = {self.test_images_folder + "/" + file: id_from_filename(file) for file in file_names}
        self.img_id_to_filepath = {v: k for k, v in filename_to_id.items()}

    def load_model(self):
        self.model_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", device_map="cuda")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", device_map="cuda")

    def infer_images(self):
        current_image_id = None
        current_image_pil = None
        self.results = []

        for question in tqdm(self.questions):
            img_id = question['image_id']
            question_text = question['question']
            question_id = question['question_id']

            if not current_image_id == img_id:
                current_image_pil = Image.open(self.img_id_to_filepath[img_id])
                if current_image_pil.mode != "RGB":
                    current_image_pil = current_image_pil.convert("RGB")

            encoding = self.model_processor(current_image_pil, question_text, return_tensors="pt")
            encoding.to("cuda")

            outputs = self.model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = self.model.config.id2label[idx]
            self.results.append(dict({"answer": answer, "question_id": question_id}))

    def write_results(self):
        with open(self.results_json_file_path, "w") as json_file:
            json.dump(self.results, json_file)


if __name__ == "__main__":
    evaluator = VQAv2ModelEvaluator()
    evaluator.load_model()
    evaluator.read_questions()
    evaluator.read_images()
    evaluator.infer_images()
    evaluator.write_results()