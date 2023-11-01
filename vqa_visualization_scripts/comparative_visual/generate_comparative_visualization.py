import json
import re
import torch
import argparse
import os
import psutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from typing import Optional
from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageFont
from transformers import ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader

class VQAv2ModelEvaluator:
    '''
    The following class can be used for generating a set of visualization collages
    that display the differences in the answers that Var and Equiconst models give
    '''
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for complete resolution-bin evaluation pipeline')
        parser.add_argument('-qf', '--question-file', nargs='?',
                            type=str,
                            default = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Datasets/vqa_var_10d/v2_OpenEnded_mscoco_val2014_questions.json",
                            required = False,
                            help='The path to the file which contains the questions (json). Either test or val')
        parser.add_argument('-if', '--images-folders', nargs='?',
                            type=str,
                            default = ["/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Datasets/vqa_baseline_no_filt/val2014",
                                       "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Datasets/vqa_var_10d/val2014",
                                       "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Datasets/vqa_equiconst_10d/val2014"],
                            required = False,
                            help='A list of paths to the folders containing the images for all models to be displayed'
                                 'Note: this script assumes that the images in all the folders have the same names!')
        parser.add_argument('-ap', '--answers-paths', nargs='?',
                            type=str,
                            default = ["/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/hugging_fc/vqav2_eval_results/baseline_val_results.json",
                                       "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/hugging_fc/vqav2_eval_results/var_val_results.json",
                                       "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/hugging_fc/vqav2_eval_results/equiconst_val_results.json"],
                            required = False,
                            help='A list of paths to the folders containing the results .json files (VQA format)')
        parser.add_argument('-vsp', '--visualization-save-path', nargs='?',
                            type=str,
                            default = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Visualizations/vilt_comp_nofinetune",
                            required = False,
                            help='Path to the location in which to save the results .json files, in VQA format')

        self.args = parser.parse_args()

        self.question_json_file_path = self.args.question_file
        self.images_folders = self.args.images_folders
        self.answers_paths = self.args.answers_paths
        self.visualization_save_path = self.args.visualization_save_path


    def read_questions(self):
        self.question_json_content = json.load(open(self.question_json_file_path))
        self.questions = self.question_json_content['questions']

    def read_answers(self):
        self.answer_contents_dict = []
        for answer_file in self.answers_paths:
            answer_json = json.load(open(answer_file))
            self.answer_contents_dict.append({item['question_id']: item['answer'] for item in answer_json})

    def read_images(self):
        file_names = [f for f in tqdm(os.listdir(self.images_folders[0])) if os.path.isfile(os.path.join(self.images_folders[0], f))]
        filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")
        def id_from_filename(filename: str) -> Optional[int]:
            match = filename_re.fullmatch(filename)
            if match is None:
                return None
            return int(match.group(1))

        filename_to_id = {file: id_from_filename(file) for file in file_names}
        self.img_id_to_filename = {v: k for k, v in filename_to_id.items()}


    def find_answer_differences(self):
        '''
        The following function returns a list of question_id-s where the 1st and 2nd json_answer file have
        disagreeing answers. In this context, this corresponds to the Variable and the Equicosnt models disagreeing.
        :return: list
        '''
        dict1 = self.answer_contents_dict[1]
        dict2 = self.answer_contents_dict[2]

        # Create dictionary with question_id's as keys and 'yes' or 'no' depending on if the answers agree.
        self.ans_diff_dict = {question_id: 'yes' if dict1.get(question_id) == dict2.get(question_id) else 'no' for question_id in
                     dict1.keys()}


    def generate_visualizations(self):
        plt.ioff()
        process = psutil.Process(os.getpid())

        for question in tqdm(self.questions):
            img_id = question['image_id']
            question_text = question['question']
            question_id = question['question_id']

            if self.ans_diff_dict[question_id] == 'no':
                print(f"Answers for image {img_id}, question {question_id} disagree. Proceeding to visualize....")
                vis_save_path = os.path.join(self.visualization_save_path, f'img_{img_id}_question_{question_id}.png')
                if os.path.exists(vis_save_path):
                    print("Collage already exists. Skipping ...")
                    continue

                fig, axs = plt.subplots(1, 3, figsize=(20, 10))  # create 3 sub-plots
                fig.suptitle(question_text, fontsize=20)

                for i, (image_folder, answer_dict) in enumerate(zip(self.images_folders, self.answer_contents_dict)):
                    current_img_path = os.path.join(image_folder, self.img_id_to_filename[img_id])
                    img_array = mpimg.imread(current_img_path)
                    axs[i].imshow(img_array)
                    axs[i].set_title(answer_dict[question_id], fontsize=18)
                    axs[i].axis('off')  # turn off axis

                plt.tight_layout(rect=[0, 0, 1, 0.96])  # adjust the layout to leave space for the title
                plt.savefig(os.path.join(self.visualization_save_path, f'img_{img_id}_question_{question_id}.png'),
                            bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                plt.close()
                mem_info = process.memory_info()
                mem_in_MB = mem_info.rss / (1024 ** 2)
                print(f'Current memory usage: {mem_in_MB} MB')



if __name__ == "__main__":
    evaluator = VQAv2ModelEvaluator()
    evaluator.read_questions()
    evaluator.read_answers()
    evaluator.read_images()
    evaluator.find_answer_differences()
    evaluator.generate_visualizations()
