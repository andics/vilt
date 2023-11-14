import os
import json
import glob
import logging
import re
import yaml

import shutil

import matplotlib.pyplot as plt

from pathlib import Path

class Utilities_helper(object):

    def __init__(self):
        self.defined = True

    def return_annotation_ids_present(self, json_target, to_print=False):

        with open(json_target) as json_file:
            json_data = json.load(json_file)

        annotation_ids_left = [x["id"] for x in json_data["annotations"]]

        if to_print:
            print("Annotation IDs left: ", annotation_ids_left)

        return annotation_ids_left

    def generate_equispaced_numbers(self, a, b, n):
        # Calculate the spacing between the numbers
        spacing = (b - a) / (n - 1)
        # Generate the equispaced numbers
        equispaced_numbers = [a + i * spacing for i in range(n)]
        return equispaced_numbers

    def check_dir_and_make_if_na(self, path_to_check):
        if os.path.exists(path_to_check):
            return True
        else:
            Path(path_to_check).mkdir(parents=True, exist_ok=True)
            return False

    def extract_json_name_from_path(self, json_path):

        json_name = os.path.split(json_path)[-1]
        json_name, extension = os.path.splitext(json_name)
        return json_name, extension


    def flatten_nested_dict(self, nested_dict, parent_key='', sep='_'):
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self.flatten_nested_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


    def concatenate_lists(self, *lists):
        """
        Concatenates multiple lists into a single list.
        """
        result = []
        for lst in lists:
            result.extend(lst)
        return result


    def extract_path_base_from_full(self, target_path):
        return os.path.dirname(target_path)


    def gather_subfolders(self, target_path):
        #Returns a list of the paths of all subfolders within the target_path folder
        list_subfolders_with_paths = [f.path for f in os.scandir(target_path) if f.is_dir()]
        return list_subfolders_with_paths


    def gather_subfiles(self, target_path):
        #Returns a list of the names of files within a folder
        files_in_dir = os.listdir(target_path)
        only_files_in_dir = [file for file in files_in_dir if os.path.isfile(os.path.join(target_path, file))]

        return only_files_in_dir


    def split_list_to_chunks_w_boundries(self, lst, n):
        """Splits the list into chunks with equal lenght when possible and outputs two numbers for each chunk: start_element, end_element"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


    def gather_json_files_from_dir(self, path_to_scan):
        files_to_return = []
        for file in os.listdir(path_to_scan):
            if file.endswith(".json"):
                files_to_return += [os.path.join(path_to_scan, file)]

        return files_to_return

    def write_data_to_json(self, json_path, data):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def extract_floats_and_nums_from_string(self, string):
        p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

        nums_to_return = []
        if re.search(p, string) is not None:
            for catch in re.finditer(p, string):
                nums_to_return.append(catch[0])

        return nums_to_return


    def get_last_checkpoint_in_dir(self, checkpoint_dir, accepted_extensions = "*.pth"):
        save_file_dir = os.path.join(checkpoint_dir, "last_checkpoint")
        #MODIFICATION START
        try:
            last_saved = max(glob.iglob(save_file_dir + '/' + accepted_extensions), key=os.path.getctime)
            print("Found the following checkpoint file: {}".format(last_saved))
            if len(last_saved) < len(save_file_dir):
                #No checkpoint file found
                last_saved = ""
        except Exception as e:
            #No checkpoint file found
            last_saved = ""

        print("Last saved checkpoint file (if any): {}".format(last_saved))
        return last_saved


    def find_common_elements_between_n_lists(self, arr):
        # initialize result with first array as a set
        result = set(arr[0])

        # now iterate through list of arrays starting from
        # second array and take intersection_update() of
        # each array with result. Every operation will
        # update value of result with common values in
        # result set and intersected set
        for currSet in arr[1:]:
            result.intersection_update(currSet)

        return list(result)


    def generate_file_names_with_addition(self, original_file_paths, addition, new_ext):
        if not isinstance(original_file_paths, list):
            #Turn passed folder path to a list
            #This makes the bellow code uniformly applicable
            temp_ = []
            temp_ += original_file_paths
            original_file_paths = temp_

        base_dir_of_paths = os.path.dirname(original_file_paths[0])

        new_paths_with_add = []

        for element in original_file_paths:
            file_base_name = os.path.basename(element)
            filename, file_extension = os.path.splitext(file_base_name)
            file_base_name = filename + addition + new_ext

            new_path = os.path.join(base_dir_of_paths, file_base_name)
            new_paths_with_add.append(new_path)

        return new_paths_with_add


    def is_num_in_range(self, number, range_list):
        if range_list[0] < number <= range_list[1]:
            return True
        else:
            return False


    def split_list_into_chunks(self, a, n):
        k, m = divmod(len(a), n)
        return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


    def create_folder_if_none_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print("Created folder: {}".format(folder_path))

    def path_exists(self, path_to_check):
        return os.path.exists(path_to_check)


    def generate_new_save_paths(self, area_threshold, new_paths_base_dir, *argv):
        area_threshold = str(area_threshold)

        #Make folders for each coverage threshold
        new_paths_base_dir = os.path.join(new_paths_base_dir, area_threshold)
        if not os.path.exists(new_paths_base_dir):
            os.makedirs(new_paths_base_dir)

        new_save_paths = []

        for original_path in argv:
            file_base_name = os.path.basename(original_path)
            filename, file_extension = os.path.splitext(file_base_name)
            file_base_name = filename + "_filt" + file_extension

            new_save_path = os.path.join(new_paths_base_dir, file_base_name)

            new_save_paths.append(new_save_path)

        return new_save_paths


    def calculate_figure_size_inches(self, desired_width_in_pixels, desired_height_in_pixels, dpi):
        width_inches = desired_width_in_pixels / dpi
        height_inches = desired_height_in_pixels / dpi

        return width_inches, height_inches

    def extract_filename_and_ext(self, path):
        file_name_and_ext = os.path.splitext(os.path.basename(path))

        return file_name_and_ext[0], file_name_and_ext[1].lower()[1:]

    def read_yaml_data(self, yaml_path):
        with open(yaml_path) as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        return yaml_data

    def change_yaml_file_value(self, file_location, variable_to_change, new_value):
        # E.g usage:
        #   variable_to_change = ['SOLVER', 'CATCHUP_PHASE_TRIGGERED']
        #   new_value = False

        with open(file_location) as f:
            doc = yaml.safe_load(f)

        self.edit_from_access_pattern(variable_to_change, doc, new_value)

        with open(file_location, 'w') as f:
            yaml.safe_dump(doc, f, default_flow_style=False)


    def edit_from_access_pattern(self, access_pattern, nested_dict, new_value):
        if len(access_pattern) == 1:
            nested_dict[access_pattern[0]] = new_value
        else:
            return self.edit_from_access_pattern(access_pattern[1:], nested_dict[access_pattern[0]], new_value)


    def to_delete_or_not_to_delete_content(self, dir_to_operate_on, to_delete):
        if not to_delete:
            for filename in os.listdir(dir_to_operate_on):
                file_path = os.path.join(dir_to_operate_on, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            print("FOLDER {} CONTENT DELETED!".format(dir_to_operate_on))


    def save_info_to_file(self, file_path, info_to_write):
        with open(file_path, 'w') as f:
            for variable in info_to_write:
                f.write('%s \n' % str(variable))


    def print_and_log(self, message, logger_name):
        logger = logging.getLogger(logger_name)

        logger.info(message)
        print(message)


    def save_info_to_file_from_dict(self, file_path, info_to_write):
        f = open(file_path, "w")
        print("LENGTH: ", len(info_to_write))
        print("INFO TO WRITE: ", info_to_write)
        for variable in info_to_write:
            to_write = str(variable) + str(info_to_write[variable])
            print("TO WRITE: ", to_write)
            f.write('%s \n' % str(to_write))

        f.close()


    def copy_file_from_to(self, source_path, destination_path):
        '''A simple function which copies a file from one location to another'''
        shutil.copy(source_path, destination_path)

    def find_element_with_annotation_id(self, json_object, annotation_id):
        annotation_id = str(annotation_id)
        #Delete the first character from the annotation ID as it correspons to the channel
        #print("Length of original annotation_id: ", len(annotation_id), "\n")
        #Remove the first element from the annotation ID to make them comparable => 1000234 becomes 000234
        annotation_id = str(annotation_id[1:])
        #print("Length of new annotation_id: ", len(annotation_id), "\n")

        for element in json_object:
            if annotation_id in str(element['id']):
                element_copy = element
                return element_copy

    def display_multi_image_collage(self, images_info_bundle, plot_size):
        '''
        :param images_info_bundle: Expected to be a Tuple of Tuples, each sub-tuple containing the Title of the image [1]
        and the image itself [0] (in cv2 format)
        :param plot_size: contains number of rows [0] and number of columns [1]
        :return: None
        '''
        fig = plt.figure(figsize=(10, 7))

        # setting values to rows and column variables
        rows = plot_size[0]
        columns = plot_size[1]

        counter = 1

        for image_tuple in images_info_bundle:
            if not image_tuple == ():
                fig.add_subplot(rows, columns, counter)
                # showing image
                plt.imshow(image_tuple[0])
                plt.axis('off')
                plt.title(image_tuple[1])
                counter += 1

        plt.show()