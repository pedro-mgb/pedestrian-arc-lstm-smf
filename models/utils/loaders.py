"""
Created on April 27th 2021.
Loaders abstraction for several kinds of models.
"""
import itertools
import os


def map_from_file_name_biwi_crowds(file_name, scene_list):
    if len(scene_list) < 2:
        return 0
    assert 'eth' in scene_list and 'hotel' in scene_list and 'univ' in scene_list and 'zara' in scene_list, \
        f'The provided scene list {scene_list} does not contain some of the actual scene labels for biwi/crowds data'
    name_lower = file_name.lower()
    idx = -1
    if 'eth' in name_lower:
        idx = scene_list.index('eth')
    elif 'hotel' in name_lower:
        idx = scene_list.index('hotel')
    elif 'zara' in name_lower:
        idx = scene_list.index('zara')
    elif 'univ' in name_lower or 'uni' in name_lower or 'student' in name_lower or 'crowds' in name_lower:
        idx = scene_list.index('univ')
    assert idx >= 0, f'The provided filename ({file_name}) is not present in biwi crowds list - {scene_list}'
    return idx


def get_models_per_scene_biwi_crowds(device, parent_folder_path, retrieve_model):
    """
    Get a list of models, for each specific scene, of biwi and crowds dataset. Should be 2 models per dataset, since
    each has two scenes - biwi has eth and hotel; crowds has univ and zara
    :param device: a torch.device to map the data to
    :param parent_folder_path: the parent folder path containing the several models
    :param retrieve_model: method object to retrieve the model from a file. Receives device and path to file.
    :return: list of models, and per-scene labels (in the same order)
    """
    # first two belong to biwi (ETH dataset), the other two to crowds (UCY dataset)
    scene_labels = ['eth', 'hotel', 'univ', 'zara']
    return get_models_per_scene(device, parent_folder_path, scene_labels, retrieve_model)


def get_models_per_scene(device, parent_folder_path, scene_labels, retrieve_model):
    """
    Get a list of models, for data belonging to several scenes
    This assumes that each model, trained on data solely belonging to one scene, is present on the folder.
    For example, if there are 4 scenes, there should be 4 (and only 4) files, each for a model trained on a scene.
    Each model should also contain the specific label identifying the scene to which it refers to.
    :param device: a torch.device to map the data to
    :param parent_folder_path: the parent folder path containing the several models
    :param scene_labels: list of labels for the several scenes
    :param retrieve_model: method object to retrieve the model from a file. Receives device and path to file.
    :return: list of models, and per-scene labels (in the same order)
    """
    file_list = sorted(os.listdir(parent_folder_path))  # sorted alphabetically
    files_and_scenes = __assert_scenes_and_models_files__(parent_folder_path, file_list, scene_labels)
    # checks performed successfully, can read the fields file
    fields_list = []
    for [file_name, scene_label] in files_and_scenes:
        full_path = os.path.join(parent_folder_path, file_name)
        fields_content = retrieve_model(device, full_path)
        fields_list.append([fields_content, scene_label])
    return fields_list


def __assert_scenes_and_models_files__(parent_folder_path, file_list, scene_labels):
    """
    assert if there is one and just one model for each of the intended scenes
    :param parent_folder_path: the parent folder path containing the several models
    :param file_list: list of files in parent_folder_path
    :param scene_labels: list of models, and per-scene labels (in the same order)
    :return: will return outside of this method if there is one model file per scene
    """
    assert len(file_list) == len(scene_labels), f"There are {len(file_list)} models present in {parent_folder_path}, " \
                                                f"but there are {len(scene_labels)} different scenes. {os.linesep} " \
                                                f"The numbers must be equal!"
    # see if each file has one of the labels, and there are no repeated labels
    covered_scenes, repeated_scenes = [], []
    labels_found, repeated_labels = 0, 0
    files_and_scenes = []
    for [_label, _file_name] in itertools.product(scene_labels, file_list):
        label = _label.lower()
        file_name = _file_name.lower()
        if label in file_name:
            if label in covered_scenes:
                repeated_labels += 1
                if label not in repeated_scenes:
                    repeated_scenes.append(label)
            else:
                labels_found += 1
                covered_scenes.append(label)
                files_and_scenes.append([file_name, label])
    assert labels_found == len(scene_labels), f"Expected model files to have labels {scene_labels}, but only found " \
                                              f"files with labels {covered_scenes}"
    assert repeated_labels == 0, f"Found model files with the following labels appearing more than once " \
                                 f"{repeated_scenes}. Make sure there is one label (from this list: {scene_labels})" \
                                 f"for each file."
    return files_and_scenes
