import numpy as np
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, \
    brats_folder_with_downloaded_train_data, num_threads_for_brats_example
from batchgenerators.utilities.file_and_folder_operations import *

try:
    import SimpleITK as sitk
except ImportError:
    print("You need to have SimpleITK installed to run this example!")
    raise ImportError("SimpleITK not found")

from multiprocessing import Pool


def get_list_of_files(base_dir):
    """
    returns a list of lists containing the filenames. The outer list contains all training examples. Each entry in the
    outer list is again a list pointing to the files of that training example in the following order:
    T1, T1c, T2, FLAIR, segmentation
    :param base_dir:
    :return:
    """
    list_of_lists = []
    for glioma_type in ['HGG', 'LGG']:
        current_directory = join(base_dir, glioma_type)
        patients = subfolders(current_directory, join=False)
        for p in patients:
            patient_directory = join(current_directory, p)
            t1_file = join(patient_directory, p + "_t1.nii.gz")
            t1c_file = join(patient_directory, p + "_t1ce.nii.gz")
            t2_file = join(patient_directory, p + "_t2.nii.gz")
            flair_file = join(patient_directory, p + "_flair.nii.gz")
            seg_file = join(patient_directory, p + "_seg.nii.gz")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
            assert all((isfile(i) for i in this_case)), "some file is missing for patient %s; make sure the following " \
                                                        "files are there: %s" % (p, str(this_case))
            list_of_lists.append(this_case)
    print("Found %d patients" % len(list_of_lists))
    return list_of_lists


def load_and_preprocess(case, patient_name, output_folder):
    """
    loads, preprocesses and saves a case
    This is what happens here:
    1) load all images and stack them to a 4d array
    2) crop to nonzero region, this removes unnecessary zero-valued regions and reduces computation time
    3) normalize the nonzero region with its mean and standard deviation
    4) save 4d tensor as numpy array. Also save metadata required to create niftis again (required for export
    of predictions)

    :param case:
    :param patient_name:
    :return:
    """
    # Load images and arrays.
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    spacing = imgs_sitk[0].GetSpacing()
    # SimpleITK spacing is reversed relative to the NumPy axis order.
    spacing = np.array(spacing)[::-1]

    direction = imgs_sitk[0].GetDirection()
    origin = imgs_sitk[0].GetOrigin()

    original_shape = imgs_npy[0].shape

    # Cast to float32 before normalization math.
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    # Find the joint nonzero bounds across all modalities.
    nonzero = [np.array(np.where(i != 0)) for i in imgs_npy]
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T

    # Crop to the nonzero box.
    imgs_npy = imgs_npy[:,
               nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1,
               ]

    # Use nonzero image voxels as the brain mask.
    nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]

    # Normalize each image channel inside the brain mask.
    for i in range(len(imgs_npy) - 1):
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0

    # BraTS uses label 4; map it to 3 for compact labels.
    imgs_npy[-1][imgs_npy[-1] == 4] = 3

    # Save the preprocessed tensor and metadata.
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)

    metadata = {
        'spacing': spacing,
        'direction': direction,
        'origin': origin,
        'original_shape': original_shape,
        'nonzero_region': nonzero
    }

    save_pickle(metadata, join(output_folder, patient_name + ".pkl"))


def save_segmentation_as_nifti(segmentation, metadata, output_file):
    original_shape = metadata['original_shape']
    seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
    nonzero = metadata['nonzero_region']
    seg_original_shape[nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1] = segmentation
    sitk_image = sitk.GetImageFromArray(seg_original_shape)
    sitk_image.SetDirection(metadata['direction'])
    sitk_image.SetOrigin(metadata['origin'])
    # Convert spacing back to SimpleITK axis order.
    sitk_image.SetSpacing(tuple(metadata['spacing'][[2, 1, 0]]))
    sitk.WriteImage(sitk_image, output_file)


if __name__ == "__main__":
    # Save to npy so training can mmap slices from fast local storage.
    # This keeps large patch-based pipelines from stalling on I/O.

    list_of_lists = get_list_of_files(brats_folder_with_downloaded_train_data)

    maybe_mkdir_p(brats_preprocessed_folder)

    patient_names = [i[0].split("/")[-2] for i in list_of_lists]

    p = Pool(processes=num_threads_for_brats_example)
    p.starmap(load_and_preprocess, zip(list_of_lists, patient_names, [brats_preprocessed_folder] * len(list_of_lists)))
    p.close()
    p.join()

    # Restore the cropped prediction to its original image space.
    img = np.load(join(brats_preprocessed_folder, "Brats17_2013_0_1.npy"))
    metadata = load_pickle(join(brats_preprocessed_folder, "Brats17_2013_0_1.pkl"))
    # Map label 3 back to the BraTS label 4 before export.
    img[-1][img[-1] == 3] = 4
    save_segmentation_as_nifti(img[-1], metadata, join(brats_preprocessed_folder, "delete_me.nii.gz"))
