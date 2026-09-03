from time import time
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, num_threads_for_brats_example
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


def get_list_of_patients(preprocessed_data_folder):
    npy_files = subfiles(preprocessed_data_folder, suffix=".npy", join=True)
    patients = [i[:-4] for i in npy_files]
    return patients


class BraTS2017DataLoader3D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)

        patch_size is the spatial size the retured batch will have

        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.num_modalities = 4
        self.indices = list(range(len(data)))

    @staticmethod
    def load_patient(patient):
        data = np.load(patient + ".npy", mmap_mode="r")
        metadata = load_pickle(patient + ".pkl")
        return data, metadata

    def generate_train_batch(self):
        # DataLoader selects the next patients for the batch.
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # Preallocate batch arrays.
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        metadata = []
        patient_names = []

        # Fill the batch sample by sample.
        for i, j in enumerate(patients_for_batch):
            patient_data, patient_metadata = self.load_patient(j)

            # Pad only when the case is smaller than patch_size.
            patient_data = pad_nd_image(patient_data, self.patch_size)

            # crop expects (b, c, x, y, z), so add a batch axis.
            patient_data, patient_seg = crop(patient_data[:-1][None], patient_data[-1:][None], self.patch_size, crop_type="random")

            data[i] = patient_data[0]
            seg[i] = patient_seg[0]

            metadata.append(patient_metadata)
            patient_names.append(j)

        return {'data': data, 'seg':seg, 'metadata':metadata, 'names':patient_names}


def get_train_transform(patch_size):
    tr_transforms = []

    # SpatialTransform crops to patch_size before later transforms run.
    # With three 0.1 probabilities, about 27% of samples are augmented.
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # Gamma applies a nonlinear intensity transform.
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # This variant inverts first, then restores the sign.
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # Blur augments already blurry modalities.
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))

    tr_transforms = Compose(tr_transforms)
    return tr_transforms


if __name__ == "__main__":
    patients = get_list_of_patients(brats_preprocessed_folder)

    train, val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)

    patch_size = (128, 128, 128)
    batch_size = 2

    # Patch training has no clear epoch size, so this example keeps setup simple.
    dataloader = BraTS2017DataLoader3D(train, batch_size, patch_size, 1)

    batch = next(dataloader)
    try:
        from batchviewer import view_batch
        # Show one sample because batchviewer supports up to 4D tensors.
        view_batch(batch['data'][0], batch['seg'][0])
    except ImportError:
        view_batch = None
        print("you can visualize batches with batchviewer. It's a nice and handy tool. You can get it here: "
              "https://github.com/FabianIsensee/BatchViewer")

    # Collect shapes before we build the training loader.
    shapes = [BraTS2017DataLoader3D.load_patient(i)[0].shape[1:] for i in patients]
    max_shape = np.max(shapes, 0)
    max_shape = np.max((max_shape, patch_size), 0)

    # Use max_shape so SpatialTransform handles crop and pad later.
    # This keeps full brains in view and avoids border artifacts here.
    dataloader_train = BraTS2017DataLoader3D(train, batch_size, max_shape, num_threads_for_brats_example)

    # This validation is patch-based and serves only as a quick progress check.
    dataloader_validation = BraTS2017DataLoader3D(val, batch_size, patch_size, max(1, num_threads_for_brats_example // 2))

    tr_transforms = get_train_transform(patch_size)

    # Keep pin_memory disabled because this example is framework-agnostic.
    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=num_threads_for_brats_example,
                                    num_cached_per_queue=3,
                                    seeds=None, pin_memory=False)
    # Validation needs fewer workers because it does not apply transforms.
    val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                     num_processes=max(1, num_threads_for_brats_example // 2), num_cached_per_queue=1,
                                     seeds=None,
                                     pin_memory=False)

    # Start workers early so batch generation overlaps main-thread work.
    tr_gen.restart()
    val_gen.restart()

    # Both generators are infinite, so iterate by batch count.
    num_batches_per_epoch = 10
    num_validation_batches_per_epoch = 3
    num_epochs = 5
    time_per_epoch = []
    start = time()
    for epoch in range(num_epochs):
        start_epoch = time()
        for b in range(num_batches_per_epoch):
            batch = next(tr_gen)

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)
        end_epoch = time()
        time_per_epoch.append(end_epoch - start_epoch)
    end = time()
    total_time = end - start
    print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
          (num_epochs, total_time, str(time_per_epoch)))

    # Reduce SpatialTransform probability first if CPU becomes the bottleneck.
    # Uncomment this block after you install batchviewer.
    if view_batch is not None:
        for _ in range(4):
            batch = next(tr_gen)
            view_batch(batch['data'][0], batch['seg'][0])
    else:
        print("Cannot visualize batches, install batchviewer first. It's a nice and handy tool. You can get it here: "
              "https://github.com/FabianIsensee/BatchViewer")
