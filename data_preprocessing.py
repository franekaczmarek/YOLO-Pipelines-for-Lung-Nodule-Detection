import os
import shutil
import random
import SimpleITK as sitk
import numpy as np
from PIL import Image, ImageOps
from lungmask import LMInferer
from scipy.ndimage import label as ndimage_label

# Define source folders
nifti_input_folder = '/home/u154686/Thesis/Data/nifty_data/images_nifty'
label_input_folder = '/home/u154686/Thesis/Data/nifty_data/labels_nifty'

# Define output folders for processed images and labels
train_output_folder_images_z = '/home/u154686/Thesis/Data/axis_z/train_data/images'
train_output_folder_labels_z = '/home/u154686/Thesis/Data/axis_z/train_data/labels'
val_output_folder_images_z = '/home/u154686/Thesis/Data/axis_z/val_data/images'
val_output_folder_labels_z = '/home/u154686/Thesis/Data/axis_z/val_data/labels'
test_output_folder_images_z = '/home/u154686/Thesis/Data/axis_z/test_data/images'
test_output_folder_labels_z = '/home/u154686/Thesis/Data/axis_z/test_data/labels'

# Function to clear all files from a folder
def clear_folder(folder):
    """
    Removes all files and subfolders inside a given folder.
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Clear and create destination folders
for folder in [train_output_folder_images_z, train_output_folder_labels_z,
               val_output_folder_images_z, val_output_folder_labels_z, test_output_folder_images_z,
               test_output_folder_labels_z]:
    clear_folder(folder)

# Function to split NIfTI files into train, val, test sets logically
def split_nifti_data(nifti_input_folder, train_ratio, val_ratio, test_ratio):
    """
    Splits the NIfTI files into training, validation, and test sets by patient IDs and returns lists of filenames.

    Args:
        nifti_input_folder: Source folder for original NIfTI images.
        train_ratio: Ratio of the dataset to use for training.
        val_ratio: Ratio of the dataset to use for validation.
        test_ratio: Ratio of the dataset to use for testing.

    Returns:
        A tuple of (train_files, val_files, test_files, train_patients, val_patients, test_patients).
    """
    # Validate that the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Read the files from the folder
    nifti_files = sorted([f for f in os.listdir(nifti_input_folder) if f.endswith(".nii.gz")])

    # Group files by patient
    patient_data = {}
    for nifti_file in nifti_files:
        # Extract patient ID from the filename (modify according to your filename format)
        patient_id = nifti_file.split('_')[2]  # Adjust index as per your naming convention
        if patient_id not in patient_data:
            patient_data[patient_id] = []
        patient_data[patient_id].append(nifti_file)

    # Shuffle patients for random split
    patients = list(patient_data.keys())
    random.shuffle(patients)

    # Compute the number of patients for each split
    num_patients = len(patients)
    train_size = int(round(num_patients * train_ratio))
    val_size = int(round(num_patients * val_ratio))
    test_size = num_patients - train_size - val_size  # Ensure all patients are assigned

    # Adjust if rounding error causes negative test_size
    if test_size < 0:
        test_size = 0
        val_size = num_patients - train_size

    # Split patients into sets
    train_patients = patients[:train_size]
    val_patients = patients[train_size:train_size + val_size]
    test_patients = patients[train_size + val_size:]

    # Collect filenames for each set
    train_files = []
    val_files = []
    test_files = []

    for patient_id in train_patients:
        train_files.extend(patient_data[patient_id])
    for patient_id in val_patients:
        val_files.extend(patient_data[patient_id])
    for patient_id in test_patients:
        test_files.extend(patient_data[patient_id])

    return train_files, val_files, test_files, train_patients, val_patients, test_patients

# Split the NIfTI data
train_files, val_files, test_files, train_patients, val_patients, test_patients = split_nifti_data(
    nifti_input_folder,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)

# Print the number of files in each set
print(f"Number of training files: {len(train_files)}")
print(f"Number of validation files: {len(val_files)}")
print(f"Number of test files: {len(test_files)}")

# Lung segmentation model setup
inferer = LMInferer(modelname='R231', modelpath='/home/u154686/.cache/torch/hub/checkpoints/unet_r231-d5d2fc3d.pth')

# Function to find bounding boxes in the given label slice (2D array) for YOLO
def find_bounding_boxes(label_slice, non_padded_bounds=None):
    """
    Finds bounding boxes in the given label slice (2D array) for YOLO, first enlarging them by 30% while keeping the center the same, then adding padding of 5 pixels. Bounding boxes are clipped to the non-padded area if provided.

    Args:
        label_slice: 2D numpy array of labels.
        non_padded_bounds: Tuple (x_min, x_max, y_min, y_max) defining the non-padded area boundaries.

    Returns:
        A list of bounding boxes in the format (x_min, x_max, y_min, y_max).
    """
    # Ensure label_slice is boolean
    label_slice = label_slice.astype(bool)
    bounding_boxes = []
    # Identify connected components (objects)
    labeled_slice, num_features = ndimage_label(label_slice)
    img_height, img_width = label_slice.shape
    # Loop through each detected object (connected component)
    for i in range(1, num_features + 1):
        object_indices = np.where(labeled_slice == i)
        # Get bounding box for this connected component
        y_min, y_max = np.min(object_indices[0]), np.max(object_indices[0])
        x_min, x_max = np.min(object_indices[1]), np.max(object_indices[1])

        # Calculate center of the bounding box
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0

        # Calculate width and height of the bounding box
        width = x_max - x_min
        height = y_max - y_min

        # Increase width and height by 30%
        new_width = width * 1.3
        new_height = height * 1.3

        # Calculate new x_min and x_max, y_min and y_max
        x_min_new = x_center - new_width / 2.0
        x_max_new = x_center + new_width / 2.0
        y_min_new = y_center - new_height / 2.0
        y_max_new = y_center + new_height / 2.0

        # Then expand bounding box by padding of 5 pixels
        x_min_new = x_min_new - 5
        x_max_new = x_max_new + 5
        y_min_new = y_min_new - 5
        y_max_new = y_max_new + 5

        # Ensure the new bounding box is within image boundaries
        x_min_new = max(int(np.round(x_min_new)), 0)
        x_max_new = min(int(np.round(x_max_new)), img_width - 1)
        y_min_new = max(int(np.round(y_min_new)), 0)
        y_max_new = min(int(np.round(y_max_new)), img_height - 1)

        # If non_padded_bounds are provided, clip the bounding boxes
        if non_padded_bounds is not None:
            np_x_min, np_x_max, np_y_min, np_y_max = non_padded_bounds
            x_min_new = max(x_min_new, np_x_min)
            x_max_new = min(x_max_new, np_x_max)
            y_min_new = max(y_min_new, np_y_min)
            y_max_new = min(y_max_new, np_y_max)

            # Ensure bounding box is still valid after clipping
            if x_min_new >= x_max_new or y_min_new >= y_max_new:
                continue  # Skip this bounding box as it's invalid after clipping

        bounding_boxes.append((x_min_new, x_max_new, y_min_new, y_max_new))
    return bounding_boxes

def merge_bounding_boxes(bounding_boxes, max_distance=10):
    """
    Merges bounding boxes that are overlapping or within max_distance pixels.

    Args:
        bounding_boxes: List of bounding boxes.
        max_distance: Maximum distance between bounding boxes to be merged.

    Returns:
        A list of merged bounding boxes.
    """
    merged_boxes = []
    while bounding_boxes:
        box = bounding_boxes.pop(0)
        merged = False
        for i, merged_box in enumerate(merged_boxes):
            if overlap(box, merged_box, max_distance):
                merged_boxes[i] = merge_boxes(box, merged_box)
                merged = True
                break
        if not merged:
            merged_boxes.append(box)
    return merged_boxes

def overlap(box1, box2, max_distance):
    """
    Checks if two bounding boxes overlap or are within max_distance pixels.

    Args:
        box1: First bounding box.
        box2: Second bounding box.
        max_distance: Maximum distance to consider for merging.

    Returns:
        True if boxes overlap or are within max_distance, False otherwise.
    """
    x1_min, x1_max, y1_min, y1_max = box1
    x2_min, x2_max, y2_min, y2_max = box2
    # Expand boxes by max_distance
    x1_min_expanded = x1_min - max_distance
    x1_max_expanded = x1_max + max_distance
    y1_min_expanded = y1_min - max_distance
    y1_max_expanded = y1_max + max_distance
    x2_min_expanded = x2_min - max_distance
    x2_max_expanded = x2_max + max_distance
    y2_min_expanded = y2_min - max_distance
    y2_max_expanded = y2_max + max_distance

    x_overlap = not (x1_max_expanded < x2_min or x2_max_expanded < x1_min)
    y_overlap = not (y1_max_expanded < y2_min or y2_max_expanded < y1_min)
    return x_overlap and y_overlap

def merge_boxes(box1, box2):
    """
    Merges two bounding boxes into one larger box that contains both.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        A merged bounding box.
    """
    x1_min, x1_max, y1_min, y1_max = box1
    x2_min, x2_max, y2_min, y2_max = box2
    x_min = min(x1_min, x2_min)
    x_max = max(x1_max, x2_max)
    y_min = min(y1_min, y2_min)
    y_max = max(y1_max, y2_max)
    return (x_min, x_max, y_min, y_max)

# Function to save images and labels with the appropriate naming convention
def save_image_and_label(image_slice, label_slice, base_filename, slice_index, folder_images, folder_labels, suffix='', class_id=0, max_distance=10, non_padded_bounds=None):
    image_filename = f"{base_filename}_slice_{slice_index}_axis_z{suffix}.png"
    label_filename = f"{base_filename}_slice_{slice_index}_axis_z{suffix}.txt"
    image_path = os.path.join(folder_images, image_filename)
    label_path = os.path.join(folder_labels, label_filename)

    # Save the image
    Image.fromarray(image_slice.astype(np.uint8)).save(image_path)

    # Find bounding boxes in label_slice
    bounding_boxes = find_bounding_boxes(label_slice, non_padded_bounds)

    # Merge bounding boxes that are close to each other
    merged_boxes = merge_bounding_boxes(bounding_boxes, max_distance)

    # Image dimensions
    img_height, img_width = image_slice.shape

    # Prepare YOLO formatted labels
    yolo_labels = []
    for box in merged_boxes:
        x_min, x_max, y_min, y_max = box

        # Calculate center x, y, width, and height in normalized format
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height

        # Append label in YOLO format: class_id, x_center, y_center, width, height
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Save labels to txt file
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_labels))

# Function to save images without labels (empty label files)
def save_normal_image(image_slice, base_filename, slice_index, folder_images, folder_labels, suffix=''):
    image_filename = f"{base_filename}_slice_{slice_index}_axis_z{suffix}.png"
    label_filename = f"{base_filename}_slice_{slice_index}_axis_z{suffix}.txt"
    image_path = os.path.join(folder_images, image_filename)
    label_path = os.path.join(folder_labels, label_filename)

    # Save the image
    Image.fromarray(image_slice.astype(np.uint8)).save(image_path)

    # Save an empty label file
    with open(label_path, 'w') as f:
        pass  # Write nothing to create an empty file

# Function to calculate the area of nodules in mm^2
def calculate_nodule_area(label_slice, pixel_spacing):
    """
    Calculates the area of nodules based on label slice and pixel spacing.

    Args:
        label_slice: 2D numpy array of labels.
        pixel_spacing: Tuple containing pixel spacing in mm (spacing_x, spacing_y).

    Returns:
        Total area of the nodules in mm^2.
    """
    nodule_pixels = np.sum(label_slice == 1)
    area_per_pixel = pixel_spacing[0] * pixel_spacing[1]  # Area of one pixel in mm^2
    total_area = nodule_pixels * area_per_pixel
    return total_area

# Step 1: Find maximum lung dimensions across all images
max_width = 0
max_height = 0

print("Step 1: Calculating maximum lung dimensions across all images...")

# Combine all files for dimension calculation
all_files = train_files + val_files + test_files

for filename in all_files:
    if filename.endswith('.nii.gz'):
        nifti_input_path = os.path.join(nifti_input_folder, filename)
        print(f"Analyzing file: {filename}")

        image = sitk.ReadImage(nifti_input_path)
        image_array = sitk.GetArrayFromImage(image)

        # Apply lung segmentation
        segmented_data = inferer.apply(image)
        lungs_mask = np.where((segmented_data == 1) | (segmented_data == 2), 1, 0)

        total_slices = image_array.shape[0]

        for slice_index in range(total_slices):
            lungs_mask_slice = lungs_mask[slice_index, :, :]

            # Check if lungs are present in the mask
            if not np.any(lungs_mask_slice):
                continue  # No lungs detected in this slice

            # Find bounding box of the lungs
            rows = np.any(lungs_mask_slice, axis=1)
            cols = np.any(lungs_mask_slice, axis=0)
            r_indices = np.where(rows)[0]
            c_indices = np.where(cols)[0]
            rmin, rmax = r_indices[0], r_indices[-1]
            cmin, cmax = c_indices[0], c_indices[-1]

            width = cmax - cmin + 1
            height = rmax - rmin + 1

            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height

print(f"Maximum lung width: {max_width}")
print(f"Maximum lung height: {max_height}")

# Desired output image size is set to the maximum detected lung size
output_image_size = (max_width, max_height)

# Step 2: Process images using the maximum lung dimensions
print("Step 2: Processing images with maximum lung dimensions...")

# Function to process a list of files
def process_files(file_list, dataset_name):
    if dataset_name == 'train':
        images_output_folder = train_output_folder_images_z
        labels_output_folder = train_output_folder_labels_z
        patient_list = train_patients
    elif dataset_name == 'val':
        images_output_folder = val_output_folder_images_z
        labels_output_folder = val_output_folder_labels_z
        patient_list = val_patients
    elif dataset_name == 'test':
        images_output_folder = test_output_folder_images_z
        labels_output_folder = test_output_folder_labels_z
        patient_list = test_patients
    else:
        print(f"Unknown dataset name: {dataset_name}")
        return

    patient_stats = {}

    for filename in file_list:
        if filename.endswith('.nii.gz'):
            nifti_input_path = os.path.join(nifti_input_folder, filename)
            label_filename = filename.replace('Images', 'Labels')
            label_input_path = os.path.join(label_input_folder, label_filename)

            # Extract patient ID from filename
            patient_id = filename.split('_')[2]  # Adjust index as per your naming convention

            if patient_id not in patient_stats:
                patient_stats[patient_id] = {
                    'no_lungs_count': 0,
                    'one_lung_count': 0,
                    'small_nodule_slices_removed': 0,
                    'total_nodules_count': 0,
                    'small_nodules_count': 0,
                    'slices_saved_count': 0,
                    'slices_with_nodules_saved_count': 0
                }

            print(f"Processing file: {filename}")

            image = sitk.ReadImage(nifti_input_path)
            label_image = sitk.ReadImage(label_input_path)

            image_array = sitk.GetArrayFromImage(image)
            label_array = sitk.GetArrayFromImage(label_image)

            spacing = image.GetSpacing()  # (spacing_x, spacing_y, spacing_z)
            pixel_spacing = (spacing[0], spacing[1])

            # Check if image_array and label_array have the same number of slices
            if image_array.shape[0] != label_array.shape[0]:
                print(f"Warning: image and label arrays have different number of slices for {filename}")
                print(f"image_array.shape[0]: {image_array.shape[0]}, label_array.shape[0]: {label_array.shape[0]}")
                # Skip this file
                print(f"Skipped file: {filename}")
                continue

            # Apply lung segmentation
            segmented_data = inferer.apply(image)
            lungs_mask = np.where((segmented_data == 1) | (segmented_data == 2), 1, 0)

            base_filename = filename.replace('.nii.gz', '')

            total_slices = image_array.shape[0]

            for slice_index in range(total_slices):
                image_slice = image_array[slice_index, :, :]
                label_slice = label_array[slice_index, :, :]
                lungs_mask_slice = lungs_mask[slice_index, :, :]

                # Check if lungs are present in the mask
                if not np.any(lungs_mask_slice):
                    # No lungs detected in this slice, skip it
                    patient_stats[patient_id]['no_lungs_count'] += 1
                    continue

                # Split the slice into left and right halves
                mid_point = lungs_mask_slice.shape[1] // 2
                left_half = lungs_mask_slice[:, :mid_point]
                right_half = lungs_mask_slice[:, mid_point:]

                # Check if there is lung tissue in both halves
                left_lung_present = np.any(left_half)
                right_lung_present = np.any(right_half)

                # If only one lung is present, skip the slice
                if not (left_lung_present and right_lung_present):
                    print(f"Only one lung detected in slice {slice_index}, skipping this slice.")
                    patient_stats[patient_id]['one_lung_count'] += 1
                    continue

                # Find bounding box of the lungs
                rows = np.any(lungs_mask_slice, axis=1)
                cols = np.any(lungs_mask_slice, axis=0)
                r_indices = np.where(rows)[0]
                c_indices = np.where(cols)[0]
                rmin, rmax = r_indices[0], r_indices[-1]
                cmin, cmax = c_indices[0], c_indices[-1]

                # Crop the image and label to this bounding box
                cropped_image_slice = image_slice[rmin:rmax+1, cmin:cmax+1]
                cropped_label_slice = label_slice[rmin:rmax+1, cmin:cmax+1]

                # Apply windowing
                window_center = -350
                window_width = 1000
                window_min = window_center - window_width / 2
                window_max = window_center + window_width / 2

                cropped_image_slice = np.clip(cropped_image_slice, window_min, window_max)

                # Normalize to 0-255
                normalized_slice = (cropped_image_slice - window_min) / (window_max - window_min) * 255
                normalized_slice = normalized_slice.astype(np.uint8)

                # Resize the cropped image to max_width x max_height while maintaining aspect ratio
                image_pil = Image.fromarray(normalized_slice)
                label_pil = Image.fromarray(cropped_label_slice.astype(np.uint8))

                # Determine the scaling factor while maintaining aspect ratio
                width_ratio = max_width / cropped_image_slice.shape[1]
                height_ratio = max_height / cropped_image_slice.shape[0]
                scaling_factor = min(width_ratio, height_ratio)

                # Resize the images
                new_width = int(cropped_image_slice.shape[1] * scaling_factor)
                new_height = int(cropped_image_slice.shape[0] * scaling_factor)
                resized_image = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_label = label_pil.resize((new_width, new_height), Image.Resampling.NEAREST)

                # Pad the resized images to max_width x max_height
                delta_w = max_width - new_width
                delta_h = max_height - new_height
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                padded_image = ImageOps.expand(resized_image, padding)
                padded_label = ImageOps.expand(resized_label, padding)

                # Record non-padded area coordinates
                pad_left, pad_top, pad_right, pad_bottom = padding
                non_padded_x_min = pad_left
                non_padded_x_max = pad_left + new_width - 1
                non_padded_y_min = pad_top
                non_padded_y_max = pad_top + new_height - 1
                non_padded_bounds = (non_padded_x_min, non_padded_x_max, non_padded_y_min, non_padded_y_max)

                # Convert padded images to arrays
                final_image_array = np.array(padded_image)
                final_label_array = np.array(padded_label)

                nodule_pixel_count = np.sum(final_label_array == 1)

                # Update total nodules count
                if nodule_pixel_count > 0:
                    patient_stats[patient_id]['total_nodules_count'] += 1

                # If in the training set and the slice has nodules, apply additional processing
                if dataset_name == 'train' and nodule_pixel_count > 0:
                    # Calculate nodule area
                    nodule_area = calculate_nodule_area(final_label_array, pixel_spacing)
                    if nodule_area < 25:
                        # Skip this slice
                        patient_stats[patient_id]['small_nodule_slices_removed'] += 1
                        patient_stats[patient_id]['small_nodules_count'] += 1
                        continue

                if nodule_pixel_count == 0:
                    # Save normal images (slices without nodules)
                    save_normal_image(final_image_array, base_filename, slice_index, images_output_folder, labels_output_folder)
                else:
                    # Save images and labels for slices with nodules
                    save_image_and_label(final_image_array, final_label_array, base_filename, slice_index, images_output_folder, labels_output_folder, suffix='', class_id=0, non_padded_bounds=non_padded_bounds)
                    patient_stats[patient_id]['slices_with_nodules_saved_count'] += 1

                # Update slices saved count
                patient_stats[patient_id]['slices_saved_count'] += 1

    # Print statistics for this dataset
    print(f"\nDataset: {dataset_name}")
    print(f"Patients: {patient_list}\n")

    for patient_id in patient_stats:
        stats = patient_stats[patient_id]
        if stats['total_nodules_count'] > 0:
            small_nodules_percentage = (stats['small_nodules_count'] / stats['total_nodules_count']) * 100
        else:
            small_nodules_percentage = 0.0

        print(f"Patient ID: {patient_id}")
        print(f"  Slices removed due to no lungs: {stats['no_lungs_count']}")
        print(f"  Slices removed due to only one lung: {stats['one_lung_count']}")
        if dataset_name == 'train':
            print(f"  Slices removed due to nodules smaller than 25mm²: {stats['small_nodule_slices_removed']} ({small_nodules_percentage:.2f}% of all nodules)")
        print(f"  Total slices saved: {stats['slices_saved_count']}")
        print(f"  Total slices with nodules saved: {stats['slices_with_nodules_saved_count']}\n")

# Set a random seed for reproducibility (optional)
random.seed(42)

# Process files for each dataset
process_files(train_files, 'train')
process_files(val_files, 'val')
process_files(test_files, 'test')
