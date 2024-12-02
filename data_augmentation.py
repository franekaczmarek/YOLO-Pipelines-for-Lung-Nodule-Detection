import SimpleITK as sitk
import numpy as np
import os
import shutil
from PIL import Image, ImageOps
from lungmask import LMInferer
from scipy.ndimage import label as ndimage_label
import re
import cv2

# Define output folders for processed images and labels
train_images_folder = '/home/u154686/Thesis/Data/axis_z/train_data/images'
train_labels_folder = '/home/u154686/Thesis/Data/axis_z/train_data/labels'
nifti_input_folder = '/home/u154686/Thesis/Data/nifty_data/images_nifty'
label_input_folder = '/home/u154686/Thesis/Data/nifty_data/labels_nifty'
output_folder_images_z = '/home/u154686/Thesis/Data/axis_z/augmented_data/images'
output_folder_labels_z = '/home/u154686/Thesis/Data/axis_z/augmented_data/labels'

# Remove existing data in output folders
folders_to_clear = [
    output_folder_images_z,
    output_folder_labels_z
]

for folder in folders_to_clear:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Fixed lung size
max_width = 502
max_height = 405
output_image_size = (max_width, max_height)

# Lung segmentation model setup (still needed for identifying regions of interest)
inferer = LMInferer(modelname='R231', modelpath='/home/u154686/.cache/torch/hub/checkpoints/unet_r231-d5d2fc3d.pth')

# Function to find bounding boxes in the given label slice (2D array) for YOLO
def find_bounding_boxes(label_slice, non_padded_bounds=None):
    """
    Finds bounding boxes in the given label slice (2D array) for YOLO, first enlarging them by 30% while keeping the center the same, then adding padding of 5 pixels. Bounding boxes are clipped to the non-padded area if provided.
    """
    label_slice = label_slice.astype(bool)
    bounding_boxes = []
    labeled_slice, num_features = ndimage_label(label_slice)
    img_height, img_width = label_slice.shape
    for i in range(1, num_features + 1):
        object_indices = np.where(labeled_slice == i)
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

        # Then add padding of 5 pixels
        x_min_new -= 5
        x_max_new += 5
        y_min_new -= 5
        y_max_new += 5

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
    """Merges bounding boxes that are overlapping or within max_distance pixels."""
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
    """Checks if two bounding boxes overlap or are within max_distance pixels."""
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
    """Merges two bounding boxes into one larger box that contains both."""
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
    cv2.imwrite(image_path, image_slice.astype(np.uint8))

    # Find bounding boxes in label_slice
    bounding_boxes = find_bounding_boxes(label_slice, non_padded_bounds)

    # Merge overlapping bounding boxes
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

# Function to calculate the area of nodules in mm^2
def calculate_nodule_area(label_slice, pixel_spacing):
    nodule_pixels = np.sum(label_slice == 1)
    area_per_pixel = pixel_spacing[0] * pixel_spacing[1]
    total_area = nodule_pixels * area_per_pixel
    return total_area

# Function to extract patient number from filename
def get_patient_number(filename):
    match = re.search(r'LIDC_IDRI_(\d+)', filename)
    if match:
        return str(int(match.group(1)))  # Remove leading zeros
    else:
        return None

# Initialize total counts
total_empty_labels = 0
total_non_empty_labels = 0
total_images_generated = 0

# Get list of patients to process from train data folder
patient_numbers = set()
print("Collecting patient numbers from train data folder...")
for filename in os.listdir(train_images_folder):
    if filename.endswith('.png') or filename.endswith('.nii.gz') or filename.endswith('.jpg'):
        patient_number = get_patient_number(filename)
        if patient_number:
            patient_numbers.add(patient_number)
print(f"Patient numbers extracted from train data: {sorted(patient_numbers)}")

# Initialize dictionaries to count empty and non-empty labels per patient
patient_label_counts = {patient: {'empty': 0, 'non_empty': 0} for patient in patient_numbers}

# First pass: Count the number of empty and non-empty label files per patient
print("Counting empty and non-empty label files per patient...")
for filename in os.listdir(train_labels_folder):
    if filename.endswith('.txt'):
        patient_number = get_patient_number(filename)
        if patient_number not in patient_numbers:
            continue  # Skip patients not in the train_data folder

        label_path = os.path.join(train_labels_folder, filename)
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if content:
                # Non-empty label
                patient_label_counts[patient_number]['non_empty'] += 1
                total_non_empty_labels += 1
            else:
                # Empty label
                patient_label_counts[patient_number]['empty'] += 1
                total_empty_labels += 1

# Print the label counts per patient
print("Label counts per patient:")
for patient, counts in patient_label_counts.items():
    print(f"Patient {patient}: Empty labels = {counts['empty']}, Non-empty labels = {counts['non_empty']}")

# Start processing and augmenting images from NIfTI files
print("\nProcessing and augmenting images...")
for filename in os.listdir(nifti_input_folder):
    if filename.endswith('.nii.gz'):
        patient_number = get_patient_number(filename)
        if patient_number not in patient_numbers:
            print(f"Skipping file {filename}: Patient {patient_number} not in train data")
            continue  # Skip patients not in the train_data folder

        nifti_input_path = os.path.join(nifti_input_folder, filename)
        label_filename = filename.replace('Images', 'Labels')
        label_input_path = os.path.join(label_input_folder, label_filename)

        print(f"\nProcessing patient: {patient_number}")
        print(f"Image file: {nifti_input_path}")
        print(f"Label file: {label_input_path}")

        if not os.path.exists(label_input_path):
            print(f"Label file {label_input_path} does not exist. Skipping this patient.")
            continue

        try:
            image = sitk.ReadImage(nifti_input_path)
            label_image = sitk.ReadImage(label_input_path)
        except Exception as e:
            print(f"Error reading image or label for patient {patient_number}: {e}")
            continue

        image_array = sitk.GetArrayFromImage(image)
        label_array = sitk.GetArrayFromImage(label_image)
        pixel_spacing = image.GetSpacing()

        # Apply lung segmentation (still needed for ROI)
        print("Applying lung segmentation...")
        segmented_data = inferer.apply(image)
        lungs_mask = np.where((segmented_data == 1) | (segmented_data == 2), 1, 0)

        base_filename = filename.replace('.nii.gz', '')

        total_slices = image_array.shape[0]

        # Collect slices that meet the criteria
        valid_slices = []
        for slice_index in range(total_slices):
            image_slice = image_array[slice_index, :, :]
            label_slice = label_array[slice_index, :, :]
            lungs_mask_slice = lungs_mask[slice_index, :, :]

            # Skip slices without lungs
            if not np.any(lungs_mask_slice):
                continue

            # Skip slices without nodules or with area less than 25 mm^2
            nodule_area = calculate_nodule_area(label_slice, pixel_spacing)
            if nodule_area < 25:
                continue

            # Store necessary data for later processing
            valid_slices.append({
                'slice_index': slice_index,
                'image_slice': image_slice,
                'label_slice': label_slice,
                'lungs_mask_slice': lungs_mask_slice
            })

        if not valid_slices:
            print(f"No valid slices found for patient {patient_number}. Skipping.")
            continue

        # Determine how many images need to be generated
        counts = patient_label_counts[patient_number]
        images_needed = max(0, counts['empty'] - counts['non_empty'])
        print(f"Images needed for patient {patient_number}: {images_needed}")

        if images_needed == 0:
            print(f"No augmentation needed for patient {patient_number}.")
            continue

        augmented_images_generated = 0
        iteration = 1
        rotation_degree_increment = 19
        zoom_factor_increment = 0.15
        rotation_degree = rotation_degree_increment
        zoom_factor = 1.25
        max_zoom_factor = 2.0  # Maximum zoom factor

        # Continue augmenting until we reach the required number of images
        while augmented_images_generated < images_needed:
            print(f"\nIteration {iteration}: Rotation {rotation_degree} degrees, Zoom factor {zoom_factor}")
            for slice_info in valid_slices:
                if augmented_images_generated >= images_needed:
                    break

                slice_index = slice_info['slice_index']
                image_slice = slice_info['image_slice']
                label_slice = slice_info['label_slice']
                lungs_mask_slice = slice_info['lungs_mask_slice']

                # Crop the image and label to lungs bounding box
                rows = np.any(lungs_mask_slice, axis=1)
                cols = np.any(lungs_mask_slice, axis=0)
                r_indices = np.where(rows)[0]
                c_indices = np.where(cols)[0]
                rmin, rmax = r_indices[0], r_indices[-1]
                cmin, cmax = c_indices[0], c_indices[-1]

                cropped_image_slice = image_slice[rmin:rmax + 1, cmin:cmax + 1]
                cropped_label_slice = label_slice[rmin:rmax + 1, cmin:cmax + 1]

                # Get number of connected components in original label slice
                original_label_cc, original_num_features = ndimage_label(cropped_label_slice)

                # Apply windowing
                window_center = -350
                window_width = 1000
                window_min = window_center - window_width / 2
                window_max = window_center + window_width / 2

                cropped_image_slice = np.clip(cropped_image_slice, window_min, window_max)

                # Normalize to 0-255
                normalized_slice = (cropped_image_slice - window_min) / (window_max - window_min) * 255
                normalized_slice = normalized_slice.astype(np.uint8)

                # Convert to OpenCV format (grayscale images)
                image_cv = normalized_slice
                label_cv = cropped_label_slice.astype(np.uint8)

                # Rotate images using OpenCV
                angle = rotation_degree
                scale = 1.0  # We'll handle scaling separately
                h, w = image_cv.shape
                center = (w // 2, h // 2)

                # Get rotation matrix
                M = cv2.getRotationMatrix2D(center, angle, scale)

                # Compute new bounding dimensions
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))

                # Adjust rotation matrix to account for translation
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]

                # Rotate images
                rotated_image = cv2.warpAffine(image_cv, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
                rotated_label = cv2.warpAffine(label_cv, M, (new_w, new_h), flags=cv2.INTER_NEAREST)

                # Find center of nodules in the rotated label
                nodule_indices = np.where(rotated_label == 1)
                if len(nodule_indices[0]) == 0:
                    print(f"Slice {slice_index}: No nodules detected after rotation. Skipping this slice.")
                    continue  # No nodules detected after rotation

                # Get number of connected components in rotated label
                rotated_label_cc, rotated_num_features = ndimage_label(rotated_label)

                # If rotated label has more connected components than original, remove the smallest ones
                if rotated_num_features > original_num_features:
                    print(f"Slice {slice_index}: More connected components after rotation ({rotated_num_features}) than original ({original_num_features}). Removing smallest components.")
                    # Get sizes of connected components
                    component_sizes = np.bincount(rotated_label_cc.flatten())
                    # Exclude background (component 0)
                    sizes = component_sizes[1:]
                    labels = np.arange(1, rotated_num_features + 1)
                    # Sort labels by size
                    sorted_labels = labels[np.argsort(-sizes)]  # Descending order
                    # Keep only the largest 'original_num_features' components
                    labels_to_keep = sorted_labels[:original_num_features]
                    # Create a mask of components to keep
                    mask = np.isin(rotated_label_cc, labels_to_keep)
                    rotated_label = np.where(mask, rotated_label, 0).astype(np.uint8)

                # Recompute nodule indices after removing small components
                nodule_indices = np.where(rotated_label == 1)
                if len(nodule_indices[0]) == 0:
                    print(f"Slice {slice_index}: No nodules remain after removing small components. Skipping this slice.")
                    continue  # No nodules remain after cleaning

                center_y = int(np.mean(nodule_indices[0]))
                center_x = int(np.mean(nodule_indices[1]))

                # Ensure zoom factor does not exceed maximum
                zoom_factor = min(zoom_factor, max_zoom_factor)

                # Calculate crop area
                crop_size_x = int(rotated_image.shape[1] / zoom_factor)
                crop_size_y = int(rotated_image.shape[0] / zoom_factor)

                left = max(0, center_x - crop_size_x // 2)
                upper = max(0, center_y - crop_size_y // 2)
                right = min(rotated_image.shape[1], center_x + crop_size_x // 2)
                lower = min(rotated_image.shape[0], center_y + crop_size_y // 2)

                # Crop images
                zoomed_image = rotated_image[upper:lower, left:right]
                zoomed_label = rotated_label[upper:lower, left:right]

                # Resize images
                width_ratio = max_width / zoomed_image.shape[1]
                height_ratio = max_height / zoomed_image.shape[0]
                scaling_factor = min(width_ratio, height_ratio)

                new_width = int(zoomed_image.shape[1] * scaling_factor)
                new_height = int(zoomed_image.shape[0] * scaling_factor)

                resized_image = cv2.resize(zoomed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                resized_label = cv2.resize(zoomed_label, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

                # Pad images
                delta_w = max_width - new_width
                delta_h = max_height - new_height
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left_pad, right_pad = delta_w // 2, delta_w - (delta_w // 2)

                padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
                padded_label = cv2.copyMakeBorder(resized_label, top, bottom, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

                # Record non-padded area coordinates
                non_padded_x_min = left_pad
                non_padded_x_max = left_pad + new_width - 1
                non_padded_y_min = top
                non_padded_y_max = top + new_height - 1
                non_padded_bounds = (non_padded_x_min, non_padded_x_max, non_padded_y_min, non_padded_y_max)

                final_image_array = padded_image
                final_label_array = padded_label

                # Save augmented images and labels
                suffix = f"_augmented_{augmented_images_generated}"
                save_image_and_label(
                    final_image_array,
                    final_label_array,
                    base_filename,
                    slice_index,
                    output_folder_images_z,
                    output_folder_labels_z,
                    suffix=suffix,
                    class_id=0,
                    max_distance=10,
                    non_padded_bounds=non_padded_bounds
                )

                augmented_images_generated += 1
                total_images_generated += 1
                print(f"Generated {augmented_images_generated}/{images_needed} images for patient {patient_number}")

                if augmented_images_generated >= images_needed:
                    print(f"Required number of images generated for patient {patient_number}.")
                    break  # Move to next patient

            # Update rotation and zoom for next iteration
            iteration += 1
            rotation_degree += rotation_degree_increment
            zoom_factor += zoom_factor_increment

            # Ensure zoom factor does not exceed maximum
            if zoom_factor > max_zoom_factor:
                zoom_factor = max_zoom_factor

        print(f"Finished processing patient {patient_number}. Total augmented images generated: {augmented_images_generated}")

print("\nProcessing and augmentation completed.")
print(f"Total empty labels: {total_empty_labels}")
print(f"Total non-empty labels: {total_non_empty_labels}")
print(f"Total images generated: {total_images_generated}")

# Source paths
src_images = '/home/u154686/Thesis/Data/axis_z/augmented_data/images'
src_labels = '/home/u154686/Thesis/Data/axis_z/augmented_data/labels'

# Destination paths
dst_images = '/home/u154686/Thesis/Data/axis_z/train_data/images'
dst_labels = '/home/u154686/Thesis/Data/axis_z/train_data/labels'

def count_empty_labels(folder):
    empty_count = 0
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            label_path = os.path.join(folder, filename)
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    empty_count += 1
    return empty_count

print("\nMoving augmented files into training dataset...")

# Count empty labels before moving
empty_labels_before = count_empty_labels(dst_labels)

def move_files(src, dst):
    # Create destination folder if it does not exist
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    # Move files
    for filename in os.listdir(src):
        src_file = os.path.join(src, filename)
        dst_file = os.path.join(dst, filename)
        
        # Check if source is a file
        if os.path.isfile(src_file):
            shutil.move(src_file, dst_file)
            # Uncomment the following line if you want to print each moved file
            # print(f"Moved: {src_file} -> {dst_file}")

# Move images and labels
move_files(src_images, dst_images)
move_files(src_labels, dst_labels)

print("File moving completed.")

# Count empty labels after moving
empty_labels_after = count_empty_labels(dst_labels)

print(f"Empty labels before moving: {empty_labels_before}")
print(f"Empty labels after moving: {empty_labels_after}")
