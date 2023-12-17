import os

from tools.data.file_ops import get_absolute_path
from tools.data.file_type import IMAGE_EXTENSIONS


def delete_files(folder_path, extensions):
    # Traverse all files in the folder
    count_deleted = 0  # Variable to count the number of deleted images
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file extension is a specified format (Add/modify file types as needed)
            if file_path.lower().endswith(tuple(extensions)):
                # Delete image files
                os.remove(file_path)
                count_deleted += 1
                print(f"Deleted: {file_path}")
    if count_deleted == 0:
        print("No image files were deleted.")
    elif count_deleted == 1:
        print("1 image file was deleted.")
    else:
        print(f"{count_deleted} image files were deleted.")


def delete_image_files(folder_path):
    delete_files(folder_path, IMAGE_EXTENSIONS)


if __name__ == '__main__':
    # Call the function and pass the folder path containing the images to delete
    folder_to_clean = get_absolute_path(relative_path='./test_samples/results')  # Replace with the actual folder path
    delete_image_files(folder_to_clean)
