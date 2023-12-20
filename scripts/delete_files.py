import os
import argparse

from tools.data.file_ops import get_absolute_path
from tools.data.file_type import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def delete_files(folder_path, extensions):
    # Traverse all files in the folder
    count_deleted = 0  # Variable to count the number of deleted images
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if extensions == 'all':
                # remove all the files
                os.remove(file_path)
                count_deleted += 1
                print(f"Deleted: {file_path}")
            else:
                # Check if the file extension is a specified format (Add/modify file types as needed)
                if file_path.lower().endswith(tuple(extensions)):
                    # Delete image files
                    os.remove(file_path)
                    count_deleted += 1
                    print(f"Deleted: {file_path}")
    if count_deleted == 0:
        print("No files were deleted.")
    elif count_deleted == 1:
        print("1 file was deleted.")
    else:
        print(f"{count_deleted} files were deleted.")


def delete_image_files(folder_path):
    delete_files(folder_path, IMAGE_EXTENSIONS)


def delete_video_files(folder_path):
    delete_files(folder_path, VIDEO_EXTENSIONS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='all')
    parser.add_argument('-d', '--dir', type=str, default='./test_samples/results')
    args = parser.parse_args()

    # Call the function and pass the folder path containing the images to delete
    folder_to_clean = get_absolute_path(relative_path=args.dir)  # Replace with the actual folder path

    match args.type:
        case 'img':
            delete_image_files(folder_to_clean)
        case 'video':
            delete_video_files(folder_to_clean)
        case _:
            delete_files(folder_to_clean, extensions='all')
