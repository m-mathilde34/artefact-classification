import os
import cv2

input_folder = "D:\images_raw"
output_folder = "D:\images_resized_64"

#get the new full path of the resized picture
def getFileOutput(filepath, outputfolder):
  path_breakdown = filepath.split("\\")

  #get each element of the final new image path
  folder_name = path_breakdown[2]
  filename = path_breakdown[-1]

  #create new path
  output = outputfolder + "\\" + folder_name + "\\" + filename
  return output


def checkFolderExist(outputpath):
  path_breakdown = outputpath.split("\\")
  full_path = "\\".join(path_breakdown[0:3])

  #create a new folder if the target one does not already exist
  if not os.path.exists(full_path):
    os.mkdir(full_path)



all_files = []

# Walk through the directory and its subdirectories
for root, _, files in os.walk(input_folder):
    for file in files:
        file_path = os.path.join(root, file)
        all_files.append(file_path)

for file_path in all_files:
    output_path = getFileOutput(file_path, output_folder)

    #check that the image does not already exist in our output folder
    if not os.path.exists(output_path):
      #get the image
      img = cv2.imread(file_path)

      if img is not None:
        #if the image exists and is not empty, resize image
        new_image = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        #create the new path and check that the folder exists
        output_path = getFileOutput(file_path, output_folder)
        checkFolderExist(output_path)

        #save image to new folder
        cv2.imwrite(output_path, new_image)
