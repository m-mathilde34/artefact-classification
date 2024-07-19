import csv
import urllib.request

filepath = 'D:\images_raw\\6_asian'
filename = '6_MET_'
list_url = []

with open('asiandata.csv') as asian_files:
    reader_file = csv.reader(asian_files)

    # skip header
    header = next(asian_files)

    for row in reader_file:
        url_string = str(row)
        url_string_clean = url_string.strip("[']")

        if url_string_clean == "":
            pass
        else:
            list_url.append(url_string_clean)
            img_name_split = url_string_clean.split('/')
            img_id = img_name_split[-2]
            filename_final = filename + img_id
            full_path = filepath + '\ ' + filename_final + '.jpeg'
            urllib.request.urlretrieve(url_string_clean, full_path)
