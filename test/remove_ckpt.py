import os

ckpt_path = os.path.abspath('../resource/ckpt_folder')

file_list = os.listdir(ckpt_path)

for file in file_list:
    if 'predict' in file:
        iter_idx = int(file.strip().split('.')[-2])
        if iter_idx < 2000:
            file_path = os.path.join(ckpt_path, file)
            os.remove(file_path)
print('')