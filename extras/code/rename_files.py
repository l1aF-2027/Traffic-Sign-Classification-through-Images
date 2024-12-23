import os
folder_path = r''
file_list = os.listdir(folder_path)
file_list.sort()
i = 0
for old_name in file_list:
    old_file = os.path.join(folder_path, old_name)
    new_file = os.path.join(folder_path,"Chidan_" + str(i) + os.path.splitext(old_name)[1])
    os.rename(old_file, new_file)
    print(f'Đã đổi tên {old_name} thành {"Chidan_" + str(i) + os.path.splitext(old_name)[1]}')
    i += 1