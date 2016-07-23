import os, shutil

def clear_Paths_folder(path = 'Plots'):
    requested_path = os.getcwd() + "/"+ path
    if not os.path.exists(requested_path):
        print("The path you requested does not exist!")
        return
    for the_file in os.listdir(requested_path):
        file_path = os.path.join(requested_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)