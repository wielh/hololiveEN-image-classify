from common_parameter import test_root_dir,train_root_dir
import os,glob,random,shutil

all_root_dir='C:\\Users\\William\\Desktop\\hololive-ai\\all_pictures'
os.chdir(all_root_dir)
for file in glob.iglob("*\\*.jpg", recursive=True):
    num = random.uniform(0,1)
    if num>0.85:
        dest_path = os.path.join(test_root_dir,file)
    else:
        dest_path = os.path.join(train_root_dir,file)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copyfile(file, dest_path)
