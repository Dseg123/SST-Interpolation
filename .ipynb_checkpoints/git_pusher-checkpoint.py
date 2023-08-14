import os
from pathlib import Path

p = Path('experiments/')
path_list = (list(p.glob('*')))

for path in path_list:
    os.system("git add " + str(path))
    os.system(f"git commit -m {str(path)}")
    os.system("git push origin main")