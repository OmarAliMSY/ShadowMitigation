from pathlib import Path


legal_extensions = ['jpg', 'jpeg']
parent_directory_path = Path(__file__).parent
dataset_path = Path(parent_directory_path, 'random')

path_holder = []
[path_holder.extend(list(dataset_path.rglob(f'*.{e}')))
               for e in legal_extensions]

for path in path_holder:
    print(dir(path))
    pass
