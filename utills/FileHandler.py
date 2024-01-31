from pathlib import Path

class FileHandler:
    def __init__(self, legal_extensions, dataset_path, foldername, base_directory=None):
        """
        Initialize the FileHandler with legal file extensions, dataset path, folder name,
        and an optional base directory.
        """
        if base_directory:
            self.dataset_path = Path(base_directory).parent / foldername
        else:
            self.dataset_path = Path(dataset_path) / foldername

        self.legal_extensions = legal_extensions
        self.foldername = foldername
        self.path_holder = []

    def files_no_extensions(self):
        """
        Returns a list of file names without extensions in the dataset path.
        """
        if not self.path_holder:
            self.path_holder.extend(
                [f for ext in self.legal_extensions for f in self.dataset_path.rglob(f'*.{ext}')]
            )
        return [f.stem for f in self.path_holder]

    def path_file(self):
        """
        Returns a list of file paths with extensions.
        """
        if not self.path_holder:
            self.path_holder.extend(
                [f for ext in self.legal_extensions for f in self.dataset_path.rglob(f'*.{ext}')]
            )
        return [f for f in self.path_holder]
    
    def files_with_extensions(self):
        """
        Returns a list of file names with extensions in the dataset path.
        """
        if not self.path_holder:
            self.path_holder.extend(
                [f for ext in self.legal_extensions for f in self.dataset_path.rglob(f'*.{ext}')]
            )
            return [f.name for f in self.path_holder]


