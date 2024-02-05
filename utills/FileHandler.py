from pathlib import Path

class FileHandler:
    def __init__(self, legal_extensions, dataset_path, foldername, base_directory=None, include_subdirectories=True):
        """
        Initialize the FileHandler with legal file extensions, dataset path, folder name,
        an optional base directory, and whether to include subdirectories in the search.
        """
        self.include_subdirectories = include_subdirectories
        if base_directory:
            self.dataset_path = Path(base_directory).parent / foldername
        else:
            self.dataset_path = Path(dataset_path) / foldername

        self.legal_extensions = legal_extensions
        self.foldername = foldername
        self.path_holder = []

    def _refresh_path_holder(self):
        """
        Refresh the list of file paths based on the include_subdirectories option.
        """
        self.path_holder.clear()
        if self.include_subdirectories:
            search_pattern = '**/*'  # Search in all subdirectories
        else:
            search_pattern = '*'  # Search only in the top directory

        for ext in self.legal_extensions:
            self.path_holder.extend(self.dataset_path.glob(f'{search_pattern}.{ext}'))

    def files_no_extensions(self):
        """
        Returns a list of file names without extensions in the dataset path.
        """
        if not self.path_holder:
            self._refresh_path_holder()
        return [f.stem for f in self.path_holder]

    def path_file(self):
        """
        Returns a list of file paths with extensions.
        """
        if not self.path_holder:
            self._refresh_path_holder()
        return [f for f in self.path_holder]
    
    def files_with_extensions(self):
        """
        Returns a list of file names with extensions in the dataset path.
        """
        if not self.path_holder:
            self._refresh_path_holder()
        return [f.name for f in self.path_holder]
