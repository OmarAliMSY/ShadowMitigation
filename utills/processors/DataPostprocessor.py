
from Processor import Processor

class DataPreprocessor(Processor):
    """
    Abstract class for data preprocessor.
    """
    
    def __init__(self):
        super(DataPreprocessor, self).__init__()
    
    def process(self, data):
        """
        Process data.
        
        :param data: data to process
        :type data: any
        :return: processed data
        :rtype: any
        """
        return data
