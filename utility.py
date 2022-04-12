"""
 Generic utility functions
"""

import os

class Utility:
    """
    utility functions
    """

    @staticmethod
    def get_model_filepath(filepath: str):
        """
        Get the model file path.
        """
        return os.path.join('model', filepath)
    @staticmethod
    def get_input_filepath(filepath: str):
        """
        Get the input file path.
        """
        return os.path.join('input', filepath)
    @staticmethod
    def get_intermediate_input_filepath(filepath: str):
        """
        Get the input file path.
        """
        return os.path.join('input', filepath)
    @staticmethod
    def get_output_filepath(filepath: str):
        """
        Get the output file path.
        """
        return os.path.join('output', filepath)
