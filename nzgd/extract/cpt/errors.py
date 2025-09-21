"""Custom exceptions for unsuccessful CPT extraction attempts."""


class FileProcessingError(Exception):
    """Custom exception class for file processing errors.

    This exception is raised when there is an error related to file processing,
    such as issues with reading or parsing files.

    Attributes:
        Inherits all attributes from the base Exception class.

    """


class IncorrectNumberOfColumnsError(Exception):
    """Indicates that an incorrect number of columns has been extracted.

    Attributes:
        Inherits all attributes from the base Exception class.

    """

    def __init__(
        self,
        message: str = "An incorrect number of columns was extracted.",
    ) -> None:
        """Initialize the exception with a custom message.

        Parameters
        ----------
        message : str, optional
            The error message to display. Defaults to "An incorrect number of columns
            was extracted.".

        """
        super().__init__(message)


class UnsupportedInputFileTypeError(Exception):
    """Indicates that an unsupported file type was encountered.

    Attributes:
        Inherits all attributes from the base Exception class.

    """


class InvalidExcelFileError(Exception):
    """Indicates that an invalid Excel file was encountered.

    Attributes:
        Inherits all attributes from the base Exception class.

    """
