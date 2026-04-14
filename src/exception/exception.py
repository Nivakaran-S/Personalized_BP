import sys


class BPException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(str(error_message))
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = -1
            self.file_name = "<unknown>"

    def __str__(self):
        return (
            f"Error occurred in python script name [{self.file_name}] "
            f"line number [{self.lineno}] error message [{self.error_message}]"
        )