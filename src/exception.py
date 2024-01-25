import os
import sys
from utils.const import PROJ_PATH

def error_message_detail(error, error_detail: sys):
    _, _, exc_traceback = error_detail.exc_info()
    file_name = os.path.relpath(exc_traceback.tb_frame.f_code.co_filename, PROJ_PATH)
    error_message = "Error occurred in module with name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_traceback.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail=error_detail)

    def __str__(self):
        return self.error_message

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        raise CustomException(ValueError("My value error"), sys)


