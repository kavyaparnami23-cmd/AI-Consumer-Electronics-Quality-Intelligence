import sys


class CustomException(Exception):
    """
    Custom Exception Class
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_detail.exc_info()

        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno

        self.error_message = f"""
==================================================
ERROR OCCURRED
--------------------------------------------------
File Name   : {self.file_name}
Line Number : {self.line_number}
Error       : {error_message}
==================================================
"""

    def __str__(self):
        return self.error_message


# ==========================
# Testing
# ==========================

def test_exception():
    try:
        a = 10
        b = 0
        c = a / b

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    test_exception()