from contextlib import contextmanager


class ConsoleLogger:
    def __init__(self):
        pass

    def log(self, description: str):
        print(description)

    @contextmanager
    def task(self, description: str):
        print(description, end="")
        try:
            yield None
        except Exception as e:
            print(" [error]")
            print(str(e))
        else:
            print(" [done]")
