def check_value(predicate, error_msg):
    if not predicate:
        raise ValueError(error_msg)
