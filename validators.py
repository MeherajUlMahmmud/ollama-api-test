from schema import Schema, And, SchemaError


def validate_registration(data):
    schema = Schema({
        'username': And(str, len, lambda x: 3 <= len(x) <= 80, error='Username must be between 3 and 80 characters'),
        'name': And(str, len, lambda x: 3 <= len(x) <= 80, error='Name must be between 3 and 80 characters'),
        'password': And(str, len, lambda x: 8 <= len(x) <= 120, error='Password must be between 8 and 120 characters'),
    })

    try:
        schema.validate(data)
        return None
    except SchemaError as e:
        return [str(e)]


def validate_login(data):
    schema = Schema({
        'username': And(str, len, lambda x: len(x) > 0, error='Username is required'),
        'password': And(str, len, lambda x: len(x) > 0, error='Password is required'),
    })

    try:
        schema.validate(data)
        return None
    except SchemaError as e:
        return [str(e)]


def validate_nid_upload(files):
    schema = Schema({
        'front_image': And(lambda x: x is not None, error='Front image is required'),
        'back_image': And(lambda x: x is not None, error='Back image is required'),
    })

    try:
        schema.validate(files)
        return None
    except SchemaError as e:
        return [str(e)]


def validate_liveness_upload(files):
    schema = Schema({
        'image': And(lambda x: x is not None, error='Image is required'),
    })

    try:
        schema.validate(files)
        return None
    except SchemaError as e:
        return [str(e)]


def validate_query(data):
    schema = Schema({
        'query': And(str, len, lambda x: len(x) > 0, error='Query is required'),
    })

    try:
        schema.validate(data)
        return None
    except SchemaError as e:
        return [str(e)]
