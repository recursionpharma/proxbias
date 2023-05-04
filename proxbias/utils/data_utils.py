from proxbias.utils.constants import DATA_DIR


def _get_data_path(name):
    return DATA_DIR.joinpath(name)
