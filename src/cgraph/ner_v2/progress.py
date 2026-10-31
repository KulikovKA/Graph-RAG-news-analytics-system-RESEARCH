"""Terminal progress on stderr; suppress animated output when redirected."""
from tqdm import tqdm


def progress(iterable=None, **kwargs):
    return tqdm(iterable, **{"disable": None, "dynamic_ncols": True, **kwargs})
