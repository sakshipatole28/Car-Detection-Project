import numpy as np
import pytest

from cardet.config import HogConfig, load_params
from cardet.features import HogExtractor


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def extractor():
    return HogExtractor(HogConfig())


@pytest.fixture
def rng():
    return np.random.default_rng(0)
