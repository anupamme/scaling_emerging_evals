import sse
import sse.analysis
import sse.metrics
import sse.models
import sse.runners
import sse.tasks


def test_package_exists():
    assert sse.__name__ == "sse"


def test_subpackages_exist():
    assert sse.analysis.__name__ == "sse.analysis"
    assert sse.metrics.__name__ == "sse.metrics"
    assert sse.models.__name__ == "sse.models"
    assert sse.runners.__name__ == "sse.runners"
    assert sse.tasks.__name__ == "sse.tasks"
