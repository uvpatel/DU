"""DU: Data Understanding."""

from DU.api.server import deploy_api
from DU.app.dashboard import run_app
from DU.core.loader import load
from DU.eda.insights import insights
from DU.eda.summary import summary
from DU.eda.visualize import plot
from DU.ml.evaluate import evaluate
from DU.ml.train import TrainResult, train
from DU.version import __version__

__all__ = [
	"__version__",
	"load",
	"summary",
	"plot",
	"insights",
	"train",
	"TrainResult",
	"evaluate",
	"deploy_api",
	"run_app",
]