import importlib
try:
    from comet_ml import Experiment as CometExperiment
    from comet_ml import OfflineExperiment as CometOfflineExperiment
except ImportError:  # pragma: no-cover
    _COMET_AVAILABLE = False
else:
    _COMET_AVAILABLE = True

from utils import Timer
import torch
from torch import is_tensor
from typing import Any, Dict, Optional, Union
class CometWriter:
    def __init__(
        self, 
        logger,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        api_key: Optional[str] = None, 
        log_dir: Optional[str] = None, 
        offline: bool = False,
        **kwargs):
        if not _COMET_AVAILABLE:
            raise ImportError(
                "You want to use `comet_ml` logger which is not installed yet,"
                " install it with `pip install comet-ml`."
            )

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.kwargs = kwargs

        self.timer = Timer()


        if (api_key is not None) and (log_dir is not None):
            self.mode = "offline" if offline else "online"
            self.api_key = api_key
            self.log_dir = log_dir

        elif api_key is not None:
            self.mode = "online"
            self.api_key = api_key
            self.log_dir = None
        elif log_dir is not None:
            self.mode = "offline"
            self.log_dir = log_dir
        else:
            logger.warning("CometLogger requires either api_key or save_dir during initialization.")

        if self.mode == "online":
            self.experiment = CometExperiment(
                api_key=self.api_key,
                project_name = self.project_name,
                **self.kwargs,
            )
        else:
            self.experiment = CometOfflineExperiment(
                offline_directory=self.log_dir,
                project_name=self.project_name,
                **self.kwargs,
            )

        if self.experiment_name:
            self.experiment.set_name(self.experiment_name)

    def set_step(self, step, epoch = None, mode='train') -> None:
        self.mode = mode
        self.step = step
        self.epoch = epoch
        if step == 0:
            self.timer.reset()
        else:
            duration = self.timer.check()
            self.add_scalar({'steps_per_sec': 1 / duration})

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        self.experiment.log_parameters(params)
    
    def log_code(self, file_name = None, folder = 'models/') -> None:
        self.experiment.log_code(file_name=file_name, folder=folder)


    def add_scalar(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None, epoch: Optional[int] = None) -> None:
        metrics_renamed = {}
        for key, val in metrics.items():
            tag = '{}/{}'.format(key, self.mode)
            if is_tensor(val):
                metrics_renamed[tag] = val.cpu().detach()
            else:
                metrics_renamed[tag] = val
        if epoch is None:
            self.experiment.log_metrics(metrics_renamed, step=self.step, epoch=self.epoch)
        else:
            self.experiment.log_metrics(metrics_renamed, epoch=epoch)

    def add_plot(self, figure_name, figure, epoch=None):
        """
        Primarily for log gate plots
        """
        self.experiment.log_figure(figure_name = figure_name, figure = figure, step = epoch)

    def add_hist3d(self, hist, name):
        """
        Primarily for log gate plots
        """
        self.experiment.log_histogram_3d(hist, name = name)

    def reset_experiment(self):
        self.experiment = None

    def finalize(self) -> None:
        self.experiment.end()
        self.reset_experiment()
                    

class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install either TensorboardX with 'pip install tensorboardx', upgrade " \
                    "PyTorch to version >= 1.1 for using 'torch.utils.tensorboard' or turn off the option in " \
                    "the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
            
        self.timer = Timer()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer.reset()
        else:
            duration = self.timer.check()
            self.add_scalar('steps_per_sec', 1 / duration)

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(metric_dict, *args, **kwargs):
                tag = list(metric_dict.keys())[0]
                data = metric_dict[tag]
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
