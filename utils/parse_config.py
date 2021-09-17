import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from sklearn.utils.fixes import loguniform
from sklearn.experimental import enable_halving_search_cv  # noqa
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')


    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        cfg_fname = Path(args.config)

        config = read_json(cfg_fname)

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type'] # can do because of __getitem__
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_params(self, search_method_params):
        tuned_parameters = self["tuned_parameters"]
        search_type = self["search_method"]["type"]

        assert search_type == "GridSearchCV" or \
                search_type == "RandomizedSearchCV" or \
                search_type == "HalvingGridSearchCV" or \
                search_type == "HalvingRandomSearchCV" , \
                f"Search type {search_type} not supported."

        if "Grid" in search_type:
            search_method_params['param_grid'] = tuned_parameters
        else:
            for method_name in tuned_parameters[0]:
                if len(tuned_parameters[0][method_name]) == 2:
                    temp = tuned_parameters[0][method_name]
                    tuned_parameters[0][method_name] = loguniform(temp[0], temp[1])
            search_method_params['param_distributions'] = tuned_parameters
        return search_method_params

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
