#from .tfbind8 import TfBind8
from src.datasets.dkitty_morphology import DKittyMorphology
from .superconductor import Superconductor
from .ant_morphology import AntMorphology
from .dkitty_morphology import DKittyMorphology
from .tfbind8_ood0 import TfBind8_OOD0
from .tfbind8_ood01 import TfBind8_OOD01
from .hopper import Hopper50
#from .dataset import ContinuousDataset

def get_dataset(task_name, oracle_name, split, gain, gain_y):
    name2class = {
        'Superconductor': Superconductor,
        'TFBind8_OOD0': TfBind8_OOD0,
        'TFBind8_OOD01': TfBind8_OOD01,
        'AntMorphology': AntMorphology,
        'DKittyMorphology': DKittyMorphology,
        'Hopper50': Hopper50
    }
    
    return name2class[task_name](
        oracle_name,
        split=split,
        gain=gain,
        gain_y=gain_y
    )