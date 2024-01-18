from ...kernel_desc import KernelDescription, get_possible_types, select_pattern
from ...autotune_binning import BinningLessOrEqual, BinningExact

class FlashKernel(KernelDescription):
    KERNEL_FAMILY = 'flash'
