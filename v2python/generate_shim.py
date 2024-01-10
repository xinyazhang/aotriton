from .rules import kernels as triton_kernels
from .tuning_database import KernelTuningDatabase
import io
import shutil
import argparse
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve()
CSRC = (SOURCE_PATH.parent.parent / 'csrc').absolute()
INCBIN = (SOURCE_PATH.parent.parent / 'third_party/incbin/').absolute()
COMMON_INCLUDE = (SOURCE_PATH.parent.parent / 'include/').absolute()
# COMPILER = SOURCE_PATH.parent / 'compile.py'
COMPILER = 'hipcc'
LINKER = 'ar'

LIBRARY_NAME = 'libaotriton'

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target_gpus", type=str, default=None, nargs='*',
                   help="Ahead of Time (AOT) Compile Architecture. PyTorch is required for autodetection if --targets is missing.")
    p.add_argument("--build_dir", type=str, default='build/', help="build directory")
    p.add_argument("--archive_only", action='store_true', help='Only generate archive library instead of shared library. No linking with dependencies.')
    args = p.parse_args()
    # print(args)
    return args

'''
ShimMakefileGenerator
 +- generate libaotriton.a
 +- KernelShimGenerator
     +- generate kernel launcher header
     +- collect objects
     +- generate kernel launcher source
'''

class Generator(object):

    def __init__(self, args, out):
        self._args = args
        self._out = out
        self._children = []

    @property
    def is_file(self):
        return False

    def generate(self):
        self.write_prelude()
        self.loop_children()
        self.write_body()
        self.write_conclude()

        if self._out is not None:
            self._out.flush()

    @property
    def children_out(self):
        return self._out

    def gen_children(self, out):
        return
        yield

    def write_prelude(self):
        pass

    def loop_children(self):
        for c in self.gen_children(self.children_out):
            c.generate()
            self._children.append(c)

    def write_body(self):
        pass

    def write_conclude(self):
        pass

class MakefileSegmentGenerator(Generator):
    @property
    def list_of_output_object_files(self) -> 'list[Path]':
        return sum([c.list_of_output_object_files for c in self._children], [])

class MakefileGenerator(MakefileSegmentGenerator):
    def __init__(self, args, grand_target, out):
        super().__init__(args, out)
        self._main_content = io.StringIO()
        self._targets = []
        self._grand_target = grand_target
        self._phony = []

    @property
    def children_out(self):
        return self._main_content

    def gen_children(self, out):
        pass

    def write_prelude(self):
        pass

    def write_body(self):
        self._main_content.seek(0)
        shutil.copyfileobj(self._main_content, self._out)

    def write_conclude(self):
        print('.PHONY: ', ' '.join(self._phony), file=self._out)

class ShimMakefileGenerator(MakefileGenerator):

    def __init__(self, args):
        # grand_target = LIBRARY_NAME + '.a' if args.archive else '.so'
        grand_target = LIBRARY_NAME
        self._build_dir = Path(args.build_dir)
        f = open(self._build_dir / 'Makefile.shim', 'w')
        super().__init__(args=args, grand_target=grand_target, out=f)
        self._library_suffixes = ['.a']  if args.archive_only else ['.a', '.so']

    def __del__(self):
        self._out.close()

    def gen_children(self, out):
        for k in triton_kernels:
            yield KernelShimGenerator(self._args, self.children_out, k)

    def write_prelude(self):
        f = self._out
        super().write_prelude()
        print(f"HIPCC={COMPILER}", file=f)
        print(f"AR={LINKER}", file=f)
        print(f"", file=f)
        print('', file=self._out)
        print(self._grand_target, ':', ' '.join([f'{LIBRARY_NAME}{s}' for s in self._library_suffixes]), '\n\n', file=self._out)

    def write_conclude(self):
        f = self._out
        all_object_files = ' '.join([str(p) for p in self.list_of_output_object_files])
        for s in self._library_suffixes:
            fn = f'{LIBRARY_NAME}{s}'
            print(fn, ': ', all_object_files, file=self._out)
            if s == '.a':
                print('\t', '${AR} -r ', fn, all_object_files, file=f)
            if s == '.so':
                print('\t', COMPILER, ' -shared -fPIC -o ', fn, all_object_files, file=f)
            print('\n\n', file=f)

    '''
    @property
    def _object_relative_paths(self):
        return [str(op.relative_to(self._build_dir)) for op in self.list_of_output_object_files]
    '''

class KernelShimGenerator(MakefileSegmentGenerator):
    AUTOTUNE_TABLE_PATH = 'autotune_table'

    def __init__(self, args, out, k : 'KernelDescription'):
        super().__init__(args, out)
        # Shim code and functional dispatcher
        self._kdesc = k
        self._kdesc.set_target_gpus(args.target_gpus)
        self._shim_path = Path(args.build_dir) / k.KERNEL_FAMILY
        self._shim_path.mkdir(parents=True, exist_ok=True)
        self._fhdr = open(self._shim_path / Path(k.SHIM_KERNEL_NAME + '.h'), 'w')
        self._fsrc = open(self._shim_path / Path(k.SHIM_KERNEL_NAME + '.cc'), 'w')
        # Autotune dispatcher
        self._autotune_path = Path(args.build_dir) / k.KERNEL_FAMILY / f'autotune.{k.SHIM_KERNEL_NAME}'
        self._autotune_path.mkdir(parents=True, exist_ok=True)
        self._ktd = KernelTuningDatabase(SOURCE_PATH.parent / 'rules', k.SHIM_KERNEL_NAME)

    def __del__(self):
        self._fhdr.close()
        self._fsrc.close()

    def write_prelude(self):
        self._kdesc.write_launcher_header(self._fhdr)

    def gen_children(self, out):
        k = self._kdesc
        p = self._shim_path
        args = self._args
        ktd = KernelTuningDatabase(SOURCE_PATH.parent / 'rules', k.SHIM_KERNEL_NAME)
        debug_counter = 0
        for gpu, fsels, lut in k.gen_tuned_kernel_lut(self._ktd):
            # print(f'KernelShimGenerator.gen_children {fsels=}')
            yield AutotuneCodeGenerator(args, self.children_out, self._autotune_path, k, gpu, fsels, lut)
            '''
            debug_counter +=1
            if debug_counter >= 2:
                break
            '''

        for o in k.gen_all_object_files(p, tuned_db=ktd):
            yield ObjectShimCodeGenerator(self._args, k, o)

    def write_conclude(self):
        self._kdesc.write_launcher_source(self._fsrc, [c._odesc for c in self._children if isinstance(c, ObjectShimCodeGenerator)])

class AutotuneCodeGenerator(MakefileSegmentGenerator):
    def __init__(self, args, fileout, outdir, k, gpu, fsels, lut):
        super().__init__(args, fileout)
        self._build_dir = Path(args.build_dir)
        self._outdir = outdir
        self._kdesc = k
        self._gpu = gpu
        self._fsels = fsels
        self._lut = lut

    def write_body(self):
        # Write the code to file
        self._ofn = self._lut.write_lut_source(self._outdir)
        self._obj_fn = self._ofn.with_suffix('.o')
        self._makefile_target = self._obj_fn.relative_to(self._build_dir)
        # Write the Makefile segment
        print('#', self._fsels, file=self._out)
        print(self._makefile_target, ':', self._ofn.relative_to(self._build_dir), file=self._out)
        cmd  = '$(HIPCC) ' + f'{self._ofn.absolute()} -I{CSRC} -I{INCBIN} -I{COMMON_INCLUDE} -o {self._obj_fn.absolute()} -c -fPIC -std=c++20'
        print('\t', cmd, '\n', file=self._out)

    @property
    def list_of_output_object_files(self) -> 'list[Path]':
        return [self._makefile_target]

# FIXME: a better name.
#        This class name is legacy and now it's only used to store
#        ObjectFileDescription objects to keep record of metadata for compiled
#        kernels
class ObjectShimCodeGenerator(Generator):
    def __init__(self, args, k, o):
        super().__init__(args, None)
        self._kdesc = k
        self._odesc = o

    def get_all_object_files(self):
        return [self._odesc._hsaco_kernel_path.with_suffix('.o')] if self._odesc.compiled_files_exist else []

    def loop_children(self):
        pass

    @property
    def list_of_output_object_files(self) -> 'list[Path]':
        return []

def main():
    args = parse()
    gen = ShimMakefileGenerator(args)
    gen.generate()

if __name__ == '__main__':
    main()
