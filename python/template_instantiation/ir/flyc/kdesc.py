# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelDescription — the codegen-facing IR for a flyc (FlyDSL-compiled) kernel.

Mirrors `ir/affine/kdesc.py`'s shape: a minimal `Interface` implementer that owns
no functional space of its own (PLAN.md 6.3) — a flyc kernel inherits and filters
the operator its `functionals_of=` names. Built by the linker
(`codegen/linker.py:_build_flycs`, Task 5a) from a `FlycDecl`, which resolves that
name against the already-built operators and passes the Operator object in as
`functionals_source`.

`_axes_overrides` / `axes_multi` delegate to `functionals_source` so
`Interface.gen_functionals` enumerates exactly the axes the operator does, but
every yielded `Functional` still carries `meta_object=self` (this flyc kdesc, not
the operator) — that is what gives filepack paths / hsaco entry names / the
`Fly.compile` KERNEL_NAME column this kernel's own NAME/FAMILY rather than the
operator's (PLAN-PHASE1.md Task 7a: "Give flyc its own zip ... keyed by the
description's name").
"""

from ..interface import Interface
from ..context_helper import ContextHelper
from ..choices import ChoiceVarAbsent
from aotriton.template_instantiation.builder import DescriptionError


# Elemental type string (as authored on an @ati.scalar) -> C scratch-member
# type. Only the types actually used by today's context helpers
# (flyc_varlen_bits/flyc_batch_size/flyc_num_seqlens/flyc_idropout_p: 'i32';
# flyc_dropout_scale: 'fp32') are exercised, but the table covers the full
# elemental vocabulary a future helper could plausibly use (PLAN-PHASE2.md
# Task 5). Deliberately NOT typed_choice.ELEMENTAL_TYPE_MAP, which maps to
# TypedChoice factory classes, not C type-name strings.
_CONTEXT_HELPER_CTYPE = {
    'i8': 'int8_t', 'i16': 'int16_t', 'i32': 'int32_t', 'i64': 'int64_t',
    'u8': 'uint8_t', 'u16': 'uint16_t', 'u32': 'uint32_t', 'u64': 'uint64_t',
    'fp16': 'float', 'bf16': 'float', 'fp32': 'float',
}


class KernelDescription(Interface):
    """A flyc kernel built from the @ati.flyc.* stacked form. ATI-native, like
    AffineKernel: subclasses the ATI Interface base (identity surface) directly,
    no functional space of its own (inherits the operator's via
    `functionals_source`), no perf space (`ir/flyc/ksignature.py` leaves perf/copt
    sections empty — the FlyDSL tuning model is unsettled, PLAN-PHASE1.md 0c)."""

    CODEGEN_MODULE = 'flyc'
    TUNE_NAME = 'flytune'
    FILE_PFX = 'flyc'
    ENUM_PREFIX = 'kFlyc_'
    is_tunable = False

    def __init__(self, built, *, family, source_path,
                functionals_source=None, tensors=None, scalars=None,
                builder_fn=None, hints_cls=None):
        # `built` is the BuiltKernel build_kernel() lowered this kernel's
        # cite-resolved FlycDecl clone into: it answers
        # "what are this kernel's arguments?" (order, disables) -- NOT "which
        # variants exist?", which stays functionals_source's job below (the
        # design caveat: a BuiltKernel giving flyc its own axes would make it
        # enumerate a second, wrong functional space).
        self._built = built
        self.NAME = built.name
        self.FAMILY = family
        # `source_path`, not `MODULE_PATH`: ir/triton/kdesc.py already calls the
        # kernel's own file `source_path`, so the old name was a third spelling
        # of one concept (FlycDecl.module_path was the second).
        self.source_path = source_path
        # The Operator this kernel's `functionals_of=` names, resolved by the
        # linker (Task 5a). None only transiently, between __new__ and the
        # linker's assignment -- every kdesc actually handed to the generator has
        # this set (collect_flyc_decl asserts functionals_of is present).
        self._functionals_source = functionals_source
        self.desc_path = None                          # set by the linker (Task 5c's DESC column)
        # The @ati.tensor/@ati.scalar decorator specs stacked on the description
        # (FlycDecl.tensors/.scalars), and the builder placeholder function itself
        # plus its hints dataclass (FlycDecl.fn/.hints_cls) -- all threaded through
        # by the linker's _build_flycs (PLAN-PHASE2.md Task 5). These are what
        # iter_launch_arguments()/iter_context_helpers() walk, and what
        # FlycTuneCodeGenerator (python/codegen/flytune.py) calls directly
        # (builder_fn(choices, hints), plus hints() for the 2nd argument) to
        # obtain the knobs dict at generate time -- WITHOUT ever calling the
        # `build` callable that same call returns, which is what would import
        # flydsl.
        self.tensors = list(tensors) if tensors else []
        self.scalars = list(scalars) if scalars else []
        self.builder_fn = builder_fn
        self.hints_cls = hints_cls

    @property
    def perf_cfields(self):
        return []

    def _axes_overrides(self):
        """(axes, overrides) to enumerate over -- the referenced operator's own
        (PLAN.md 6.3). `Interface.gen_functionals` uses these but still stamps
        `meta_object=self` on every yielded Functional (see module docstring)."""
        return self._require_functionals_source()._axes_overrides()

    @property
    def axes_multi(self):
        """Delegated to functionals_source: `Functional.compact_choices` /
        `.unified_signature` (ir/functional.py) read `meta_object.axes_multi`, and
        meta_object is THIS kdesc once `_axes_overrides` is wired in (see above)."""
        return self._require_functionals_source().axes_multi

    def _require_functionals_source(self):
        assert self._functionals_source is not None, (
            f'flyc kernel {self.NAME!r} has no functionals_source '
            f'(PLAN-PHASE1.md Task 5a)')
        return self._functionals_source

    def list_functional_params(self):
        return self._require_functionals_source().list_functional_params()

    @property
    def func_cfields(self):
        # The kernarg ABI is not the operator's params struct (PLAN.md's third
        # consequence) — Phase 1 declares no struct contribution. Phase 2's
        # wires_to consumption is what would populate this.
        return []

    # --- identity: borrow the operator's params/context/call_options surface.
    # Mirrors ir/triton/kdesc.py's SHARED_IFACE-aware param_class_name, except
    # for flyc SHARED_IFACE is always the referenced operator (never None) --
    # a flyc kernel never owns its own params struct.

    @property
    def SHARED_IFACE(self):
        return self._require_functionals_source()

    @property
    def param_class_name(self):
        return self.SHARED_IFACE.param_class_name

    @property
    def godel_number(self):
        return self._require_functionals_source().godel_number

    @property
    def axes_all_ordered(self):
        return self._require_functionals_source().axes_all_ordered

    def axis_of_arg(self, aname):
        return self._require_functionals_source().axis_of_arg(aname)

    def axis_by_var(self, var_name):
        return self._require_functionals_source().axis_by_var(var_name)

    def override_for(self, aname):
        return self._require_functionals_source().override_for(aname)

    def apparel_of(self, real_arg):
        return self._require_functionals_source().apparel_of(real_arg)

    def real_of(self, apparel_arg):
        return self._require_functionals_source().real_of(apparel_arg)

    def is_functional_disabled(self, functional):
        # Mirrors ir/triton/kdesc.py's is_functional_disabled exactly:
        # self._built.disables is the cite-resolved list (build_kernel /
        # resolve_cites already applied the rule that a local @ati.disable
        # replaces the cited one), so there is no separate "self._disable"
        # concept left here.
        for d in self._built.disables:
            try:
                if d.holds(functional):
                    return True
            except ChoiceVarAbsent as e:
                pred = getattr(d.when, '__name__', d.when)
                raise DescriptionError(
                    f"kernel {self.NAME!r}: the @ati.disable predicate {pred!r} reads "
                    f"a choice variable this kernel does not have ({e}). It is likely "
                    f"inherited via @ati.cite from a kernel with a different choice "
                    f"space. Declare a local @ati.disable on {self.NAME!r} that uses "
                    f"only its own choice variables (a local disable replaces the "
                    f"cited one).") from e
        return False

    # --- builder invocation: the code generator (FlycTuneCodeGenerator,
    # python/codegen/flytune.py) calls `self.builder_fn(choices, hints)`
    # directly and keeps only the `knobs` half of its `(build, knobs)` return
    # -- see that module for the sys.path setup.
    # There is deliberately no `build()` method on this class: the generator
    # must never invoke the FlyDSL compiler, and a wrapper method here would
    # just be one more place that could grow a `build()` call by mistake. Only
    # `python/flyc_compile.py`, run by ninja at build time, may call the
    # deferred `build` callable a description's fn(choices, hints) returns. ---

    def hints(self):
        """The hints dataclass instance passed as the builder's 2nd argument, or
        None if the description declared no @ati.flyc.hints. flytune.py has no
        `--hints` CLI override at generate time (flyc_compile.py's is a Phase-1,
        stand-alone-driver-only concept), so defaults are always used."""
        return self.hints_cls() if self.hints_cls is not None else None

    # --- launch-argument vector (PLAN-PHASE2.md Task 5) ---

    def iter_launch_arguments(self):
        """Yield the C++ launch-argument vector entries in the REAL kernel's
        signature order (see codegen.common.LaunchArg). Unlike triton's
        equivalent, there is no `aux` (no global_scratch/profile_scratch: the
        flyc kernel's 44 parameters don't include Triton's two trailing scratch
        pointers) and a 4th LaunchArg.kind, 'context_helper', is possible."""
        # Lazy: same aotriton.codegen -> template_instantiation cycle as
        # ir/triton/kdesc.py's iter_launch_arguments.
        from aotriton.codegen.common import LaunchArg

        real_param_order = self._built.arguments

        tensor_ptr_lookup = {}
        for t in self.tensors:
            for name in t.arg_names:
                tensor_ptr_lookup[name] = t
        scalar_lookup = {}
        for s in self.scalars:
            for name in s.arg_names:
                scalar_lookup[name] = s
        stride_owner = {}
        for t in self.tensors:
            for dim, sname in enumerate(t.match_strides(real_param_order)):
                stride_owner[sname] = (t, dim)

        def _apparel(spec):
            return spec.wires_to if isinstance(spec.wires_to, str) else spec.arg_name

        for name in real_param_order:
            if name in stride_owner:
                t, dim = stride_owner[name]
                yield LaunchArg(aname=name, kind='tensor_stride',
                                 expr=f'params.{_apparel(t)}->kparam_stride({dim})')
            elif name in tensor_ptr_lookup:
                t = tensor_ptr_lookup[name]
                yield LaunchArg(aname=name, kind='tensor_ptr',
                                 expr=f'params.{_apparel(t)}->kparam_data_ptr()')
            elif name in scalar_lookup:
                s = scalar_lookup[name]
                if isinstance(s.wires_to, ContextHelper):
                    yield LaunchArg(aname=name, kind='context_helper',
                                     expr=f'CAST(&context.scratch_params.{s.wires_to.name})')
                else:
                    yield LaunchArg(aname=name, kind='scalar',
                                     expr=f'CAST(&params.{_apparel(s)})')
            else:
                assert False, (
                    f'flyc kernel {self.NAME!r}: undeclared kernel parameter '
                    f'{name!r} (not bound by any @ati.tensor/@ati.scalar and not '
                    f'a resolved stride argument)')

    def iter_context_helpers(self):
        """Yield (helper_name, c_type) once per distinct `ati.context_helper`
        referenced by this kernel's @ati.scalar specs, in first-seen order.
        Drives both the context struct's declares/scratch-members (flyc.h) and
        the pp_args preamble that populates them (PLAN-PHASE2.md Task 5)."""
        seen = {}
        for s in self.scalars:
            if not isinstance(s.wires_to, ContextHelper):
                continue
            name = s.wires_to.name
            assert s.type_ is not None, (
                f'flyc kernel {self.NAME!r}: context_helper scalar {s.arg_name!r} '
                f'has no explicit elemental type (dtype ChoiceVar not supported '
                f'for context helpers)')
            assert s.type_ in _CONTEXT_HELPER_CTYPE, (
                f'flyc kernel {self.NAME!r}: context_helper scalar {s.arg_name!r} '
                f'has unrecognised elemental type {s.type_!r}')
            ctype = _CONTEXT_HELPER_CTYPE[s.type_]
            if name in seen:
                assert seen[name] == ctype, (
                    f'flyc kernel {self.NAME!r}: context_helper {name!r} is '
                    f'referenced with inconsistent C types {seen[name]!r} vs '
                    f'{ctype!r}')
                continue
            seen[name] = ctype
            yield name, ctype
