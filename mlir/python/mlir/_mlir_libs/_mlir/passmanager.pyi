# Originally imported via:
#   stubgen {...} -m mlir._mlir_libs._mlir.passmanager
# Local modifications:
#   * Relative imports for cross-module references.
#   * Add __all__


from . import ir as _ir

__all__ = [
    "PassManager",
]

class PassManager:
    def __init__(self, context: _ir.Context | None = None) -> None: ...
    def _CAPICreate(self) -> object: ...
    def _testing_release(self) -> None: ...
    def enable_ir_printing(
        self,
        print_before_all: bool = False,
        print_after_all: bool = True,
        print_module_scope: bool = False,
        print_after_change: bool = False,
        print_after_failure: bool = False,
    ) -> None: ...
    def enable_reproducer_before_all(self, output_dir: str) -> None: ...
    def enable_verifier(self, enable: bool) -> None: ...
    @staticmethod
    def parse(pipeline: str, context: _ir.Context | None = None) -> PassManager: ...
    def run(self, module: _ir._OperationBase) -> None: ...
    @property
    def _CAPIPtr(self) -> object: ...
