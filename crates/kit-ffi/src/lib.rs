//! `handl-kit-ffi` — the foreign-function-interface kit.
//!
//! Extracted from `handl-kit-canonical`: the `ffi` handl module
//! (`lib-ffi`'s `ffi.hnl` surface) plus its native op wiring. The module
//! always resolves; the `ffi` Cargo feature (on by default) decides
//! whether the ops are real (`handl-plugin-ffi`) or the
//! graceful-degradation stubs (ADR-0128 §0.9).

use handl_kit_core::{KitResolver, ModuleCtor, ModuleEntry, build_type_module};
use handl_plugin_lib_handl::{HandlModule, OpRegistry};

#[cfg(feature = "ffi")]
use handl_plugin_ffi::{FfiBind, FfiCall, FfiErrno, FfiLoad, FfiRead, FfiSym};

// ---------------------------------------------------------------------------
// ffi — foreign-function interface package (`ffi.hnl` + native ops)
// ---------------------------------------------------------------------------

const FFI_HNL: &str = handl_lib_ffi::FFI_HNL;
const FFI_FUNCS: &[&str] = &[
    "Errno", "Handle", "call", "bind", "errno", "load", "read", "sym",
];
const FFI_OPS: &[&str] = &[
    "ffi.call",
    "ffi.bind",
    "ffi.errno",
    "ffi.load",
    "ffi.read",
    "ffi.sym",
];

#[cfg(feature = "ffi")]
fn ffi_op_registry() -> OpRegistry {
    let mut registry = OpRegistry::new();
    registry.register("ffi.call", std::sync::Arc::new(FfiCall));
    registry.register("ffi.bind", std::sync::Arc::new(FfiBind));
    registry.register("ffi.errno", std::sync::Arc::new(FfiErrno));
    registry.register("ffi.load", std::sync::Arc::new(FfiLoad));
    registry.register("ffi.read", std::sync::Arc::new(FfiRead));
    registry.register("ffi.sym", std::sync::Arc::new(FfiSym));
    registry
}

/// Stub `ffi.*` ops for builds without the `ffi` feature (ADR-0128 §0.9).
///
/// The `ffi` module stays registered and resolvable — what changes is the
/// ops' behavior: capability ops (`call`/`bind`/`read`/`errno`) raise a
/// clear "not compiled in" error, while the non-raising probes (`load`/
/// `sym`) keep their `{__tag__: "err", value: msg}` contract so soname/
/// versioned probes in dependent packages (`unicode_c`, `intl_c`,
/// `entropy`) degrade to their fallback paths instead of propagating.
#[cfg(not(feature = "ffi"))]
mod ffi_stubs {
    use handl_platform::{EvalContext, EvalError, Function};
    use serde_json::{Value as JsonValue, json};

    /// The reason every stub op reports.
    const DISABLED: &str =
        "ffi unavailable: handl-kit-ffi was built without the `ffi` feature";

    /// Capability op stub (`ffi.call`/`ffi.bind`/`ffi.read`/`ffi.errno`):
    /// cannot fabricate a result, so it raises.
    pub struct FfiUnsupported(pub &'static str);

    impl Function for FfiUnsupported {
        fn call(&self, _args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
            Err(EvalError::Other(format!("{}: {DISABLED}", self.0)))
        }
    }

    /// Non-raising probe stub (`ffi.load`/`ffi.sym`): preserves the
    /// `{"__tag__": "err", "value": msg}` return contract so `is_ok(...)`
    /// probes in `.hnl` packages keep working.
    pub struct FfiProbeUnsupported(pub &'static str);

    impl Function for FfiProbeUnsupported {
        fn call(&self, _args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
            Ok(json!({"__tag__": "err", "value": format!("{}: {DISABLED}", self.0)}))
        }
    }
}

#[cfg(not(feature = "ffi"))]
fn ffi_op_registry() -> OpRegistry {
    use ffi_stubs::{FfiProbeUnsupported, FfiUnsupported};
    let mut registry = OpRegistry::new();
    registry.register("ffi.call", std::sync::Arc::new(FfiUnsupported("ffi.call")));
    registry.register("ffi.bind", std::sync::Arc::new(FfiUnsupported("ffi.bind")));
    registry.register("ffi.errno", std::sync::Arc::new(FfiUnsupported("ffi.errno")));
    registry.register("ffi.load", std::sync::Arc::new(FfiProbeUnsupported("ffi.load")));
    registry.register("ffi.read", std::sync::Arc::new(FfiUnsupported("ffi.read")));
    registry.register("ffi.sym", std::sync::Arc::new(FfiProbeUnsupported("ffi.sym")));
    registry
}

/// Build the partial [`HandlModule`] for `ffi`: the authored `ffi.hnl`
/// surface with `call`, `bind`, and `errno` injected as native ops.
fn build_ffi_module() -> HandlModule {
    HandlModule::new("ffi", FFI_HNL, FFI_FUNCS)
        .with_dependency(build_type_module())
        .with_ops(FFI_OPS, std::sync::Arc::new(ffi_op_registry()))
}

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------

/// The kit's module table — just `ffi`.
pub const MODULES: &[(&'static str, ModuleCtor)] = &[("ffi", |_: &KitResolver, e| {
    ModuleEntry::Handl(build_ffi_module().with_eval_engine(e))
})];

/// Register the `ffi` module into the shared kit table. `Once`-safe.
pub fn register() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| handl_kit_core::register_modules(MODULES));
}
