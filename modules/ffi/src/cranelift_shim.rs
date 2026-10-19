//! Cranelift-generated FFI call stubs (ADR-0083 §4, Phase 8.1).
//!
//! This module replaces the former arity-0..=6 macro-generated transmute
//! dispatch with JIT-compiled call trampolines. For each distinct *flat*
//! C signature (a return kind plus one [`ArgClass`] per marshalled C
//! argument) a native stub is generated via cranelift and cached for the
//! process lifetime. Because the argument list is emitted as a loop over
//! the signature, there is **no arity ceiling**.
//!
//! ## Trampoline contract
//!
//! Every generated stub has the uniform C signature
//!
//! ```c
//! void shim(void *sym, const i64 *int_args, const f64 *float_args, i64 *out);
//! ```
//!
//! - `sym` is the raw `dlsym` pointer of the target C function.
//! - `int_args` holds all integer/pointer arguments (in argument order),
//!   one `i64` slot each.
//! - `float_args` holds all `f64` arguments (in argument order).
//! - `out` receives the result: the integer return value, the
//!   `f64::to_bits` of a float return, or `0` for `void` returns.
//!
//! Inside the stub the known-static signature selects which array and
//! offset each argument loads from, builds a cranelift [`Signature`] with
//! the exact C ABI parameter/return types, and performs an indirect call
//! through `sym`. The caller (`FfiOpFunction`) therefore sees one uniform
//! calling convention regardless of the target signature.
//!
//! Integer/pointer arguments are passed as `i64` (sound on x86-64 SysV and
//! AArch64, matching the previous dispatcher — a 32-bit `int` argument is
//! sign/zero-extended in its calling register). A 32-bit integer *return*
//! is sign-extended from `i32` to `i64` inside the stub, matching the
//! `CRetKind::Int32` semantics documented in [`crate::marshal`].

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use handl_platform::EvalError;

use crate::marshal::{CArg, CRet, CRetKind};

// ---------------------------------------------------------------------------
// Signature classification
// ---------------------------------------------------------------------------

/// The machine-level class of one marshalled C argument.
///
/// Only the class matters for code generation: `Int` covers every
/// integer/pointer descriptor (`i32`, `i64`, `ptr`, `cstr`, `handle`, and
/// both halves of the buffer-descriptor `(ptr, len)` pairs); `Float`
/// covers `f64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ArgClass {
    /// Integer or pointer argument, passed in an `i64` slot.
    Int,
    /// Double-precision float argument, passed in an `f64` slot.
    Float,
}

impl From<&CArg> for ArgClass {
    fn from(arg: &CArg) -> Self {
        match arg {
            CArg::Int(_) => ArgClass::Int,
            CArg::Float(_) => ArgClass::Float,
        }
    }
}

/// Cache key for generated stubs: the flat C signature only.
#[derive(PartialEq, Eq, Hash)]
struct ShimKey {
    ret: CRetKind,
    args: Vec<ArgClass>,
}

// ---------------------------------------------------------------------------
// The shim
// ---------------------------------------------------------------------------

/// The uniform trampoline signature every generated stub implements.
///
/// `unsafe extern "C" fn(void *sym, const i64 *int_args, const f64
/// *float_args, i64 *out)`
type ShimFn = unsafe extern "C" fn(*mut c_void, *const i64, *const f64, *mut i64);

/// A compiled call stub for one flat C signature.
///
/// Owns the [`JITModule`] so the generated code remains mapped and
/// executable for the shim's lifetime. Shims are cached process-wide (see
/// [`resolve_shim`]) so a given signature is JIT-compiled at most once.
pub(crate) struct CraneliftShim {
    /// Keeps the JIT'd machine code alive. Never read directly.
    _module: JITModule,
    /// Entry point of the generated trampoline.
    code: ShimFn,
    /// The return kind the stub produces (drives `out`-slot decoding).
    ret: CRetKind,
}

// The generated code is position-independent machine code; the JITModule
// owns the memory and the ShimFn pointer is a plain code address. Sharing
// the shim across threads is sound because the stub itself is stateless —
// all per-call data (arg arrays, out slot) is supplied by the caller.
unsafe impl Send for CraneliftShim {}
unsafe impl Sync for CraneliftShim {}

impl CraneliftShim {
    /// Compile a call stub for the given flat signature.
    ///
    /// `args` is one [`ArgClass`] per marshalled C argument (i.e. the
    /// *flattened* list after buffer descriptors have expanded to their
    /// `(ptr, len)` pairs).
    fn compile(ret: CRetKind, args: &[ArgClass]) -> Result<Self, EvalError> {
        let builder = JITBuilder::new(default_libcall_names())
            .map_err(|e| EvalError::Other(format!("cranelift JIT setup error: {e}")))?;
        let mut module = JITModule::new(builder);
        let ptr_ty = module.isa().pointer_type();

        // Stub signature: (sym, int_args, float_args, out) -> void.
        let mut stub_sig = module.make_signature();
        for _ in 0..4 {
            stub_sig.params.push(AbiParam::new(ptr_ty));
        }

        let func_id = module
            .declare_function("ffi_call_shim", Linkage::Local, &stub_sig)
            .map_err(|e| EvalError::Other(format!("cranelift declare error: {e}")))?;

        // Callee signature: the exact C ABI of the target function.
        let mut callee_sig = module.make_signature();
        for arg in args {
            callee_sig.params.push(AbiParam::new(match arg {
                ArgClass::Int => types::I64,
                ArgClass::Float => types::F64,
            }));
        }
        match ret {
            CRetKind::Int32 => callee_sig.returns.push(AbiParam::new(types::I32)),
            CRetKind::Int64 => callee_sig.returns.push(AbiParam::new(types::I64)),
            CRetKind::Float => callee_sig.returns.push(AbiParam::new(types::F64)),
            CRetKind::Void => {}
        }

        let mut ctx = module.make_context();
        ctx.func.signature = stub_sig;
        let callee_sig_ref = ctx.func.import_signature(callee_sig);

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);

        let sym = builder.block_params(block)[0];
        let int_args = builder.block_params(block)[1];
        let float_args = builder.block_params(block)[2];
        let out = builder.block_params(block)[3];

        // Load each argument from its class-specific array. The offsets are
        // static: each Int arg consumes one i64 slot, each Float one f64
        // slot, in argument order.
        let mut call_args: Vec<cranelift::prelude::Value> = Vec::with_capacity(args.len());
        let mut int_idx = 0i32;
        let mut float_idx = 0i32;
        for arg in args {
            match arg {
                ArgClass::Int => {
                    let v =
                        builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), int_args, int_idx * 8);
                    int_idx += 1;
                    call_args.push(v);
                }
                ArgClass::Float => {
                    let v = builder.ins().load(
                        types::F64,
                        MemFlags::trusted(),
                        float_args,
                        float_idx * 8,
                    );
                    float_idx += 1;
                    call_args.push(v);
                }
            }
        }

        // Indirect call through the raw symbol pointer.
        let call = builder.ins().call_indirect(callee_sig_ref, sym, &call_args);
        let results = builder.inst_results(call).to_vec();

        // Normalise the result into a single i64 slot written to `out`.
        let stored = match ret {
            // Sign-extend i32 -> i64 so a `-1` int return stays negative
            // (see CRetKind docs in marshal.rs).
            CRetKind::Int32 => builder.ins().sextend(types::I64, results[0]),
            CRetKind::Int64 => results[0],
            // Float returns are written back as raw f64 bits.
            CRetKind::Float => builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), results[0]),
            CRetKind::Void => builder.ins().iconst(types::I64, 0),
        };
        builder.ins().store(MemFlags::trusted(), stored, out, 0);
        builder.ins().return_(&[]);
        builder.finalize();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| EvalError::Other(format!("cranelift define error: {e}")))?;
        module
            .finalize_definitions()
            .map_err(|e| EvalError::Other(format!("cranelift finalize error: {e}")))?;

        let ptr = module.get_finalized_function(func_id);
        // SAFETY: `ptr` is the entry address of machine code generated with
        // exactly the stub signature declared above; `_module` keeps it
        // mapped for the shim's lifetime.
        let code = unsafe { std::mem::transmute::<*const u8, ShimFn>(ptr) };

        Ok(Self {
            _module: module,
            code,
            ret,
        })
    }

    /// Invoke the stub against a raw symbol pointer and marshalled args.
    ///
    /// Splits `c_args` into the int/float slot arrays expected by the stub
    /// (order preserved), calls the trampoline, and decodes the `out` slot
    /// per the stub's return kind.
    ///
    /// # Safety
    ///
    /// `sym` must be a valid function pointer whose C signature matches the
    /// signature this shim was compiled for. `c_args` must be the same list
    /// (in length and classes) that the shim was compiled from.
    pub(crate) unsafe fn call(&self, sym: *mut c_void, c_args: &[CArg]) -> CRet {
        let n_int = c_args.iter().filter(|a| matches!(a, CArg::Int(_))).count();
        let n_float = c_args.len() - n_int;
        let mut int_args: Vec<i64> = Vec::with_capacity(n_int);
        let mut float_args: Vec<f64> = Vec::with_capacity(n_float);
        for arg in c_args {
            match arg {
                CArg::Int(v) => int_args.push(*v),
                CArg::Float(v) => float_args.push(*v),
            }
        }

        let mut out: i64 = 0;
        // SAFETY: per this fn's contract — the stub loads at most n_int/n_float
        // slots (the counts it was compiled with) and writes one i64 to `out`.
        unsafe {
            (self.code)(sym, int_args.as_ptr(), float_args.as_ptr(), &mut out);
        }

        match self.ret {
            CRetKind::Int32 | CRetKind::Int64 => CRet::Int(out),
            CRetKind::Float => CRet::Float(f64::from_bits(out as u64)),
            CRetKind::Void => CRet::Void,
        }
    }
}

// ---------------------------------------------------------------------------
// Process-wide shim cache
// ---------------------------------------------------------------------------

/// Cache of compiled stubs: flat C signature → [`CraneliftShim`].
static SHIM_CACHE: OnceLock<Mutex<HashMap<ShimKey, Arc<CraneliftShim>>>> = OnceLock::new();

fn shim_cache() -> &'static Mutex<HashMap<ShimKey, Arc<CraneliftShim>>> {
    SHIM_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolve (compile-once, then reuse) the call stub for a flat signature.
///
/// `c_args` supplies both the signature (via each arg's [`ArgClass`]) and,
/// implicitly, the arity — buffer descriptors have already been expanded by
/// the marshaller, so each entry maps to exactly one C argument slot.
pub(crate) fn resolve_shim(
    ret: CRetKind,
    c_args: &[CArg],
) -> Result<Arc<CraneliftShim>, EvalError> {
    let key = ShimKey {
        ret,
        args: c_args.iter().map(ArgClass::from).collect(),
    };

    // Fast path: stub already compiled.
    {
        let cache = crate::lock_cache(shim_cache());
        if let Some(shim) = cache.get(&key) {
            return Ok(Arc::clone(shim));
        }
    }

    // Compile outside the lock so a slow compile doesn't block other
    // signature lookups. A duplicate compile under a race is harmless —
    // the cache insert just overwrites.
    let shim = Arc::new(CraneliftShim::compile(ret, &key.args)?);

    let mut cache = crate::lock_cache(shim_cache());
    Ok(Arc::clone(
        cache.entry(key).or_insert_with(|| Arc::clone(&shim)),
    ))
}
