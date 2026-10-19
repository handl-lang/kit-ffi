//! Generic FFI dispatcher for handl.
//!
//! This crate provides [`FfiOpFunction`], a [`handl_platform::Function`]
//! implementation that dispatches handl calls to arbitrary C symbols loaded
//! via `dlopen` / `dlsym`. It plugs into the existing [`OpRegistry`] with no
//! engine changes.
//!
//! ## Pipeline
//!
//! 1. **Load** the C library via `dlopen` (cached by name — opened once,
//!    reused for the process lifetime).
//! 2. **Look up** the symbol via `dlsym` (cached by library + symbol).
//! 3. **Marshal** each `serde_json::Value` argument to a C type per its
//!    descriptor string.
//! 4. **Call** the C symbol via a cranelift-generated call stub
//!    (`cranelift_shim`) — no arity ceiling.
//! 5. **Unmarshal** the return (C type → `serde_json::Value`).
//! 6. **Error clause** — when `error_on_lt_zero` is set and the integer return
//!    is negative, extract `errno` (via `__errno_location()` on Linux) and
//!    return `Err(EvalError::Other(format!("errno: {errno}")))`.
//!
//! ## Marshalling descriptors
//!
//! | Descriptor | C type            | JSON source         |
//! |------------|-------------------|---------------------|
//! | `"i32"`    | `int32_t` (value) | number (range-checked) |
//! | `"i64"`    | `int64_t` (value) | number              |
//! | `"f64"`    | `double` (value)  | number              |
//! | `"cstr"`   | `const char *`    | string (NUL-checked)|
//! | `"ptr"`    | `void *`          | number (address)    |
//! | `"handle"` | `void *`          | `{"__resource__": {"ptr": N}}` (opaque handle, ADR-0083 §2.2) |
//! | `"void"`   | `void` (return)   | — (returns null)    |
//! | `"in_buf"` | `const void * + size_t` | `Bytes` (`{__bytes__:[...]}` or `[...]`) |
//! | `"cbytes"` | `const void * + size_t` | `Bytes` (raw, no NUL termination)        |
//! | `"in_ptr"` | `const void *`          | `Bytes` (bare pointer, no size arg)      |
//! | `"out_buf:N"` | `void * + size_t` | — (allocates N; returns `{ ret, buf }`) |
//!
//! Buffer descriptors (`in_buf` / `cbytes` / `out_buf:N`) require the `Bytes`
//! type (ADR-0078 Phase 2 / ADR-0083 §2.1). Struct descriptors remain
//! deferred.
//!
//! [`OpRegistry`]: handl_plugin_lib_handl::OpRegistry

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use handl_platform::{EvalContext, EvalError, Function, Module};
use libloading::Library;
use serde_json::{Map, Value as JsonValue, json};

use crate::cranelift_shim::resolve_shim;
use crate::error::last_errno;
use crate::marshal::{
    CArg, CRet, CRetKind, descriptor_consumes_handl_arg, marshal_call_args, marshal_read,
    return_kind, unmarshal_return, wrap_io_struct_return, wrap_outs_return,
};

mod cranelift_shim;
mod error;
mod marshal;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Global caches (process-lifetime)
// ---------------------------------------------------------------------------

/// Cache of opened libraries: library name → `libloading::Library`.
///
/// Libraries are opened once via `dlopen` and kept resident for the lifetime
/// of the process so that cached symbol pointers remain valid.
static LIBRARY_CACHE: OnceLock<Mutex<HashMap<String, Library>>> = OnceLock::new();

/// Cache of resolved symbols: (library, symbol) → raw symbol pointer.
///
/// The raw pointer is valid as long as the owning [`Library`] stays in
/// [`LIBRARY_CACHE`], which is never evicted.
static SYMBOL_CACHE: OnceLock<Mutex<HashMap<(String, String), SymbolPtr>>> = OnceLock::new();

/// A raw symbol pointer wrapper that is `Send` + `Sync`.
///
/// `*mut c_void` is not `Send` by default (raw pointers are never
/// auto-`Send`/`Sync`). The pointer is only dereferenced inside `unsafe`
/// call sites that hold no lock, so sharing it across threads is sound.
#[derive(Clone, Copy)]
struct SymbolPtr(*mut c_void);
unsafe impl Send for SymbolPtr {}
unsafe impl Sync for SymbolPtr {}

fn library_cache() -> &'static Mutex<HashMap<String, Library>> {
    LIBRARY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn symbol_cache() -> &'static Mutex<HashMap<(String, String), SymbolPtr>> {
    SYMBOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lock a cache mutex, recovering the guard when a previous holder panicked.
/// The caches only ever hold fully-inserted entries — a panic mid-operation
/// can at worst drop an in-flight insert, never corrupt the map — so the
/// poisoned state is safe to reuse. Without this, one panicking worker would
/// poison the process-wide cache and fail every subsequent FFI call on all
/// threads.
pub(crate) fn lock_cache<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Resolve a library by name, opening it once and caching the handle.
fn resolve_library(name: &str) -> Result<(), EvalError> {
    let mut cache = lock_cache(library_cache());
    if cache.contains_key(name) {
        return Ok(());
    }
    let lib = if name == "self" {
        // Special name: load the current process's symbol table so tests
        // can call `#[no_mangle] extern "C"` functions via the platform's
        // `this()` constructor (libloading 0.8: `os::unix::Library::this()`).
        #[cfg(unix)]
        {
            Library::from(libloading::os::unix::Library::this())
        }
        #[cfg(not(unix))]
        {
            return Err(EvalError::Other(
                "library \"self\" is not supported on this platform".to_string(),
            ));
        }
    } else {
        unsafe { Library::new(name) }
            .map_err(|e| EvalError::Other(format!("dlopen failed for {name}: {e}")))?
    };
    cache.insert(name.to_string(), lib);
    Ok(())
}

/// Resolve a symbol within a library, caching the raw pointer.
///
/// The library is opened (and cached) on first access; the symbol is looked
/// up via `dlsym` and its raw address cached under `(library, symbol)`.
fn resolve_symbol(library: &str, symbol: &str) -> Result<*mut c_void, EvalError> {
    let key = (library.to_string(), symbol.to_string());

    // Fast path: symbol already cached.
    {
        let cache = lock_cache(symbol_cache());
        if let Some(&SymbolPtr(ptr)) = cache.get(&key) {
            return Ok(ptr);
        }
    }

    // Ensure the library is loaded.
    resolve_library(library)?;

    // Look up the symbol and extract the raw pointer.
    let ptr = {
        let cache = lock_cache(library_cache());
        let lib = cache
            .get(library)
            .ok_or_else(|| EvalError::Other(format!("library not loaded: {library}")))?;
        let sym: libloading::Symbol<unsafe extern "C" fn()> = unsafe { lib.get(symbol.as_bytes()) }
            .map_err(|e| EvalError::Other(format!("dlsym failed for {symbol}: {e}")))?;
        unsafe { sym.try_as_raw_ptr() }
            .ok_or_else(|| EvalError::Other(format!("null symbol: {symbol}")))?
    };

    // Cache and return.
    let mut cache = lock_cache(symbol_cache());
    cache.insert(key, SymbolPtr(ptr));
    Ok(ptr)
}

// ---------------------------------------------------------------------------
// Call dispatch (Cranelift-generated trampolines)
// ---------------------------------------------------------------------------

/// Call a C function pointer with the given marshalled arguments.
///
/// Resolves (compiling on first use, then cached) a cranelift-generated call
/// stub for the flat signature `(ret_kind, classes(args))` and invokes it.
/// There is no arity ceiling — the stub loads each argument from the
/// int/float slot arrays and issues an indirect call with the exact C ABI
/// signature. See [`crate::cranelift_shim`] for the trampoline contract.
fn call_c(sym: *mut c_void, ret: CRetKind, args: &[CArg]) -> Result<CRet, EvalError> {
    let shim = resolve_shim(ret, args)?;
    // SAFETY: `shim` was compiled for exactly this signature (same arg
    // classes in the same order, same return kind); `sym` is the `dlsym`
    // result for the caller-declared signature.
    Ok(unsafe { shim.call(sym, args) })
}

// ---------------------------------------------------------------------------
// FfiOpFunction
// ---------------------------------------------------------------------------

/// A generic FFI dispatcher function implementing [`Function`].
///
/// Each call resolves the library (cached `dlopen`), resolves the symbol
/// (cached `dlsym`), marshals the JSON arguments to C types per
/// `arg_descriptors`, calls the C symbol, unmarshals the return per
/// `return_descriptor`, and optionally checks the error clause.
///
/// When `error_on_lt_zero` is `true` and the integer return value is
/// negative, the dispatcher reads `errno` synchronously and returns
/// `Err(EvalError::Other(format!("errno: {errno}")))` instead of the raw
/// return value.
///
/// ## Handle lifecycle (ADR-0083 §2.2)
///
/// When `return_descriptor` is `"handle"`, the raw C return pointer is
/// wrapped in the JSON debug form of a `TaggedValue::Resource`
/// (`{"__resource__": {"ptr": N}}`). The optional [`finalizer`](Self::finalizer)
/// field is the mechanism for attaching a cleanup function (e.g.
/// `libc::close`) to the resource so it is released automatically when the
/// handl value goes out of scope (RAII). Because the current dispatcher
/// returns `serde_json::Value` (which cannot carry a function pointer), the
/// finalizer is not yet wired into the return path — it defaults to `None`.
/// Once the dispatcher is upgraded to return `handl_value::TaggedValue`
/// directly, the finalizer will be attached to the `Resource` on return.
#[derive(Clone)]
pub struct FfiOpFunction {
    /// Library name passed to `dlopen` (e.g. `"libc.so.6"`).
    library: String,
    /// Symbol name passed to `dlsym` (e.g. `"getpid"`).
    symbol: String,
    /// Per-argument marshalling descriptors (e.g. `["i32", "cstr"]`).
    arg_descriptors: Vec<String>,
    /// Return-value descriptor (e.g. `"i32"`, `"void"`, `"handle"`).
    return_descriptor: String,
    /// When `true`, a negative integer return triggers an errno extraction.
    error_on_lt_zero: bool,
    /// Optional finalizer for `"handle"` return values (ADR-0083 §2.2).
    ///
    /// When `Some`, this function is attached to the `Resource` wrapping the
    /// C return pointer so the underlying resource is released when the
    /// handl value drops. Currently not wired into the JSON return path (see
    /// the type-level docs); set via [`with_finalizer`](Self::with_finalizer).
    finalizer: Option<unsafe extern "C" fn(*mut c_void)>,
}

impl FfiOpFunction {
    /// Create a new FFI op function.
    ///
    /// - `library` — library name for `dlopen` (e.g. `"libc.so.6"`).
    /// - `symbol` — symbol name for `dlsym` (e.g. `"getpid"`).
    /// - `arg_descriptors` — one descriptor string per argument.
    /// - `return_descriptor` — the return type descriptor.
    /// - `error_on_lt_zero` — enable the negative-return error clause.
    pub fn new(
        library: impl Into<String>,
        symbol: impl Into<String>,
        arg_descriptors: Vec<String>,
        return_descriptor: impl Into<String>,
        error_on_lt_zero: bool,
    ) -> Self {
        Self {
            library: library.into(),
            symbol: symbol.into(),
            arg_descriptors,
            return_descriptor: return_descriptor.into(),
            error_on_lt_zero,
            finalizer: None,
        }
    }

    /// Attach a finalizer to this FFI op function for `"handle"` return
    /// values (ADR-0083 §2.2 — finalizable values).
    ///
    /// The finalizer is a C-ABI cleanup function (e.g. `libc::close`) that
    /// will be called with the opaque handle pointer when the handl
    /// `Resource` value wrapping it goes out of scope. This is the RAII
    /// mechanism that prevents fd / mmap / socket leaks in long-running
    /// processes (the Phase 4 async-runtime prerequisite).
    ///
    /// # Safety
    ///
    /// The caller must ensure `finalizer` is safe to call with the opaque
    /// pointer returned by the C symbol and that it releases the underlying
    /// resource exactly once.
    pub unsafe fn with_finalizer(mut self, finalizer: unsafe extern "C" fn(*mut c_void)) -> Self {
        self.finalizer = Some(finalizer);
        self
    }

    /// Returns the finalizer currently attached to this op function, if any.
    pub fn finalizer(&self) -> Option<unsafe extern "C" fn(*mut c_void)> {
        self.finalizer
    }
}

impl Function for FfiOpFunction {
    fn call(&self, args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
        // Resolve symbol (cached dlopen + dlsym).
        let sym = resolve_symbol(&self.library, &self.symbol)?;

        // Marshal args. Buffer descriptors (`in_buf` / `cbytes` / `out_buf:N`)
        // expand to two C args each; `out_buf:N` consumes no handl arg. The
        // backing storage (CStrings, source Vec<u8>s, output buffers) is owned
        // by `marshaled` and stays alive for the duration of the C call.
        let marshaled = marshal_call_args(&self.arg_descriptors, args)?;
        if std::env::var("HNL_FFI_DEBUG").is_ok() {
            eprintln!(
                "[ffi] {}.{} args={:?} c_args={:?}",
                self.library, self.symbol, args, marshaled.c_args
            );
        }

        // Determine return kind and call.
        let ret_kind = return_kind(&self.return_descriptor)?;
        let ret = call_c(sym, ret_kind, &marshaled.c_args)?;

        // Error clause: negative integer return → extract errno.
        if self.error_on_lt_zero {
            if let CRet::Int(v) = &ret {
                if *v < 0 {
                    let errno = last_errno();
                    return Err(EvalError::Other(format!("errno: {errno}")));
                }
            }
        }

        // Unmarshal return, then wrap as `{ ret, buf }` / `{ ret, buf,
        // outs }` when out-slots or `out_struct` descriptors were used
        // (ADR-0083 §2.1).
        let raw_ret = unmarshal_return(&self.return_descriptor, &ret)?;
        if !marshaled.io_struct_idx.is_empty() {
            let ios: Vec<_> = marshaled
                .io_struct_idx
                .iter()
                .map(|&i| &marshaled.struct_args[i])
                .collect();
            Ok(wrap_io_struct_return(
                &raw_ret,
                &ios,
                &marshaled.out_slots,
                &marshaled.scalar_outs,
            ))
        } else {
            Ok(wrap_outs_return(
                &raw_ret,
                &marshaled.out_slots,
                &marshaled.scalar_outs,
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Host `ffi` module
// ---------------------------------------------------------------------------

fn require_str<'a>(arg: &'a JsonValue, what: &str) -> Result<&'a str, EvalError> {
    arg.as_str()
        .ok_or_else(|| EvalError::TypeMismatch(format!("{what} must be a string")))
}

fn require_array<'a>(arg: &'a JsonValue, what: &str) -> Result<&'a Vec<JsonValue>, EvalError> {
    arg.as_array()
        .ok_or_else(|| EvalError::TypeMismatch(format!("{what} must be an array")))
}

fn require_string_array(arg: &JsonValue, what: &str) -> Result<Vec<String>, EvalError> {
    arg.as_array()
        .ok_or_else(|| EvalError::TypeMismatch(format!("{what} must be an array of strings")))?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_str()
                .map(String::from)
                .ok_or_else(|| EvalError::TypeMismatch(format!("{what}[{i}] must be a string")))
        })
        .collect()
}

/// `ffi.call(lib, sym, all_args, descriptors, ret_kind, errno?) -> Value`
///
/// Immediately call the C symbol with the given descriptors.
pub struct FfiCall;

impl Function for FfiCall {
    fn call(&self, args: &[JsonValue], ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
        if args.len() < 5 {
            return Err(EvalError::TypeMismatch(
                "ffi.call(lib, sym, all_args, descriptors, ret_kind, errno?)".to_string(),
            ));
        }
        let lib = require_str(&args[0], "lib")?;
        let sym = require_str(&args[1], "sym")?;
        let all_args = require_array(&args[2], "all_args")?;
        let descriptors = require_string_array(&args[3], "descriptors")?;
        let ret_kind = require_str(&args[4], "ret_kind")?;
        let error_on_lt_zero = args.get(5).and_then(|v| v.as_bool()).unwrap_or(false);

        let op = FfiOpFunction::new(lib, sym, descriptors.clone(), ret_kind, error_on_lt_zero);
        op.call(all_args, ctx)
    }
}

/// `ffi.bind(lib, sym, bound_args, descriptors, ret_kind, errno?) -> Function`
///
/// Returns a first-class function value (a `FunctionValue`) that captures
/// the bound handl arguments. When called, it invokes `ffi.call` with the
/// bound prefix followed by the caller-supplied remaining arguments.
pub struct FfiBind;

impl Function for FfiBind {
    fn call(&self, args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
        if args.len() < 5 {
            return Err(EvalError::TypeMismatch(
                "ffi.bind(lib, sym, bound_args, descriptors, ret_kind, errno?)".to_string(),
            ));
        }
        let lib = require_str(&args[0], "lib")?;
        let sym = require_str(&args[1], "sym")?;
        let bound = require_array(&args[2], "bound_args")?;
        let descriptors = require_string_array(&args[3], "descriptors")?;
        let ret_kind = require_str(&args[4], "ret_kind")?;
        let error_on_lt_zero = args.get(5).and_then(|v| v.as_bool()).unwrap_or(false);

        let non_out_count = descriptors
            .iter()
            .filter(|d| descriptor_consumes_handl_arg(d))
            .count();
        if bound.len() > non_out_count {
            return Err(EvalError::TypeMismatch(format!(
                "bound_args has {} values but only {} consume a handl argument",
                bound.len(),
                non_out_count
            )));
        }
        let unbound_count = non_out_count - bound.len();

        // Build the handl expression `ffi.call(__lib, __sym, [...], ...)`.
        let mut arg_names: Vec<String> = bound
            .iter()
            .enumerate()
            .map(|(i, _)| format!("__b{i}"))
            .collect();
        for i in 0..unbound_count {
            arg_names.push(format!("__arg_{i}"));
        }
        let args_list = arg_names.join(", ");
        let body =
            format!("ffi.call(__lib, __sym, [{args_list}], __descriptors, __ret_kind, __errno)");

        // Captured constants for the closure.
        let mut captured = Vec::new();
        captured.push(json!("__lib"));
        captured.push(JsonValue::String(lib.to_string()));
        captured.push(json!("__sym"));
        captured.push(JsonValue::String(sym.to_string()));
        for (i, v) in bound.iter().enumerate() {
            captured.push(json!(format!("__b{i}")));
            captured.push(v.clone());
        }
        captured.push(json!("__descriptors"));
        captured.push(JsonValue::Array(
            descriptors
                .iter()
                .map(|s| JsonValue::String(s.clone()))
                .collect(),
        ));
        captured.push(json!("__ret_kind"));
        captured.push(JsonValue::String(ret_kind.to_string()));
        captured.push(json!("__errno"));
        captured.push(JsonValue::Bool(error_on_lt_zero));

        // Captured array is [[name, value], ...].
        let mut captured_pairs = Vec::new();
        let mut it = captured.into_iter();
        while let (Some(name), Some(value)) = (it.next(), it.next()) {
            let mut pair = Vec::new();
            pair.push(name);
            pair.push(value);
            captured_pairs.push(JsonValue::Array(pair));
        }

        // Unbound parameters.
        let mut params = Vec::new();
        for i in 0..unbound_count {
            let mut obj = Map::new();
            obj.insert("name".to_string(), JsonValue::String(format!("__arg_{i}")));
            obj.insert("schema".to_string(), JsonValue::Null);
            obj.insert("pattern".to_string(), JsonValue::Null);
            params.push(JsonValue::Object(obj));
        }

        let mut obj = Map::new();
        obj.insert("body".to_string(), JsonValue::String(body));
        obj.insert("params".to_string(), JsonValue::Array(params));
        obj.insert("captured".to_string(), JsonValue::Array(captured_pairs));
        Ok(JsonValue::Object(obj))
    }
}

/// `ffi.errno() -> Number`
///
/// Return the current C `errno` value for this thread. This must be called
/// immediately after the failing C call whose error is being checked.
pub struct FfiErrno;

impl Function for FfiErrno {
    fn call(&self, args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
        if !args.is_empty() {
            return Err(EvalError::TypeMismatch("ffi.errno()".to_string()));
        }
        Ok(json!(last_errno()))
    }
}

/// `ffi.load(lib) -> {"__tag__": "ok"|"err", "value": ...}`
///
/// Non-raising library probe: attempts `dlopen` for `lib` (cached) and
/// returns an ok/err tag instead of raising. This is the Result-pattern
/// way to test library availability — versioned sonames, optional
/// dependencies — without `error.try`.
pub struct FfiLoad;

impl Function for FfiLoad {
    fn call(&self, args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
        if args.len() != 1 {
            return Err(EvalError::TypeMismatch("ffi.load(lib)".to_string()));
        }
        let lib = require_str(&args[0], "lib")?;
        match resolve_library(lib) {
            Ok(()) => Ok(json!({"__tag__": "ok", "value": null})),
            Err(e) => Ok(json!({"__tag__": "err", "value": e.to_string()})),
        }
    }
}

/// `ffi.sym(lib, sym) -> {"__tag__": "ok"|"err", "value": addr|msg}`
///
/// Non-raising symbol probe: resolves `sym` in `lib` (`dlopen`ing the
/// library first if needed). On success the ok `value` is the symbol's
/// address; on failure the err `value` is the `dlsym`/`dlopen` message.
pub struct FfiSym;

impl Function for FfiSym {
    fn call(&self, args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
        if args.len() != 2 {
            return Err(EvalError::TypeMismatch("ffi.sym(lib, sym)".to_string()));
        }
        let lib = require_str(&args[0], "lib")?;
        let sym = require_str(&args[1], "sym")?;
        match resolve_symbol(lib, sym) {
            Ok(ptr) => Ok(json!({"__tag__": "ok", "value": ptr as i64})),
            Err(e) => Ok(json!({"__tag__": "err", "value": e.to_string()})),
        }
    }
}

/// `ffi.read(ptr, descriptor) -> Value`
///
/// Generic pointer dereference: reads memory at `ptr` (a number address
/// or `{"__resource__": {"ptr": N}}` handle) and decodes it per
/// `descriptor` — the same field grammar as struct descriptors
/// (`i8`..`u64`/`isize`/`usize` scalars, `ptr`, `bytes:N`, `arr:N:T`,
/// `rec:{...}`, `cstr`). This is the primitive that lets handl chase
/// out-pointer results (`T **res`), linked lists (`ai_next`), and
/// pointed-to payloads (`ai_addr`) — `ffi.call` marshals arguments, but
/// only `ffi.read` reaches memory C did not hand back by value.
pub struct FfiRead;

impl Function for FfiRead {
    fn call(&self, args: &[JsonValue], _ctx: &mut EvalContext) -> Result<JsonValue, EvalError> {
        if args.len() != 2 {
            return Err(EvalError::TypeMismatch(
                "ffi.read(ptr, descriptor)".to_string(),
            ));
        }
        let addr = args[0]
            .as_object()
            .and_then(|obj| obj.get("__resource__"))
            .and_then(|r| r.get("ptr"))
            .and_then(|v| v.as_i64())
            .or_else(|| args[0].as_i64())
            .ok_or_else(|| {
                EvalError::TypeMismatch(
                    "ffi.read ptr must be a number or resource handle".to_string(),
                )
            })?;
        if addr == 0 {
            return Err(EvalError::Other("ffi.read: null pointer".to_string()));
        }
        let desc = require_str(&args[1], "descriptor")?;
        marshal_read(addr as usize, desc)
    }
}

static FFI_BIND: FfiBind = FfiBind;
static FFI_CALL: FfiCall = FfiCall;
static FFI_ERRNO: FfiErrno = FfiErrno;
static FFI_LOAD: FfiLoad = FfiLoad;
static FFI_READ: FfiRead = FfiRead;
static FFI_SYM: FfiSym = FfiSym;

/// The host `ffi` module, exposing `ffi.bind`, `ffi.call`, `ffi.errno`,
/// `ffi.load`, `ffi.read`, and `ffi.sym`.
#[derive(Clone, Copy)]
pub struct FfiModule;

impl Module for FfiModule {
    fn name(&self) -> &str {
        "ffi"
    }

    fn function(&self, name: &str) -> Option<&dyn Function> {
        match name {
            "bind" => Some(&FFI_BIND),
            "call" => Some(&FFI_CALL),
            "errno" => Some(&FFI_ERRNO),
            "load" => Some(&FFI_LOAD),
            "read" => Some(&FFI_READ),
            "sym" => Some(&FFI_SYM),
            _ => None,
        }
    }

    fn functions(&self) -> Vec<(&str, &dyn Function)> {
        vec![
            ("bind", &FFI_BIND as &dyn Function),
            ("call", &FFI_CALL as &dyn Function),
            ("errno", &FFI_ERRNO as &dyn Function),
            ("load", &FFI_LOAD as &dyn Function),
            ("read", &FFI_READ as &dyn Function),
            ("sym", &FFI_SYM as &dyn Function),
        ]
    }
}
