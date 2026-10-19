//! Marshalling between `serde_json::Value` and C ABI types.
//!
//! The dispatcher marshals each handl argument to a C type per a descriptor
//! string, calls the C symbol, and then unmarshals the return value back to a
//! JSON value.
//!
//! ## Marshalling descriptors
//!
//! | Descriptor     | C type                  | JSON source          |
//! |----------------|-------------------------|----------------------|
//! | `"i32"`        | `int32_t` (value)       | number (range-checked) |
//! | `"i64"`        | `int64_t` (value)       | number               |
//! | `"u8"`-`"u64"` | `uint8_t`..`uint64_t`   | number (range-checked) |
//! | `"f64"`        | `double` (value)        | number               |
//! | `"cstr"`       | `const char *`          | string (NUL-checked) |
//! | `"ptr"`        | `void *`                | number (address)     |
//! | `"handle"`     | `void *`                | `{"__resource__": {"ptr": N}}` (opaque handle, ADR-0083 §2.2) |
//! | `"void"`       | `void` (return)         | — (returns null)     |
//! | `"in_buf"`     | `const void * + size_t` | `Bytes`              |
//! | `"cbytes"`     | `const void * + size_t` | `Bytes`              |
//! | `"in_ptr"`     | `const void *`          | `Bytes`              |
//! | `"out_buf:N"`  | `void * + size_t`       | — (allocates N)      |
//! | `"out_ptr:N"`  | `void *`                | — (allocates N)      |
//! | `"io_buf:N"`   | `void * + size_t *`     | — (allocates N)      |
//! | `"io_scalar:T"`| `T *` (value-result)    | number (seed value)  |
//! | `"in_struct:{…}"`  | `const T *` (built) | record             |
//! | `"io_struct:{…}"`  | `T *` (filled)    | — (allocates struct) |
//! | `"out_struct:{…}"` | `T *` (filled)    | — (= io_struct)      |
//!
//! Struct field specs are `name:kind[@offset[:slotwidth]]` — explicit
//! offsets/sizes carry the backend's real target layout (libclang); without
//! them fields lay out sequentially. Kinds: scalars (`i8`..`u64`,
//! `isize`, `usize`), `bytes` (Bytes → ptr), `buf:N` (alloc → ptr),
//! `iov`/`iovbuf:K:M` (iovec arrays), `ptr` (raw pointer slot),
//! `arr:N:T` (inline array), `rec:{...}` (inline nested struct).
//!
//! `in_buf` / `cbytes` / `in_ptr` / `out_buf:N` / `out_ptr:N` / `io_buf:N`
//! are the buffer descriptors introduced with the `Bytes` type (ADR-0078
//! Phase 2 / ADR-0083 §2.1). `in_buf` and `cbytes` each expand to **two** C
//! arguments — `(ptr, len)` — and consume one handl `Bytes` argument.
//! `out_buf:N` allocates `N` bytes, passes `(ptr, N)` as two C arguments,
//! consumes **no** handl argument, and produces an out-slot trimmed to the
//! `ret` byte count. `out_ptr:N` allocates `N` bytes and passes the bare
//! pointer as **one** C argument (e.g. `int sv[2]` in `socketpair`) — the
//! slot is the full `N` bytes. `in_ptr` is the input counterpart: the
//! `Bytes` pointer alone as **one** C argument, for signatures whose
//! length travels separately or with a non-`size_t` width (e.g. ICU's
//! `UChar * + int32_t` pairs). `io_buf:N` allocates an `N`-byte buffer
//! plus a `size_t` cell initialised to `N` and passes `(ptr, &len)` — the
//! POSIX value-result pattern (`sockaddr * + socklen_t *` in `accept`,
//! `getsockname`, `recvfrom`); the slot is trimmed to the length the
//! callee wrote back.
//!
//! The return becomes `{ ret, buf, outs }`: `buf` is the first out-slot
//! (kept for the common single-out case) and `outs` lists every out-slot
//! in descriptor order.
//!
//! `Bytes` values are carried as `serde_json::Value` using the
//! `{"__bytes__": [n, ...]}` encoding (matching `TaggedValue::Bytes` in
//! `handl-value`). For input, a plain JSON array of numbers is also accepted
//! as a convenience.

use std::ffi::{CStr, CString, c_char};

use handl_platform::EvalError;
use serde_json::{Map, Value as JsonValue};

// ---------------------------------------------------------------------------
// Intermediate C-side representations
// ---------------------------------------------------------------------------

/// A marshalled C argument ready to be passed across the FFI boundary.
///
/// All integer and pointer descriptors (`i32`, `i64`, `ptr`, `cstr`) are
/// widened to `i64`. This is sound on x86-64 SysV and AArch64 where a 32-bit
/// integer argument is sign/zero-extended in its calling register, so passing
/// the full 64-bit slot is equivalent. Floating-point values stay `f64`
/// because they travel in separate (XMM) registers and must not be
/// bit-cast to an integer slot.
#[derive(Clone, Copy, Debug)]
pub(crate) enum CArg {
    /// An integer or pointer argument, widened to 64 bits.
    Int(i64),
    /// A double-precision floating-point argument.
    Float(f64),
}

/// The *kind* of return value expected from the C function.
///
/// This drives which function-pointer signature the dispatcher transmutes the
/// raw symbol to. Crucially, `Int32` and `Int64` produce *different* return
/// types in the transmuted signature: on x86-64 a function returning `int`
/// only writes the lower 32 bits of `rax` (the upper 32 are zeroed, **not**
/// sign-extended), so reading the full 64-bit `rax` as `i64` would turn a
/// `-1` `int` return into `4_294_967_295`. Reading it as `i32` and then
/// sign-extending to `i64` gives the correct `-1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CRetKind {
    /// 32-bit integer return (read as `i32`, sign-extended to `i64`).
    Int32,
    /// 64-bit integer or pointer return (read as `i64`).
    Int64,
    /// Double-precision floating-point return.
    Float,
    /// No return value.
    Void,
}

/// A value returned from a C function.
pub(crate) enum CRet {
    /// Integer or pointer return.
    Int(i64),
    /// Floating-point return.
    Float(f64),
    /// Void return.
    Void,
}

// ---------------------------------------------------------------------------
// Marshalling: JSON -> C
// ---------------------------------------------------------------------------

/// Read a JSON value as an `i64`, accepting integral floats (`18080.0`).
/// Handl ops like `char.code_point_at` produce `f64` numbers, so integer
/// values frequently arrive at the FFI boundary in float form — the same
/// tolerance `plugin-lib-char-native::require_int` applies.
fn as_integral_i64(value: &JsonValue) -> Option<i64> {
    value.as_i64().or_else(|| {
        value
            .as_f64()
            .filter(|n| n.fract() == 0.0 && n.is_finite())
            .map(|n| n as i64)
    })
}

/// Marshal a single JSON argument to a [`CArg`] per its descriptor string.
///
/// For `cstr` descriptors the produced [`CString`] is pushed into
/// `keep_alive` so it lives until the C call returns; the returned [`CArg`]
/// only holds the raw pointer value.
pub(crate) fn marshal_arg(
    descriptor: &str,
    value: &JsonValue,
    keep_alive: &mut Vec<CString>,
) -> Result<CArg, EvalError> {
    match descriptor {
        "i32" => {
            let n = as_integral_i64(value)
                .ok_or_else(|| EvalError::TypeMismatch("expected number for i32".to_string()))?;
            let v = i32::try_from(n)
                .map_err(|_| EvalError::TypeMismatch(format!("i32 overflow: {n}")))?;
            Ok(CArg::Int(v as i64))
        }
        "i64" => {
            let n = as_integral_i64(value)
                .ok_or_else(|| EvalError::TypeMismatch("expected number for i64".to_string()))?;
            Ok(CArg::Int(n))
        }
        // Unsigned scalars (`u8`/`u16`/`u32`/`u64` — e.g. zlib's `uLong`,
        // `uInt`): range-checked, widened to `i64` like every other integer
        // descriptor.
        "u8" | "u16" | "u32" | "u64" => {
            let n = as_integral_i64(value).ok_or_else(|| {
                EvalError::TypeMismatch(format!("expected number for {descriptor}"))
            })?;
            let max = match descriptor {
                "u8" => u8::MAX as i64,
                "u16" => u16::MAX as i64,
                "u32" => u32::MAX as i64,
                _ => u64::MAX as i64,
            };
            if n < 0 || (descriptor != "u64" && n > max) {
                return Err(EvalError::TypeMismatch(format!(
                    "{descriptor} out of range: {n}"
                )));
            }
            Ok(CArg::Int(n))
        }
        "f64" => {
            let n = value
                .as_f64()
                .ok_or_else(|| EvalError::TypeMismatch("expected number for f64".to_string()))?;
            Ok(CArg::Float(n))
        }
        "cstr" => {
            let s = value.as_str().ok_or_else(|| {
                EvalError::TypeMismatch(format!(
                    "expected string for cstr, got {}",
                    &serde_json::to_string(value)
                        .unwrap_or_default()
                        .chars()
                        .take(120)
                        .collect::<String>()
                ))
            })?;
            let c = CString::new(s)
                .map_err(|_| EvalError::TypeMismatch("embedded NUL byte in cstr".to_string()))?;
            let ptr = c.as_ptr();
            keep_alive.push(c);
            Ok(CArg::Int(ptr as i64))
        }
        "ptr" => {
            let n = as_integral_i64(value)
                .ok_or_else(|| EvalError::TypeMismatch("expected number for ptr".to_string()))?;
            Ok(CArg::Int(n))
        }
        // `handle` accepts a `{"__resource__": {"ptr": N}}` JSON value (the
        // debug form of `TaggedValue::Resource`) and passes the raw pointer
        // through to the C call. A bare number is also accepted for
        // convenience (e.g. a POSIX fd passed as an i32).
        "handle" => {
            if let Some(ptr_n) = value
                .as_object()
                .and_then(|obj| obj.get("__resource__"))
                .and_then(|r| r.get("ptr"))
                .and_then(as_integral_i64)
            {
                return Ok(CArg::Int(ptr_n));
            }
            let n = as_integral_i64(value).ok_or_else(|| {
                EvalError::TypeMismatch("expected resource or number for handle".to_string())
            })?;
            Ok(CArg::Int(n))
        }
        other => Err(EvalError::TypeMismatch(format!(
            "unknown arg descriptor: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Buffer descriptors: in_buf / cbytes / out_buf:N (ADR-0083 §2.1)
// ---------------------------------------------------------------------------

/// How an out-slot's bytes are trimmed when read back after the call.
pub(crate) enum OutTrim {
    /// First `ret` bytes of the buffer (`out_buf:N` — `recv`/`read` style,
    /// where the C return is the byte count). Empty when `ret < 0`.
    ByRet,
    /// The full `N` bytes (`out_ptr:N` — fixed-size out-params like
    /// `int sv[2]`).
    Full,
    /// First `*len_cell` bytes (`io_buf:N` — POSIX value-result, the
    /// callee writes the actual length back through the pointer).
    ByLenCell,
}

/// An allocated output buffer for `out_buf:N` / `out_ptr:N` / `io_buf:N`.
///
/// Owns the backing `Vec<u8>` (and the `size_t` cell for `io_buf`) so it
/// stays valid across the C call and is available for the post-call
/// `{ ret, buf, outs }` extraction.
pub(crate) struct OutSlot {
    /// The allocated buffer (zero-initialized). Valid until this is dropped.
    pub buf: Vec<u8>,
    /// The declared capacity `N` (== `buf.len()`).
    pub len: usize,
    /// Trim rule applied when reading `buf` back.
    pub trim: OutTrim,
    /// `io_buf` only: the `size_t` cell whose address is passed to C,
    /// initialised to `N` (capacity in, actual length out). Empty for the
    /// other kinds.
    pub len_cell: Vec<u8>,
}

/// An allocated scalar out-cell for `io_scalar:TYPE` — the caller seeds the
/// initial value, C writes through the pointer, the cell reads back as an
/// integer in the `vals` out-list.
pub(crate) struct ScalarSlot {
    /// The cell bytes (width = the scalar type's size).
    pub cell: Vec<u8>,
    /// Whether the scalar is signed (sub-64-bit negatives sign-extend).
    pub signed: bool,
}

impl ScalarSlot {
    fn read(&self) -> i64 {
        read_scalar(&self.cell, 0, self.cell.len(), self.signed)
    }
}

/// The result of marshalling a whole argument list.
///
/// `c_args` is the flat list of C arguments to pass (buffer descriptors
/// expand to two entries each). `keep_alive` / `byte_keep_alive` keep the
/// backing storage alive for the duration of the C call. `out_bufs` records
/// any `out_buf:N` slots so the return can be wrapped as `{ ret, buf }`.
pub(crate) struct MarshalResult {
    pub c_args: Vec<CArg>,
    /// `CString`s owned by the marshaller (for `cstr` args).
    pub keep_alive: Vec<CString>,
    /// Source `Vec<u8>`s for `in_buf` / `cbytes` args (C reads from these).
    pub byte_keep_alive: Vec<Vec<u8>>,
    /// Allocated output buffers for `out_buf:N` / `out_ptr:N` / `io_buf:N`.
    pub out_slots: Vec<OutSlot>,
    /// Allocated scalar cells for `io_scalar:TYPE` descriptors.
    pub scalar_outs: Vec<ScalarSlot>,
    /// Built structs for `in_struct:{...}` / `io_struct:{...}` descriptors.
    /// `io_struct` entries are read back into records after the call.
    pub struct_args: Vec<StructArg>,
    /// Indexes into `struct_args` that are `io_struct` (read-back) entries.
    pub io_struct_idx: Vec<usize>,
}

/// Decode a JSON `Bytes` value into a `Vec<u8>`.
///
/// Accepts either the canonical `{"__bytes__": [n, ...]}` encoding (matching
/// `TaggedValue::Bytes`) or, as a convenience, a plain JSON array of numbers.
/// Each element must be an integer in `0..=255`; out-of-range or non-integer
/// elements are a marshal error rather than a silent truncation.
pub(crate) fn json_to_bytes(value: &JsonValue) -> Result<Vec<u8>, EvalError> {
    let arr: &[JsonValue] = if let Some(obj) = value.as_object() {
        match obj.get("__bytes__").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => {
                return Err(EvalError::TypeMismatch(
                    "expected Bytes ({__bytes__: [numbers]})".to_string(),
                ));
            }
        }
    } else if let Some(a) = value.as_array() {
        a
    } else {
        return Err(EvalError::TypeMismatch(
            "expected Bytes value (object or array)".to_string(),
        ));
    };
    let mut out = Vec::with_capacity(arr.len());
    for el in arr {
        let n = as_integral_i64(el).ok_or_else(|| {
            EvalError::TypeMismatch("Bytes array must contain numbers".to_string())
        })?;
        let b = u8::try_from(n)
            .map_err(|_| EvalError::TypeMismatch(format!("byte out of range 0..=255: {n}")))?;
        out.push(b);
    }
    Ok(out)
}

/// Encode a byte slice as the canonical `{"__bytes__": [n, ...]}` JSON value.
pub(crate) fn bytes_to_json(bytes: &[u8]) -> JsonValue {
    let arr: Vec<JsonValue> = bytes.iter().map(|&b| JsonValue::from(b as i64)).collect();
    let mut map = Map::new();
    map.insert("__bytes__".to_string(), JsonValue::Array(arr));
    JsonValue::Object(map)
}

/// Marshal a full argument list, expanding buffer descriptors.
///
/// Unlike [`marshal_arg`] (which handles a single C argument), this walks the
/// whole descriptor list because `in_buf` / `cbytes` / `out_buf:N` each
/// expand to **two** C arguments. `out_buf:N` consumes no handl argument; the
/// other descriptors consume exactly one. The number of handl arguments must
/// equal the number of descriptors that consume one.
pub(crate) fn marshal_call_args(
    descriptors: &[String],
    args: &[JsonValue],
) -> Result<MarshalResult, EvalError> {
    let mut res = MarshalResult {
        c_args: Vec::with_capacity(descriptors.len() * 2),
        keep_alive: Vec::with_capacity(descriptors.len()),
        byte_keep_alive: Vec::with_capacity(descriptors.len()),
        out_slots: Vec::new(),
        scalar_outs: Vec::new(),
        struct_args: Vec::new(),
        io_struct_idx: Vec::new(),
    };
    let mut arg_idx = 0usize;
    for desc in descriptors {
        if desc == "in_buf" || desc == "cbytes" {
            let value = args
                .get(arg_idx)
                .ok_or_else(|| EvalError::TypeMismatch(format!("missing argument for {desc}")))?;
            arg_idx += 1;
            let bytes = json_to_bytes(value)?;
            let ptr = bytes.as_ptr() as i64;
            let len = bytes.len() as i64;
            // Move `bytes` into keep-alive *after* capturing the pointer; the
            // Vec's heap allocation is stable across the move.
            res.byte_keep_alive.push(bytes);
            res.c_args.push(CArg::Int(ptr));
            res.c_args.push(CArg::Int(len));
        } else if desc == "in_ptr" {
            let value = args.get(arg_idx).ok_or_else(|| {
                EvalError::TypeMismatch("missing argument for in_ptr".to_string())
            })?;
            arg_idx += 1;
            let bytes = json_to_bytes(value)?;
            let ptr = bytes.as_ptr() as i64;
            res.byte_keep_alive.push(bytes);
            res.c_args.push(CArg::Int(ptr));
        } else if let Some(n_str) = desc.strip_prefix("out_buf:") {
            let n = parse_size("out_buf", n_str)?;
            let mut buf = vec![0u8; n];
            let ptr = buf.as_mut_ptr() as i64;
            res.c_args.push(CArg::Int(ptr));
            res.c_args.push(CArg::Int(n as i64));
            res.out_slots.push(OutSlot {
                buf,
                len: n,
                trim: OutTrim::ByRet,
                len_cell: Vec::new(),
            });
        } else if let Some(n_str) = desc.strip_prefix("out_ptr:") {
            let n = parse_size("out_ptr", n_str)?;
            let mut buf = vec![0u8; n];
            let ptr = buf.as_mut_ptr() as i64;
            res.c_args.push(CArg::Int(ptr));
            res.out_slots.push(OutSlot {
                buf,
                len: n,
                trim: OutTrim::Full,
                len_cell: Vec::new(),
            });
        } else if let Some(n_str) = desc.strip_prefix("io_buf:") {
            let n = parse_size("io_buf", n_str)?;
            let mut buf = vec![0u8; n];
            // Value-result cell: capacity in, actual length out.
            let mut len_cell = n.to_ne_bytes().to_vec();
            let ptr = buf.as_mut_ptr() as i64;
            let len_ptr = len_cell.as_mut_ptr() as i64;
            res.c_args.push(CArg::Int(ptr));
            res.c_args.push(CArg::Int(len_ptr));
            res.out_slots.push(OutSlot {
                buf,
                len: n,
                trim: OutTrim::ByLenCell,
                len_cell,
            });
        } else if let Some(ty) = desc.strip_prefix("io_scalar:") {
            // Value-result scalar: caller seeds the cell, C writes back.
            let size = scalar_size(ty)?;
            let value = args.get(arg_idx).and_then(as_integral_i64).ok_or_else(|| {
                EvalError::TypeMismatch(format!("missing int argument for {desc}"))
            })?;
            arg_idx += 1;
            let mut cell = vec![0u8; size];
            write_int(&mut cell, 0, size, value);
            let ptr = cell.as_mut_ptr() as i64;
            res.c_args.push(CArg::Int(ptr));
            res.scalar_outs.push(ScalarSlot {
                cell,
                signed: is_signed_scalar(ty),
            });
        } else if let Some(inner) = desc
            .strip_prefix("in_struct:")
            .or_else(|| desc.strip_prefix("io_struct:"))
            .or_else(|| desc.strip_prefix("out_struct:"))
        {
            // `out_struct` is `io_struct` minus the buffer-ref field kinds:
            // allocate the struct, let the callee fill it, read it back.
            let is_io = !desc.starts_with("in_struct:");
            let inner = inner
                .strip_prefix('{')
                .and_then(|s| s.strip_suffix('}'))
                .ok_or_else(|| {
                    EvalError::TypeMismatch(format!("struct descriptor must be {{...}}: {desc}"))
                })?;
            let record = if is_io {
                None
            } else {
                let value = args.get(arg_idx).ok_or_else(|| {
                    EvalError::TypeMismatch(format!("missing argument for {desc}"))
                })?;
                arg_idx += 1;
                Some(value)
            };
            let st = marshal_struct_arg(inner, record)?;
            let ptr = st.buf.as_ptr() as i64;
            res.c_args.push(CArg::Int(ptr));
            if is_io {
                res.io_struct_idx.push(res.struct_args.len());
            }
            res.struct_args.push(st);
        } else {
            let value = args
                .get(arg_idx)
                .ok_or_else(|| EvalError::TypeMismatch(format!("missing argument for {desc}")))?;
            arg_idx += 1;
            let arg = marshal_arg(desc, value, &mut res.keep_alive)?;
            res.c_args.push(arg);
        }
    }
    if arg_idx != args.len() {
        return Err(EvalError::TypeMismatch(format!(
            "expected {arg_idx} handl args, got {}",
            args.len()
        )));
    }
    Ok(res)
}

fn parse_size(kind: &str, n_str: &str) -> Result<usize, EvalError> {
    n_str
        .parse()
        .map_err(|_| EvalError::TypeMismatch(format!("invalid {kind} size: {n_str:?}")))
}

/// Read an out-slot back as a `Bytes` JSON value, trimmed per its kind.
fn slot_json(s: &OutSlot, ret_i64: i64) -> JsonValue {
    let take = match s.trim {
        OutTrim::ByRet => {
            if ret_i64 < 0 {
                0
            } else {
                (ret_i64 as usize).min(s.len)
            }
        }
        OutTrim::Full => s.len,
        OutTrim::ByLenCell => {
            let n = usize::from_ne_bytes(
                s.len_cell
                    .as_slice()
                    .try_into()
                    .unwrap_or([0u8; std::mem::size_of::<usize>()]),
            );
            n.min(s.len)
        }
    };
    bytes_to_json(&s.buf[..take])
}

/// Wrap a raw C return as a `{ ret, buf, outs }` record when out-slots
/// were used.
///
/// `raw_ret` is the already-unmarshalled C return value. `outs` lists
/// every out-slot in descriptor order, each trimmed per its kind
/// (`ByRet` clamps to the return byte count, `Full` keeps all `N` bytes,
/// `ByLenCell` reads the callee's value-result length). `buf` aliases the
/// first slot for the common single-out case.
pub(crate) fn wrap_outs_return(
    raw_ret: &JsonValue,
    out_slots: &[OutSlot],
    scalar_outs: &[ScalarSlot],
) -> JsonValue {
    if out_slots.is_empty() && scalar_outs.is_empty() {
        return raw_ret.clone();
    }
    let ret_i64 = raw_ret.as_i64().unwrap_or(0);
    let mut map = Map::new();
    map.insert("ret".to_string(), raw_ret.clone());
    if !out_slots.is_empty() {
        let outs: Vec<JsonValue> = out_slots.iter().map(|s| slot_json(s, ret_i64)).collect();
        map.insert("buf".to_string(), outs[0].clone());
        map.insert("outs".to_string(), JsonValue::Array(outs));
    }
    if !scalar_outs.is_empty() {
        let vals: Vec<JsonValue> = scalar_outs
            .iter()
            .map(|s| JsonValue::from(s.read()))
            .collect();
        map.insert("vals".to_string(), JsonValue::Array(vals));
    }
    JsonValue::Object(map)
}

// ---------------------------------------------------------------------------
// Struct descriptors: in_struct / io_struct
//
// `in_struct:{f:kind,...}` consumes one Handl record arg and *builds* a C
// struct whose pointer fields reference sibling buffers kept alive for the
// call — the `msghdr` pattern (`sendmsg`). `io_struct:{...}` consumes no
// Handl arg and allocates everything — the recv side (`recvmsg`) — and the
// fields are read back into a record after the call.
//
// Field kinds: scalar integers (`i8`..`u64`, `isize`, `usize`), `bytes`
// (Bytes arg → pointer), `iov` (List<Bytes> arg → iovec array), `buf:N`
// (allocate N bytes → pointer, read back as Bytes), `iovbuf:K:M` (allocate
// K buffers of M bytes + iovec array, read back as List<Bytes> trimmed by
// the cumulative `ret` byte count). A scalar field the record does not
// provide and whose name ends in `len` is auto-filled with the byte length
// / capacity / element count of the preceding reference field — the POSIX
// `msg_namelen`/`msg_iovlen`/`msg_controllen` convention.
// ---------------------------------------------------------------------------

/// Field kind inside an `in_struct:{...}` / `io_struct:{...}` descriptor.
#[derive(Debug)]
pub(crate) enum StructFieldKind {
    /// Integer scalar (`i8`..`u64`, `isize`, `usize`).
    Scalar(String),
    /// `bytes` — Handl record field is `Bytes`; address goes in the struct.
    BytesRef,
    /// `buf:N` — N-byte callee-filled buffer; pointer goes in the struct.
    Buf(usize),
    /// `iov` — Handl record field is `List<Bytes>`; builds an iovec array.
    IovRef,
    /// `iovbuf:K:M` — K callee-filled buffers of M bytes + iovec array.
    IovBuf(usize, usize),
    /// `ptr` — raw pointer slot; Handl value is a numeric address (or
    /// `{"__resource__": {"ptr": N}}` handle). Reads back as the address.
    Ptr,
    /// `arr:N:TYPE` — inline fixed array of N scalars. Handl value is
    /// `Bytes` for byte-wide elements, `List<Int>` otherwise.
    Arr(usize, String),
    /// `rec:{...}` — inline nested struct; Handl value is a nested record.
    Rec(Vec<ParsedField>),
}

/// A parsed struct field: `(name, kind, byte offset, slot width)`.
///
/// The offset and width come from the descriptor (`@off[:size]`) when the
/// backend emits libclang's real layout, or from sequential default layout
/// when hand-written descriptors omit them.
pub(crate) type ParsedField = (String, StructFieldKind, usize, usize);

/// One field of a struct descriptor.
pub(crate) struct StructField {
    pub name: String,
    pub kind: StructFieldKind,
    pub offset: usize,
    /// Slot width in bytes (scalar width or pointer width).
    pub size: usize,
    /// Index into [`StructArg::aux`] for reference kinds.
    pub aux: Option<usize>,
}

/// Buffers a struct field's pointer refers to.
pub(crate) struct FieldAux {
    /// `bytes`/`buf`: the single pointed-to buffer. `iov`/`iovbuf`: the
    /// per-element buffers.
    pub elems: Vec<Vec<u8>>,
    /// The iovec array for `iov`/`iovbuf` fields (empty otherwise). Never
    /// read after the call — it exists purely to keep the memory the
    /// struct points at alive for the call duration.
    #[allow(dead_code)]
    pub array: Vec<u8>,
}

/// A built C struct for `in_struct`/`io_struct`. Owns the struct bytes plus
/// every buffer its pointer fields reference — all kept alive until the C
/// call returns.
pub(crate) struct StructArg {
    /// The struct memory. Its address is the C argument.
    pub buf: Vec<u8>,
    pub fields: Vec<StructField>,
    pub aux: Vec<FieldAux>,
}

fn scalar_size(kind: &str) -> Result<usize, EvalError> {
    match kind {
        "i8" | "u8" => Ok(1),
        "i16" | "u16" => Ok(2),
        "i32" | "u32" => Ok(4),
        "i64" | "u64" | "isize" | "usize" => Ok(8),
        other => Err(EvalError::TypeMismatch(format!(
            "struct field: unsupported scalar {other}"
        ))),
    }
}

/// Parse a field kind spec (the part before `@offset`) and return it with
/// its default slot width. `ptrw` is the pointer-slot width used when the
/// descriptor doesn't state one explicitly.
fn parse_field_kind(spec: &str, ptrw: usize) -> Result<(StructFieldKind, usize), EvalError> {
    let ptr_sized = |k| Ok((k, ptrw));
    match spec {
        "bytes" => return ptr_sized(StructFieldKind::BytesRef),
        "iov" => return ptr_sized(StructFieldKind::IovRef),
        "ptr" => return ptr_sized(StructFieldKind::Ptr),
        _ => {}
    }
    if let Some(rest) = spec.strip_prefix("buf:") {
        return ptr_sized(StructFieldKind::Buf(parse_size("buf", rest)?));
    }
    if let Some(rest) = spec.strip_prefix("iovbuf:") {
        let (k, m) = rest
            .split_once(':')
            .ok_or_else(|| EvalError::TypeMismatch(format!("iovbuf needs K:M, got {spec:?}")))?;
        return ptr_sized(StructFieldKind::IovBuf(
            parse_size("iovbuf count", k)?,
            parse_size("iovbuf size", m)?,
        ));
    }
    if let Some(rest) = spec.strip_prefix("arr:") {
        let (n, ty) = rest
            .split_once(':')
            .ok_or_else(|| EvalError::TypeMismatch(format!("arr needs N:TYPE, got {spec:?}")))?;
        let n = parse_size("arr count", n)?;
        let w = scalar_size(ty)?;
        return Ok((StructFieldKind::Arr(n, ty.to_string()), n * w));
    }
    if let Some(rest) = spec.strip_prefix("rec:") {
        let inner = rest
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .ok_or_else(|| EvalError::TypeMismatch(format!("rec needs {{...}}, got {spec:?}")))?;
        let nested = parse_struct_fields(inner, ptrw)?;
        let total = nested
            .iter()
            .map(|(_, _, off, sz)| off + sz)
            .max()
            .unwrap_or(0);
        return Ok((StructFieldKind::Rec(nested), total));
    }
    scalar_size(spec).map(|w| (StructFieldKind::Scalar(spec.to_string()), w))
}

/// Split a field spec on the `@` layout marker at brace depth zero —
/// `@`s inside nested `rec:{...}` specs belong to the nested fields.
fn split_layout(spec: &str) -> (&str, Option<&str>) {
    let mut depth = 0i32;
    for (i, c) in spec.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            '@' if depth == 0 => return (&spec[..i], Some(&spec[i + 1..])),
            _ => {}
        }
    }
    (spec, None)
}

/// Split a `{...}` field list on commas at brace depth zero — nested
/// `rec:{...}` specs contain commas of their own.
fn split_fields(inner: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in inner.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&inner[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&inner[start..]);
    out
}

fn write_int(buf: &mut [u8], offset: usize, size: usize, value: i64) {
    let bytes = value.to_ne_bytes();
    buf[offset..offset + size].copy_from_slice(&bytes[..size]);
}

/// Zero-extending integer read.
fn read_int(buf: &[u8], offset: usize, size: usize) -> i64 {
    let mut tmp = [0u8; 8];
    tmp[..size].copy_from_slice(&buf[offset..offset + size]);
    i64::from_ne_bytes(tmp)
}

/// Sign-aware integer read: sub-64-bit signed fields get sign-extended so
/// a kernel-written `i32` `-1` reads back as `-1`, not `4294967295`.
fn read_scalar(buf: &[u8], offset: usize, size: usize, signed: bool) -> i64 {
    let v = read_int(buf, offset, size);
    if signed && size < 8 && size > 0 && (v >> (size * 8 - 1)) & 1 == 1 {
        v | (-1i64 << (size * 8))
    } else {
        v
    }
}

fn is_signed_scalar(kind: &str) -> bool {
    kind.starts_with('i')
}

/// Parse an `in_struct:{...}`/`io_struct:{...}` descriptor's field list.
///
/// Each field is `name:kind` or `name:kind@off[:size]` — an explicit
/// offset/slot-width carries the backend's real target layout; without it
/// fields are laid out sequentially with alignment == width (the legacy
/// LP64-compatible fallback for hand-written descriptors).
fn parse_struct_fields(inner: &str, ptrw: usize) -> Result<Vec<ParsedField>, EvalError> {
    let mut fields = Vec::new();
    let mut offset = 0usize;
    for pair in split_fields(inner) {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (name, spec) = pair
            .split_once(':')
            .ok_or_else(|| EvalError::TypeMismatch(format!("invalid struct field: {pair}")))?;
        let spec = spec.trim();
        let (kind_spec, layout) = split_layout(spec);
        let kind_spec = kind_spec.trim();
        let layout = layout.map(str::trim);
        let (kind, default_w) = parse_field_kind(kind_spec, ptrw)?;
        let (off, size) = match layout {
            Some(l) => {
                let (off_s, size_s) = l.split_once(':').map_or((l, None), |(a, b)| (a, Some(b)));
                let off: usize = off_s.trim().parse().map_err(|_| {
                    EvalError::TypeMismatch(format!("invalid field offset: {pair}"))
                })?;
                let size = size_s
                    .map(|s| s.trim().parse::<usize>())
                    .transpose()
                    .map_err(|_| EvalError::TypeMismatch(format!("invalid field size: {pair}")))?
                    .unwrap_or(default_w);
                (off, size)
            }
            None => {
                let misalign = offset % default_w;
                if misalign != 0 {
                    offset += default_w - misalign;
                }
                (offset, default_w)
            }
        };
        fields.push((name.trim().to_string(), kind, off, size));
        offset = off + size;
    }
    Ok(fields)
}

/// Default pointer-slot width for hand-written descriptors without an
/// explicit `:size` — the backend always emits one; 8 covers LP64 hosts.
const DEFAULT_PTR_WIDTH: usize = 8;

/// Build a C struct from a descriptor and (for `in_struct`) a Handl record.
pub(crate) fn marshal_struct_arg(
    inner: &str,
    record: Option<&JsonValue>,
) -> Result<StructArg, EvalError> {
    let parsed = parse_struct_fields(inner, DEFAULT_PTR_WIDTH)?;
    let total = parsed
        .iter()
        .map(|(_, _, off, sz)| off + sz)
        .max()
        .unwrap_or(0);
    let mut arg = StructArg {
        buf: vec![0u8; total],
        fields: Vec::with_capacity(parsed.len()),
        aux: Vec::new(),
    };
    let obj = record.and_then(|v| v.as_object());
    let aux_idx = fill_fields(&mut arg.buf, &parsed, obj, &mut arg.aux)?;
    for ((name, kind, offset, size), aux) in parsed.into_iter().zip(aux_idx) {
        arg.fields.push(StructField {
            name,
            kind,
            offset,
            size,
            aux,
        });
    }
    Ok(arg)
}

/// Write every parsed field into `buf`, appending referenced buffers to
/// `aux` so they stay alive for the call. Returns one `aux` index per
/// field (`Some` for reference kinds — `bytes`/`buf`/`iov`/`iovbuf`).
fn fill_fields(
    buf: &mut [u8],
    parsed: &[ParsedField],
    obj: Option<&Map<String, JsonValue>>,
    aux: &mut Vec<FieldAux>,
) -> Result<Vec<Option<usize>>, EvalError> {
    let mut aux_idx = Vec::with_capacity(parsed.len());
    let mut last_ref_len = 0usize;
    for (name, kind, offset, size) in parsed {
        let name = name.as_str();
        match kind {
            StructFieldKind::Scalar(_) => {
                aux_idx.push(None);
                let value = obj
                    .and_then(|o| o.get(name))
                    .and_then(as_integral_i64)
                    .unwrap_or_else(|| {
                        // Auto-*len* fill: an absent scalar named `*len`
                        // takes the preceding reference field's length.
                        if name.ends_with("len") {
                            last_ref_len as i64
                        } else {
                            0
                        }
                    });
                write_int(buf, *offset, *size, value);
            }
            StructFieldKind::Ptr => {
                aux_idx.push(None);
                let value = obj
                    .and_then(|o| o.get(name))
                    .and_then(|v| {
                        as_integral_i64(v).or_else(|| {
                            v.as_object()
                                .and_then(|r| r.get("__resource__"))
                                .and_then(|r| r.get("ptr"))
                                .and_then(as_integral_i64)
                        })
                    })
                    .unwrap_or(0);
                write_int(buf, *offset, *size, value);
            }
            StructFieldKind::Arr(n, ty) => {
                aux_idx.push(None);
                let esz = scalar_size(ty)?;
                if esz == 1 {
                    if let Some(bytes) = obj
                        .and_then(|o| o.get(name))
                        .map(json_to_bytes)
                        .transpose()?
                    {
                        let take = bytes.len().min(n * esz);
                        buf[*offset..*offset + take].copy_from_slice(&bytes[..take]);
                    }
                } else if let Some(list) = obj.and_then(|o| o.get(name)).and_then(|v| v.as_array())
                {
                    for (i, el) in list.iter().take(*n).enumerate() {
                        if let Some(v) = as_integral_i64(el) {
                            write_int(buf, *offset + i * esz, esz, v);
                        }
                    }
                }
            }
            StructFieldKind::Rec(nested) => {
                aux_idx.push(None);
                let nested_obj = obj.and_then(|o| o.get(name)).and_then(|v| v.as_object());
                fill_fields(&mut buf[*offset..*offset + size], nested, nested_obj, aux)?;
            }
            StructFieldKind::BytesRef => {
                let bytes = obj
                    .and_then(|o| o.get(name))
                    .map(json_to_bytes)
                    .transpose()?;
                match bytes {
                    Some(bytes) if !bytes.is_empty() => {
                        last_ref_len = bytes.len();
                        let ptr = bytes.as_ptr() as i64;
                        aux.push(FieldAux {
                            elems: vec![bytes],
                            array: Vec::new(),
                        });
                        aux_idx.push(Some(aux.len() - 1));
                        write_int(buf, *offset, *size, ptr);
                    }
                    // Absent/empty → NULL pointer, zero length.
                    _ => {
                        last_ref_len = 0;
                        aux.push(FieldAux {
                            elems: Vec::new(),
                            array: Vec::new(),
                        });
                        aux_idx.push(Some(aux.len() - 1));
                    }
                }
            }
            StructFieldKind::Buf(n) => {
                let b = vec![0u8; *n];
                last_ref_len = *n;
                let ptr = b.as_ptr() as i64;
                aux.push(FieldAux {
                    elems: vec![b],
                    array: Vec::new(),
                });
                aux_idx.push(Some(aux.len() - 1));
                write_int(buf, *offset, *size, ptr);
            }
            StructFieldKind::IovRef | StructFieldKind::IovBuf(_, _) => {
                let elems: Vec<Vec<u8>> = match kind {
                    StructFieldKind::IovRef => obj
                        .and_then(|o| o.get(name))
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().map(json_to_bytes).collect())
                        .transpose()?
                        .unwrap_or_default(),
                    StructFieldKind::IovBuf(k, m) => vec![vec![0u8; *m]; *k],
                    _ => unreachable!(),
                };
                last_ref_len = elems.len();
                // iovec entry = {ptr, len} — each pointer-width.
                let mut array = vec![0u8; elems.len() * 2 * size];
                for (i, el) in elems.iter().enumerate() {
                    write_int(&mut array, i * 2 * size, *size, el.as_ptr() as i64);
                    write_int(&mut array, i * 2 * size + size, *size, el.len() as i64);
                }
                // Empty iov → NULL array pointer (Vec::as_ptr on an empty
                // Vec is dangling, not null).
                let ptr = if array.is_empty() {
                    0
                } else {
                    array.as_ptr() as i64
                };
                aux.push(FieldAux { elems, array });
                aux_idx.push(Some(aux.len() - 1));
                write_int(buf, *offset, *size, ptr);
            }
        }
    }
    Ok(aux_idx)
}

/// Read an `io_struct` back into a Handl record after the call.
///
/// Scalars read from the struct; `buf:N` fields become `Bytes` trimmed by
/// the following `*len` scalar's read-back value; `iovbuf` fields become
/// `List<Bytes>` trimmed by the cumulative `ret` count (the kernel fills
/// iovecs in order). `bytes`/`iov` input refs are omitted.
pub(crate) fn unmarshal_io_struct(arg: &StructArg, ret_i64: i64) -> JsonValue {
    let mut map = Map::new();
    // Scalars first — `buf` trims depend on their read-back values.
    let mut scalar_vals: Vec<(String, i64)> = Vec::new();
    for f in &arg.fields {
        if let StructFieldKind::Scalar(s) = &f.kind {
            scalar_vals.push((
                f.name.clone(),
                read_scalar(&arg.buf, f.offset, f.size, is_signed_scalar(s)),
            ));
        }
    }
    let mut iov_credit = ret_i64.max(0) as usize;
    for (i, f) in arg.fields.iter().enumerate() {
        match &f.kind {
            StructFieldKind::Scalar(s) => {
                let v = read_scalar(&arg.buf, f.offset, f.size, is_signed_scalar(s));
                map.insert(f.name.clone(), JsonValue::from(v));
            }
            StructFieldKind::Ptr => {
                let v = read_int(&arg.buf, f.offset, f.size);
                map.insert(f.name.clone(), JsonValue::from(v));
            }
            StructFieldKind::Arr(n, ty) => {
                let esz = scalar_size(ty).unwrap_or(1);
                let end = (f.offset + n * esz).min(arg.buf.len());
                if esz == 1 {
                    map.insert(f.name.clone(), bytes_to_json(&arg.buf[f.offset..end]));
                } else {
                    let signed = is_signed_scalar(ty);
                    let list: Vec<JsonValue> = (0..*n)
                        .map(|i| read_scalar(&arg.buf, f.offset + i * esz, esz, signed))
                        .map(JsonValue::from)
                        .collect();
                    map.insert(f.name.clone(), JsonValue::Array(list));
                }
            }
            StructFieldKind::Rec(nested) => {
                let m = decode_fields(&arg.buf[f.offset..], nested);
                map.insert(f.name.clone(), JsonValue::Object(m));
            }
            StructFieldKind::Buf(_) => {
                let aux = &arg.aux[f.aux.unwrap_or(0)];
                let take = arg
                    .fields
                    .get(i + 1)
                    .filter(|n| matches!(n.kind, StructFieldKind::Scalar(_)))
                    .and_then(|n| {
                        scalar_vals
                            .iter()
                            .find(|(nm, _)| *nm == n.name)
                            .map(|(_, v)| *v)
                    })
                    .map(|v| (v.max(0) as usize).min(aux.elems[0].len()))
                    .unwrap_or_else(|| aux.elems[0].len());
                map.insert(f.name.clone(), bytes_to_json(&aux.elems[0][..take]));
            }
            StructFieldKind::IovBuf(_, _) => {
                let aux = &arg.aux[f.aux.unwrap_or(0)];
                let list: Vec<JsonValue> = aux
                    .elems
                    .iter()
                    .map(|el| {
                        let take = iov_credit.min(el.len());
                        iov_credit -= take;
                        bytes_to_json(&el[..take])
                    })
                    .collect();
                map.insert(f.name.clone(), JsonValue::Array(list));
            }
            // `bytes`/`iov` input refs are omitted on read-back.
            _ => {}
        }
    }
    JsonValue::Object(map)
}

/// Decode nested `rec:{...}` fields (scalars, arrays, pointers, further
/// nesting — embedded structs hold no buffer refs) into a record map.
fn decode_fields(buf: &[u8], fields: &[ParsedField]) -> Map<String, JsonValue> {
    let mut map = Map::new();
    for (name, kind, offset, size) in fields {
        match kind {
            StructFieldKind::Scalar(s) => {
                let v = read_scalar(buf, *offset, *size, is_signed_scalar(s));
                map.insert(name.clone(), JsonValue::from(v));
            }
            StructFieldKind::Ptr => {
                map.insert(name.clone(), JsonValue::from(read_int(buf, *offset, *size)));
            }
            StructFieldKind::Arr(n, ty) => {
                let esz = scalar_size(ty).unwrap_or(1);
                if esz == 1 {
                    let end = (*offset + n * esz).min(buf.len());
                    map.insert(name.clone(), bytes_to_json(&buf[*offset..end]));
                } else {
                    let signed = is_signed_scalar(ty);
                    let list: Vec<JsonValue> = (0..*n)
                        .map(|i| read_scalar(buf, *offset + i * esz, esz, signed))
                        .map(JsonValue::from)
                        .collect();
                    map.insert(name.clone(), JsonValue::Array(list));
                }
            }
            StructFieldKind::Rec(nested) => {
                map.insert(
                    name.clone(),
                    JsonValue::Object(decode_fields(&buf[*offset..], nested)),
                );
            }
            _ => {}
        }
    }
    map
}

/// Wrap a raw C return as `{ ret, buf, outs, vals }` when `io_struct` (or
/// `out_struct`) was used: `buf` is the read-back struct record, `outs` any
/// byte out-slots, `vals` any `io_scalar` read-backs.
pub(crate) fn wrap_io_struct_return(
    raw_ret: &JsonValue,
    io_structs: &[&StructArg],
    out_slots: &[OutSlot],
    scalar_outs: &[ScalarSlot],
) -> JsonValue {
    let st = match io_structs.first() {
        Some(st) => st,
        None => return raw_ret.clone(),
    };
    let ret_i64 = raw_ret.as_i64().unwrap_or(0);
    let mut map = Map::new();
    map.insert("ret".to_string(), raw_ret.clone());
    map.insert("buf".to_string(), unmarshal_io_struct(st, ret_i64));
    if !out_slots.is_empty() {
        let outs: Vec<JsonValue> = out_slots.iter().map(|s| slot_json(s, ret_i64)).collect();
        map.insert("outs".to_string(), JsonValue::Array(outs));
    }
    if !scalar_outs.is_empty() {
        let vals: Vec<JsonValue> = scalar_outs
            .iter()
            .map(|s| JsonValue::from(s.read()))
            .collect();
        map.insert("vals".to_string(), JsonValue::Array(vals));
    }
    JsonValue::Object(map)
}

// ---------------------------------------------------------------------------
// ffi.read — generic pointer dereference
// ---------------------------------------------------------------------------

/// Copy `len` bytes at `addr` into a Vec. `addr` must be valid for `len`
/// bytes of reads — `ffi.read` is an unsafe surface; a bad pointer faults.
fn read_mem(addr: usize, len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    // SAFETY: the caller declares the pointer valid for `len` bytes; the
    // copy bounds the unsafe window to this call.
    unsafe { std::slice::from_raw_parts(addr as *const u8, len).to_vec() }
}

/// Decode a value at a raw pointer per `desc` — the `ffi.read` backend.
///
/// Supported descriptors: scalars (`i8`..`u64`, `isize`, `usize`), `ptr`
/// (the address stored at the slot), `bytes:N`, `arr:N:T`, `rec:{...}`
/// (same field grammar as struct descriptors), and `cstr` (NUL-terminated
/// string, bounded at 1 MiB).
pub(crate) fn marshal_read(addr: usize, desc: &str) -> Result<JsonValue, EvalError> {
    let ptrw = std::mem::size_of::<usize>();
    if desc == "ptr" {
        return Ok(JsonValue::from(read_int(&read_mem(addr, ptrw), 0, ptrw)));
    }
    if desc == "cstr" {
        const MAX: usize = 1 << 20;
        let mut n = 0usize;
        while n < MAX {
            // SAFETY: same contract as read_mem; the NUL scan is byte-wise.
            let b = unsafe { (addr as *const u8).add(n).read() };
            if b == 0 {
                break;
            }
            n += 1;
        }
        let buf = read_mem(addr, n);
        return String::from_utf8(buf)
            .map(JsonValue::String)
            .map_err(|_| EvalError::TypeMismatch("ffi.read cstr: invalid UTF-8".to_string()));
    }
    if let Some(rest) = desc.strip_prefix("bytes:") {
        let n = parse_size("bytes", rest)?;
        return Ok(bytes_to_json(&read_mem(addr, n)));
    }
    if let Some(rest) = desc.strip_prefix("arr:") {
        let (n_str, ty) = rest
            .split_once(':')
            .ok_or_else(|| EvalError::TypeMismatch(format!("arr needs N:TYPE, got {desc:?}")))?;
        let n = parse_size("arr count", n_str)?;
        let esz = scalar_size(ty)?;
        let buf = read_mem(addr, n * esz);
        if esz == 1 {
            return Ok(bytes_to_json(&buf));
        }
        let signed = is_signed_scalar(ty);
        let list: Vec<JsonValue> = (0..n)
            .map(|i| read_scalar(&buf, i * esz, esz, signed))
            .map(JsonValue::from)
            .collect();
        return Ok(JsonValue::Array(list));
    }
    if let Some(inner) = desc.strip_prefix("rec:") {
        let inner = inner
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .ok_or_else(|| EvalError::TypeMismatch(format!("rec needs {{...}}, got {desc:?}")))?;
        let fields = parse_struct_fields(inner, ptrw)?;
        let total = fields
            .iter()
            .map(|(_, _, off, sz)| off + sz)
            .max()
            .unwrap_or(0);
        let buf = read_mem(addr, total);
        return Ok(JsonValue::Object(decode_fields(&buf, &fields)));
    }
    let sz = scalar_size(desc)?;
    let buf = read_mem(addr, sz);
    Ok(JsonValue::from(read_scalar(
        &buf,
        0,
        sz,
        is_signed_scalar(desc),
    )))
}

/// Returns `true` for descriptors that consume one Handl argument.
pub(crate) fn descriptor_consumes_handl_arg(desc: &str) -> bool {
    if desc == "void" {
        return false;
    }
    for prefix in [
        "out_buf:",
        "out_ptr:",
        "io_buf:",
        "out_struct:",
        "io_struct:",
    ] {
        if desc.starts_with(prefix) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Unmarshalling: C -> JSON
// ---------------------------------------------------------------------------

/// Determine the return kind from a return descriptor string.
pub(crate) fn return_kind(descriptor: &str) -> Result<CRetKind, EvalError> {
    match descriptor {
        "i32" => Ok(CRetKind::Int32),
        "i64" | "u8" | "u16" | "u32" | "u64" | "ptr" | "cstr" | "handle" => Ok(CRetKind::Int64),
        "f64" => Ok(CRetKind::Float),
        "void" => Ok(CRetKind::Void),
        other => Err(EvalError::TypeMismatch(format!(
            "unknown return descriptor: {other}"
        ))),
    }
}

/// Unmarshal a C return value to a JSON value per the return descriptor.
pub(crate) fn unmarshal_return(descriptor: &str, ret: &CRet) -> Result<JsonValue, EvalError> {
    match descriptor {
        "i32" => match ret {
            CRet::Int(v) => Ok(JsonValue::from(*v as i32 as i64)),
            _ => Err(EvalError::Other(
                "expected integer return for i32".to_string(),
            )),
        },
        "i64" | "u8" | "u16" | "u32" | "u64" => match ret {
            CRet::Int(v) => Ok(JsonValue::from(*v)),
            _ => Err(EvalError::Other(format!(
                "expected integer return for {descriptor}"
            ))),
        },
        "f64" => match ret {
            CRet::Float(v) => serde_json::Number::from_f64(*v)
                .map(JsonValue::Number)
                .ok_or_else(|| EvalError::Other("non-finite f64 return".to_string())),
            _ => Err(EvalError::Other(
                "expected float return for f64".to_string(),
            )),
        },
        "ptr" => match ret {
            CRet::Int(v) => Ok(JsonValue::from(*v)),
            _ => Err(EvalError::Other(
                "expected integer return for ptr".to_string(),
            )),
        },
        // `handle` wraps the raw C return pointer in the JSON debug form of
        // a `TaggedValue::Resource` (ADR-0083 §2.2): `{"__resource__":
        // {"ptr": N}}`. The finalizer cannot be carried across the JSON
        // boundary — when this JSON is decoded back to a `TaggedValue`, the
        // resulting `Resource` has `finalizer = None`. The `FfiOpFunction`
        // stores an optional finalizer field for future use once the
        // dispatcher returns `TaggedValue` directly instead of JSON.
        "handle" => match ret {
            CRet::Int(v) => {
                let mut inner = Map::new();
                inner.insert("ptr".to_string(), JsonValue::from(*v));
                let mut outer = Map::new();
                outer.insert("__resource__".to_string(), JsonValue::Object(inner));
                Ok(JsonValue::Object(outer))
            }
            _ => Err(EvalError::Other(
                "expected integer return for handle".to_string(),
            )),
        },
        "cstr" => match ret {
            CRet::Int(v) => {
                if *v == 0 {
                    return Ok(JsonValue::Null);
                }
                let ptr = *v as *const c_char;
                let s = unsafe { CStr::from_ptr(ptr) }
                    .to_str()
                    .map_err(|e| EvalError::Other(format!("non-utf8 cstr return: {e}")))?;
                Ok(JsonValue::from(s))
            }
            _ => Err(EvalError::Other(
                "expected integer return for cstr".to_string(),
            )),
        },
        "void" => Ok(JsonValue::Null),
        other => Err(EvalError::TypeMismatch(format!(
            "unknown return descriptor: {other}"
        ))),
    }
}
