//! Tests for the FFI dispatcher using real Linux syscalls.
//!
//! All tests are `#[cfg(target_os = "linux")]` because they call into libc
//! directly via `dlopen` / `dlsym` and rely on glibc-specific symbols
//! (`getpid`, `abs`, `labs`, `fabs`, `strlen`, `open`) and errno semantics
//! (`__errno_location`, `ENOENT = 2`).

#![cfg(target_os = "linux")]

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

use crate::marshal::{
    CArg, CRet, bytes_to_json, json_to_bytes, marshal_arg, marshal_call_args, marshal_struct_arg,
    unmarshal_io_struct, unmarshal_return,
};
use crate::{FfiLoad, FfiOpFunction, FfiSym};

/// Call an [`FfiOpFunction`] with a fresh [`EvalContext`].
fn call(func: &FfiOpFunction, args: &[Value]) -> Result<Value, EvalError> {
    let mut ctx = EvalContext::new();
    func.call(args, &mut ctx)
}

// -------------------------------------------------------------------------
// getpid — no args, returns i32
// -------------------------------------------------------------------------

#[test]
fn getpid_returns_current_pid() {
    let func = FfiOpFunction::new("libc.so.6", "getpid", vec![], "i32", false);
    let result = call(&func, &[]).expect("getpid call should succeed");
    let pid = result.as_i64().expect("getpid should return an integer") as u32;
    assert_eq!(pid, std::process::id());
}

// -------------------------------------------------------------------------
// Marshalling round-trips via real libc functions
// -------------------------------------------------------------------------

#[test]
fn marshal_i32_roundtrip() {
    // abs(-42) = 42
    let func = FfiOpFunction::new("libc.so.6", "abs", vec!["i32".to_string()], "i32", false);
    let result = call(&func, &[json!(-42)]).expect("abs call should succeed");
    assert_eq!(result.as_i64(), Some(42));
}

#[test]
fn marshal_i64_roundtrip() {
    // labs(long) on 64-bit Linux: long is 64-bit.
    let val: i64 = -9_223_372_036_854_775_807; // i64::MIN + 1
    let func = FfiOpFunction::new("libc.so.6", "labs", vec!["i64".to_string()], "i64", false);
    let result = call(&func, &[json!(val)]).expect("labs call should succeed");
    assert_eq!(result.as_i64(), Some(val.abs()));
}

#[test]
fn marshal_f64_roundtrip() {
    // fabs(-42.5) = 42.5 — fabs lives in libm, not libc.
    let func = FfiOpFunction::new("libm.so.6", "fabs", vec!["f64".to_string()], "f64", false);
    let result = call(&func, &[json!(-42.5)]).expect("fabs call should succeed");
    let f = result.as_f64().expect("fabs should return a number");
    assert!((f - 42.5).abs() < f64::EPSILON);
}

#[test]
fn marshal_cstr_roundtrip() {
    // strlen("hello") = 5 — verifies the cstr arg is correctly null-terminated
    // and passed as a pointer.
    let func = FfiOpFunction::new(
        "libc.so.6",
        "strlen",
        vec!["cstr".to_string()],
        "i64",
        false,
    );
    let result = call(&func, &[json!("hello")]).expect("strlen call should succeed");
    assert_eq!(result.as_i64(), Some(5));
}

#[test]
fn marshal_cstr_return_unmarshal() {
    // Direct unit test: unmarshal a CRet::Int holding a valid cstr pointer.
    let c = std::ffi::CString::new("hello").unwrap();
    let ptr = c.as_ptr() as i64;
    let ret = CRet::Int(ptr);
    let v = unmarshal_return("cstr", &ret).expect("cstr unmarshal");
    assert_eq!(v, json!("hello"));

    // Null pointer → JSON null.
    let ret = CRet::Int(0);
    let v = unmarshal_return("cstr", &ret).expect("cstr null unmarshal");
    assert_eq!(v, Value::Null);
}

// -------------------------------------------------------------------------
// Error cases
// -------------------------------------------------------------------------

#[test]
fn error_on_i32_overflow() {
    // 2^31 exceeds i32 range — marshal should fail before the call.
    let func = FfiOpFunction::new("libc.so.6", "abs", vec!["i32".to_string()], "i32", false);
    let result = call(&func, &[json!(2_147_483_648_i64)]); // 2^31
    assert!(
        matches!(result, Err(EvalError::TypeMismatch(_))),
        "expected TypeMismatch for i32 overflow, got {result:?}"
    );
}

#[test]
fn error_on_embedded_nul_in_cstr() {
    let func = FfiOpFunction::new(
        "libc.so.6",
        "strlen",
        vec!["cstr".to_string()],
        "i64",
        false,
    );
    let result = call(&func, &[json!("hello\0world")]);
    assert!(
        matches!(result, Err(EvalError::TypeMismatch(_))),
        "expected TypeMismatch for embedded NUL, got {result:?}"
    );
}

// -------------------------------------------------------------------------
// Error clause: open() on a nonexistent path → errno ENOENT (2)
// -------------------------------------------------------------------------

#[test]
fn error_clause_open_nonexistent_extracts_errno() {
    // open("/nonexistent/path", O_RDONLY=0) returns -1, errno = ENOENT (2).
    let func = FfiOpFunction::new(
        "libc.so.6",
        "open",
        vec!["cstr".to_string(), "i32".to_string()],
        "i32",
        true, // error_on_lt_zero
    );
    let result = call(
        &func,
        &[json!("/nonexistent/handl-ffi-test-path"), json!(0)],
    );
    match result {
        Err(EvalError::Other(msg)) => {
            assert!(
                msg.contains("errno: 2"),
                "expected errno 2 (ENOENT) in error message, got: {msg}"
            );
        }
        other => panic!("expected Err(EvalError::Other) with errno, got {other:?}"),
    }
}

#[test]
fn error_clause_disabled_returns_raw_negative() {
    // Same call but with error_on_lt_zero = false — should return -1 as i32.
    let func = FfiOpFunction::new(
        "libc.so.6",
        "open",
        vec!["cstr".to_string(), "i32".to_string()],
        "i32",
        false,
    );
    let result = call(
        &func,
        &[json!("/nonexistent/handl-ffi-test-path"), json!(0)],
    )
    .expect("open should succeed (return -1)");
    assert_eq!(result.as_i64(), Some(-1));
}

// -------------------------------------------------------------------------
// Marshal unit tests (no C call needed)
// -------------------------------------------------------------------------

#[test]
fn marshal_i32_value() {
    let mut keep_alive = Vec::new();
    let arg = marshal_arg("i32", &json!(42), &mut keep_alive).expect("marshal i32");
    assert!(matches!(arg, CArg::Int(42)));
    assert!(keep_alive.is_empty());
}

#[test]
fn marshal_cstr_owns_cstring() {
    let mut keep_alive = Vec::new();
    let arg = marshal_arg("cstr", &json!("hi"), &mut keep_alive).expect("marshal cstr");
    match arg {
        CArg::Int(ptr) => {
            assert!(ptr != 0);
        }
        other => panic!("expected CArg::Int for cstr, got {other:?}"),
    }
    assert_eq!(keep_alive.len(), 1);
}

#[test]
fn marshal_unknown_descriptor_errors() {
    let mut keep_alive = Vec::new();
    let result = marshal_arg("bytes", &json!(42), &mut keep_alive);
    assert!(matches!(result, Err(EvalError::TypeMismatch(_))));
}

// -------------------------------------------------------------------------
// Bytes marshalling helpers (no C call needed)
// -------------------------------------------------------------------------

#[test]
fn json_to_bytes_accepts_tagged_object() {
    let v = json!({"__bytes__": [0xDE, 0xAD, 0xBE, 0xEF]});
    let b = json_to_bytes(&v).expect("tagged bytes decode");
    assert_eq!(b, vec![0xDE, 0xAD, 0xBE, 0xEF]);
}

#[test]
fn json_to_bytes_accepts_plain_array() {
    let v = json!([1, 2, 3, 255]);
    let b = json_to_bytes(&v).expect("plain-array bytes decode");
    assert_eq!(b, vec![1, 2, 3, 255]);
}

#[test]
fn json_to_bytes_rejects_out_of_range() {
    let v = json!({"__bytes__": [256]});
    assert!(matches!(json_to_bytes(&v), Err(EvalError::TypeMismatch(_))));
}

#[test]
fn json_to_bytes_rejects_non_object() {
    let v = json!("not bytes");
    assert!(matches!(json_to_bytes(&v), Err(EvalError::TypeMismatch(_))));
}

#[test]
fn bytes_to_json_roundtrips() {
    let original = vec![0u8, 127, 128, 255];
    let encoded = bytes_to_json(&original);
    let decoded = json_to_bytes(&encoded).expect("roundtrip decode");
    assert_eq!(decoded, original);
    // Encoding is the canonical {"__bytes__": [...]} form.
    assert!(encoded.as_object().unwrap().contains_key("__bytes__"));
}

// -------------------------------------------------------------------------
// Buffer FFI ops via real Linux syscalls (read / write)
//
// Uses memfd_create(2) to obtain an anonymous in-memory file with no path,
// then exercises write(2) via `in_buf` and read(2) via `out_buf:N`, verifying
// the bytes round-trip across the FFI boundary.
// -------------------------------------------------------------------------

/// Open an anonymous memfd for buffer-op tests. Returns the raw fd.
fn memfd() -> i32 {
    let memfd = FfiOpFunction::new(
        "libc.so.6",
        "memfd_create",
        vec!["cstr".to_string(), "i32".to_string()],
        "i32",
        true,
    );
    let fd_val = call(&memfd, &[json!("handl-ffi-bytes-test"), json!(0)])
        .expect("memfd_create should succeed");
    let fd = fd_val.as_i64().expect("fd should be an integer") as i32;
    assert!(fd >= 0, "memfd_create returned invalid fd: {fd}");
    fd
}

/// Close a raw fd, panicking on failure.
fn close_fd(fd: i32) {
    let close_fn = FfiOpFunction::new("libc.so.6", "close", vec!["i32".to_string()], "i32", true);
    call(&close_fn, &[json!(fd)]).expect("close should succeed");
}

#[test]
fn in_buf_write_roundtrips_bytes_to_memfd() {
    let fd = memfd();

    // write(fd, buf, count) — `in_buf` expands to (ptr, len).
    let payload = json!({"__bytes__": [0xDE, 0xAD, 0xBE, 0xEF]});
    let write_fn = FfiOpFunction::new(
        "libc.so.6",
        "write",
        vec!["i32".to_string(), "in_buf".to_string()],
        "i64",
        true,
    );
    let written = call(&write_fn, &[json!(fd), payload]).expect("write should succeed");
    assert_eq!(written.as_i64(), Some(4));

    close_fd(fd);
}

#[test]
fn out_buf_read_returns_ret_and_buf_record() {
    let fd = memfd();

    // Seed the memfd with 4 known bytes via write(2) + in_buf.
    let payload = json!({"__bytes__": [0xDE, 0xAD, 0xBE, 0xEF]});
    let write_fn = FfiOpFunction::new(
        "libc.so.6",
        "write",
        vec!["i32".to_string(), "in_buf".to_string()],
        "i64",
        true,
    );
    call(&write_fn, &[json!(fd), payload]).expect("write should succeed");

    // Rewind to offset 0: lseek(fd, 0, SEEK_SET=0).
    let lseek = FfiOpFunction::new(
        "libc.so.6",
        "lseek",
        vec!["i32".to_string(), "i64".to_string(), "i32".to_string()],
        "i64",
        true,
    );
    call(&lseek, &[json!(fd), json!(0), json!(0)]).expect("lseek should succeed");

    // read(fd, buf, count) — `out_buf:16` allocates 16 bytes, expands to
    // (ptr, 16), and the return is wrapped as { ret, buf }.
    let read_fn = FfiOpFunction::new(
        "libc.so.6",
        "read",
        vec!["i32".to_string(), "out_buf:16".to_string()],
        "i64",
        true,
    );
    // Only one handl arg (fd): out_buf consumes none.
    let result = call(&read_fn, &[json!(fd)]).expect("read should succeed");
    let obj = result
        .as_object()
        .expect("read result should be a {ret, buf} record");
    assert_eq!(
        obj.get("ret").and_then(|v| v.as_i64()),
        Some(4),
        "ret should be the byte count read"
    );
    let buf = obj.get("buf").expect("buf field should be present");
    let bytes_arr = buf
        .get("__bytes__")
        .and_then(|v| v.as_array())
        .expect("buf should be a {__bytes__:[...]} value");
    let got: Vec<u8> = bytes_arr
        .iter()
        .map(|v| v.as_i64().unwrap() as u8)
        .collect();
    assert_eq!(got, vec![0xDE, 0xAD, 0xBE, 0xEF]);

    close_fd(fd);
}

#[test]
fn out_buf_clamps_buf_to_ret_count() {
    // read into a 16-byte buffer from an empty memfd: ret == 0, buf == [].
    let fd = memfd();
    let read_fn = FfiOpFunction::new(
        "libc.so.6",
        "read",
        vec!["i32".to_string(), "out_buf:16".to_string()],
        "i64",
        true,
    );
    let result = call(&read_fn, &[json!(fd)]).expect("read should succeed");
    let obj = result.as_object().expect("{ret, buf} record");
    assert_eq!(obj.get("ret").and_then(|v| v.as_i64()), Some(0));
    let bytes_arr = obj
        .get("buf")
        .and_then(|v| v.get("__bytes__"))
        .and_then(|v| v.as_array())
        .expect("buf bytes array");
    assert!(bytes_arr.is_empty(), "buf should be empty when ret == 0");

    close_fd(fd);
}

/// Decode the `outs` list of a `{ret, buf, outs}` record into byte vecs.
fn outs_bytes(result: &Value) -> Vec<Vec<u8>> {
    result
        .get("outs")
        .and_then(|v| v.as_array())
        .expect("outs field should be present")
        .iter()
        .map(|v| {
            v.get("__bytes__")
                .and_then(|b| b.as_array())
                .expect("out slot should be {__bytes__: [...]}")
                .iter()
                .map(|n| n.as_i64().unwrap() as u8)
                .collect()
        })
        .collect()
}

#[test]
fn out_ptr_socketpair_returns_two_fds() {
    // socketpair(AF_UNIX=1, SOCK_STREAM=1, 0, int sv[2]) — sv is a fixed
    // 8-byte out-param passed as a bare pointer → `out_ptr:8` (one C arg,
    // no handl arg). The slot keeps all 8 bytes (two i32 fds).
    let sp = FfiOpFunction::new(
        "libc.so.6",
        "socketpair",
        vec![
            "i32".to_string(),
            "i32".to_string(),
            "i32".to_string(),
            "out_ptr:8".to_string(),
        ],
        "i32",
        true,
    );
    let result = call(&sp, &[json!(1), json!(1), json!(0)]).expect("socketpair should succeed");
    let obj = result.as_object().expect("{ret, buf, outs} record");
    assert_eq!(obj.get("ret").and_then(|v| v.as_i64()), Some(0));

    let outs = outs_bytes(&result);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), 8);
    let fd0 = i32::from_ne_bytes(outs[0][0..4].try_into().unwrap());
    let fd1 = i32::from_ne_bytes(outs[0][4..8].try_into().unwrap());
    assert!(fd0 >= 0 && fd1 >= 0 && fd0 != fd1);

    // Prove the pair is connected: write to fd0, read from fd1.
    let write_fn = FfiOpFunction::new(
        "libc.so.6",
        "write",
        vec!["i32".to_string(), "in_buf".to_string()],
        "i64",
        true,
    );
    call(&write_fn, &[json!(fd0), json!({"__bytes__": [0xCA, 0xFE]})])
        .expect("write to socketpair");
    let read_fn = FfiOpFunction::new(
        "libc.so.6",
        "read",
        vec!["i32".to_string(), "out_buf:16".to_string()],
        "i64",
        true,
    );
    let read_res = call(&read_fn, &[json!(fd1)]).expect("read from socketpair");
    let buf = read_res
        .get("buf")
        .and_then(|v| v.get("__bytes__"))
        .and_then(|v| v.as_array())
        .expect("buf bytes");
    let got: Vec<u8> = buf.iter().map(|v| v.as_i64().unwrap() as u8).collect();
    assert_eq!(got, vec![0xCA, 0xFE]);

    close_fd(fd0);
    close_fd(fd1);
}

#[test]
fn io_buf_getsockname_trims_to_written_len() {
    // getsockname(fd, sockaddr*, socklen_t*) — the value-result pattern:
    // `io_buf:128` passes (ptr, &len) where the len cell is seeded with
    // the 128-byte capacity and the kernel writes the actual length back.
    let sp = FfiOpFunction::new(
        "libc.so.6",
        "socketpair",
        vec![
            "i32".to_string(),
            "i32".to_string(),
            "i32".to_string(),
            "out_ptr:8".to_string(),
        ],
        "i32",
        true,
    );
    let result = call(&sp, &[json!(1), json!(1), json!(0)]).expect("socketpair");
    let sv = &outs_bytes(&result)[0];
    let fd = i32::from_ne_bytes(sv[0..4].try_into().unwrap());

    let gsn = FfiOpFunction::new(
        "libc.so.6",
        "getsockname",
        vec!["i32".to_string(), "io_buf:128".to_string()],
        "i32",
        true,
    );
    let result = call(&gsn, &[json!(fd)]).expect("getsockname should succeed");
    assert_eq!(result.get("ret").and_then(|v| v.as_i64()), Some(0));

    let outs = outs_bytes(&result);
    assert_eq!(outs.len(), 1);
    // sockaddr_un for an unnamed AF_UNIX socket is just the 2-byte
    // family — the slot is trimmed to what the kernel wrote, not 128.
    assert!(outs[0].len() < 128, "should be trimmed to written length");
    assert_eq!(
        u16::from_ne_bytes(outs[0][0..2].try_into().unwrap()),
        1, // AF_UNIX
        "sockaddr family should be AF_UNIX"
    );

    close_fd(fd);
}

/// Make a connected AF_UNIX SOCK_STREAM socketpair; returns (fd0, fd1).
fn sockpair() -> (i32, i32) {
    let sp = FfiOpFunction::new(
        "libc.so.6",
        "socketpair",
        vec![
            "i32".to_string(),
            "i32".to_string(),
            "i32".to_string(),
            "out_ptr:8".to_string(),
        ],
        "i32",
        true,
    );
    let result = call(&sp, &[json!(1), json!(1), json!(0)]).expect("socketpair");
    let sv = &outs_bytes(&result)[0];
    (
        i32::from_ne_bytes(sv[0..4].try_into().unwrap()),
        i32::from_ne_bytes(sv[4..8].try_into().unwrap()),
    )
}

#[test]
fn sendmsg_in_struct_recvmsg_io_struct_roundtrip() {
    // sendmsg(fd, const msghdr*, flags): `in_struct` builds the struct
    // from a Handl record — `iov` turns a List<Bytes> into an iovec array
    // and absent `*len` scalars auto-fill (namelen=0, iovlen=2).
    let (fd0, fd1) = sockpair();
    let sendmsg = FfiOpFunction::new(
        "libc.so.6",
        "sendmsg",
        vec![
            "i32".to_string(),
            "in_struct:{msg_name:bytes,msg_namelen:u32,msg_iov:iov,msg_iovlen:u64,msg_control:bytes,msg_controllen:u64,msg_flags:i32}".to_string(),
            "i32".to_string(),
        ],
        "i64",
        true,
    );
    let msg = json!({
        "msg_iov": [
            {"__bytes__": [0xDE, 0xAD]},
            {"__bytes__": [0xBE, 0xEF, 0x42]}
        ],
        "msg_flags": 0
    });
    let sent = call(&sendmsg, &[json!(fd0), msg, json!(0)]).expect("sendmsg should succeed");
    assert_eq!(sent.as_i64(), Some(5), "sendmsg should send all 5 bytes");

    // recvmsg(fd, msghdr*, flags): `io_struct` allocates name/control
    // buffers and an iov array callee-side; read-back trims bufs by the
    // written *len fields and iovecs by the cumulative ret count.
    let recvmsg = FfiOpFunction::new(
        "libc.so.6",
        "recvmsg",
        vec![
            "i32".to_string(),
            "io_struct:{msg_name:buf:128,msg_namelen:u32,msg_iov:iovbuf:2:8,msg_iovlen:u64,msg_control:buf:256,msg_controllen:u64,msg_flags:i32}".to_string(),
            "i32".to_string(),
        ],
        "i64",
        true,
    );
    let result = call(&recvmsg, &[json!(fd1), json!(0)]).expect("recvmsg should succeed");
    let obj = result.as_object().expect("{ret, buf} record");
    assert_eq!(obj.get("ret").and_then(|v| v.as_i64()), Some(5));
    let msg = obj
        .get("buf")
        .and_then(|v| v.as_object())
        .expect("msghdr record");

    let iov = msg
        .get("msg_iov")
        .and_then(|v| v.as_array())
        .expect("msg_iov list");
    let iov_bytes: Vec<Vec<u8>> = iov
        .iter()
        .map(|v| {
            v.get("__bytes__")
                .and_then(|b| b.as_array())
                .unwrap()
                .iter()
                .map(|n| n.as_i64().unwrap() as u8)
                .collect()
        })
        .collect();
    // Scatter-gather fill: each iovec has 8-byte capacity, so all 5
    // bytes land in iov[0] and iov[1] stays empty.
    assert_eq!(iov_bytes[0], vec![0xDE, 0xAD, 0xBE, 0xEF, 0x42]);
    assert_eq!(iov_bytes[1], Vec::<u8>::new());

    close_fd(fd0);
    close_fd(fd1);
}

#[test]
fn io_scalar_ioctl_fionread_returns_pending_bytes() {
    // ioctl(fd, FIONREAD, &n) — `io_scalar:i32` seeds an int cell from a
    // handl arg, passes its pointer, and reads it back into `vals`.
    let (fd0, fd1) = sockpair();
    let write_fn = FfiOpFunction::new(
        "libc.so.6",
        "write",
        vec!["i32".to_string(), "in_buf".to_string()],
        "i64",
        true,
    );
    call(
        &write_fn,
        &[json!(fd0), json!({"__bytes__": [1, 2, 3, 4, 5]})],
    )
    .expect("write should succeed");

    const FIONREAD: i64 = 0x541B;
    let ioctl = FfiOpFunction::new(
        "libc.so.6",
        "ioctl",
        vec![
            "i32".to_string(),
            "i64".to_string(),
            "io_scalar:i32".to_string(),
        ],
        "i32",
        true,
    );
    let result =
        call(&ioctl, &[json!(fd1), json!(FIONREAD), json!(0)]).expect("ioctl should succeed");
    assert_eq!(result.get("ret").and_then(|v| v.as_i64()), Some(0));
    let vals = result
        .get("vals")
        .and_then(|v| v.as_array())
        .expect("vals field should be present");
    assert_eq!(
        vals[0].as_i64(),
        Some(5),
        "FIONREAD should report 5 pending bytes"
    );

    close_fd(fd0);
    close_fd(fd1);
}

#[test]
fn cbytes_passes_ptr_len_pair_to_write() {
    // `cbytes` is semantically identical to `in_buf` for write(2): it passes
    // (ptr, len) with no NUL termination or UTF-8 validation. Verify a buffer
    // containing a NUL byte round-trips (cstr would reject this).
    let fd = memfd();
    let payload = json!({"__bytes__": [0x00, 0x01, 0x02, 0xFF]});
    let write_fn = FfiOpFunction::new(
        "libc.so.6",
        "write",
        vec!["i32".to_string(), "cbytes".to_string()],
        "i64",
        true,
    );
    let written = call(&write_fn, &[json!(fd), payload]).expect("write should succeed");
    assert_eq!(written.as_i64(), Some(4));

    // Read it back to confirm the NUL byte survived.
    let lseek = FfiOpFunction::new(
        "libc.so.6",
        "lseek",
        vec!["i32".to_string(), "i64".to_string(), "i32".to_string()],
        "i64",
        true,
    );
    call(&lseek, &[json!(fd), json!(0), json!(0)]).expect("lseek should succeed");
    let read_fn = FfiOpFunction::new(
        "libc.so.6",
        "read",
        vec!["i32".to_string(), "out_buf:16".to_string()],
        "i64",
        true,
    );
    let result = call(&read_fn, &[json!(fd)]).expect("read should succeed");
    let bytes_arr = result
        .get("buf")
        .and_then(|v| v.get("__bytes__"))
        .and_then(|v| v.as_array())
        .expect("buf bytes array");
    let got: Vec<u8> = bytes_arr
        .iter()
        .map(|v| v.as_i64().unwrap() as u8)
        .collect();
    assert_eq!(got, vec![0x00, 0x01, 0x02, 0xFF]);

    close_fd(fd);
}

#[test]
fn out_buf_invalid_size_errors() {
    let read_fn = FfiOpFunction::new(
        "libc.so.6",
        "read",
        vec!["i32".to_string(), "out_buf:notanumber".to_string()],
        "i64",
        true,
    );
    let result = call(&read_fn, &[json!(0)]);
    assert!(
        matches!(result, Err(EvalError::TypeMismatch(_))),
        "expected TypeMismatch for invalid out_buf size, got {result:?}"
    );
}

#[test]
fn in_buf_wrong_value_type_errors() {
    let write_fn = FfiOpFunction::new(
        "libc.so.6",
        "write",
        vec!["i32".to_string(), "in_buf".to_string()],
        "i64",
        true,
    );
    // Passing a string where Bytes is expected.
    let result = call(&write_fn, &[json!(1), json!("not bytes")]);
    assert!(
        matches!(result, Err(EvalError::TypeMismatch(_))),
        "expected TypeMismatch for non-Bytes in_buf, got {result:?}"
    );
}

// -------------------------------------------------------------------------
// Handle descriptor (ADR-0083 §2.2 — finalizable values / RAII)
//
// The `"handle"` return descriptor wraps the raw C return pointer in the
// JSON debug form of a `TaggedValue::Resource`: `{"__resource__": {"ptr":
// N}}`. These tests verify the wrapping using a real syscall (`open` on a
// temp file) and that the JSON decodes back to a `TaggedValue::Resource`
// with the correct pointer.
// -------------------------------------------------------------------------

#[test]
fn handle_return_wraps_pointer_in_resource_json() {
    // Create a temp file, then open(2) it with return descriptor "handle".
    // The returned JSON should be {"__resource__": {"ptr": N}} where N is
    // the raw fd (a small non-negative integer cast to a pointer).
    let dir = std::env::temp_dir();
    let path = dir.join("handl-ffi-handle-test.txt");
    std::fs::write(&path, b"hello").expect("write temp file");

    let open_fn = FfiOpFunction::new(
        "libc.so.6",
        "open",
        vec!["cstr".to_string(), "i32".to_string()],
        "handle",
        true,
    );
    let result = call(
        &open_fn,
        &[
            json!(path.to_str().unwrap()),
            json!(0), // O_RDONLY
        ],
    )
    .expect("open should succeed");
    let obj = result
        .as_object()
        .expect("handle return should be a {__resource__:{ptr:N}} object");
    let res = obj
        .get("__resource__")
        .expect("should have __resource__ key");
    let ptr = res
        .get("ptr")
        .and_then(|v| v.as_i64())
        .expect("should have ptr field");
    assert!(ptr >= 0, "fd should be non-negative, got {ptr}");

    // Clean up: close the fd (it leaked here since the JSON form has no
    // finalizer — this is expected; the finalizer mechanism is documented
    // as not-yet-wired for the JSON return path).
    close_fd(ptr as i32);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn handle_return_decodes_to_tagged_value_resource() {
    // Verify the JSON form produced by the "handle" descriptor decodes back
    // to a TaggedValue::Resource with the correct pointer and no finalizer.
    use handl_value::TaggedValue;

    let dir = std::env::temp_dir();
    let path = dir.join("handl-ffi-handle-decode-test.txt");
    std::fs::write(&path, b"hello").expect("write temp file");

    let open_fn = FfiOpFunction::new(
        "libc.so.6",
        "open",
        vec!["cstr".to_string(), "i32".to_string()],
        "handle",
        true,
    );
    let result = call(
        &open_fn,
        &[
            json!(path.to_str().unwrap()),
            json!(0), // O_RDONLY
        ],
    )
    .expect("open should succeed");

    // Convert the JSON return to a TaggedValue — it should be a Resource.
    let tv: TaggedValue = result.into();
    match &tv {
        TaggedValue::Resource(r) => {
            let ptr = r.ptr() as i64;
            assert!(ptr >= 0, "fd should be non-negative, got {ptr}");
            assert!(
                r.finalizer.is_none(),
                "decoded resource has no finalizer (JSON cannot carry one)"
            );
            close_fd(ptr as i32);
        }
        other => panic!("expected TaggedValue::Resource, got {other:?}"),
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn handle_arg_accepts_resource_json() {
    // The "handle" *arg* descriptor accepts a {"__resource__":{"ptr":N}}
    // value and passes the pointer through. Verify by opening a file with
    // "handle" return, then passing that result to close(2) via a "handle"
    // arg descriptor.
    let dir = std::env::temp_dir();
    let path = dir.join("handl-ffi-handle-arg-test.txt");
    std::fs::write(&path, b"hello").expect("write temp file");

    let open_fn = FfiOpFunction::new(
        "libc.so.6",
        "open",
        vec!["cstr".to_string(), "i32".to_string()],
        "handle",
        true,
    );
    let handle_val = call(
        &open_fn,
        &[
            json!(path.to_str().unwrap()),
            json!(0), // O_RDONLY
        ],
    )
    .expect("open should succeed");
    let ptr = handle_val
        .get("__resource__")
        .and_then(|r| r.get("ptr"))
        .and_then(|v| v.as_i64())
        .expect("extract ptr from handle");
    assert!(ptr >= 0);

    // close(fd) via a "handle" arg descriptor — should succeed (return 0).
    let close_fn = FfiOpFunction::new(
        "libc.so.6",
        "close",
        vec!["handle".to_string()],
        "i32",
        true,
    );
    let result = call(&close_fn, &[handle_val]).expect("close should succeed");
    assert_eq!(result.as_i64(), Some(0), "close should return 0 on success");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn handle_arg_accepts_bare_number() {
    // The "handle" arg descriptor also accepts a bare number (e.g. a POSIX
    // fd) for convenience.
    let dir = std::env::temp_dir();
    let path = dir.join("handl-ffi-handle-bare-test.txt");
    std::fs::write(&path, b"hello").expect("write temp file");

    // Open with a plain "i32" return to get the raw fd.
    let open_fn = FfiOpFunction::new(
        "libc.so.6",
        "open",
        vec!["cstr".to_string(), "i32".to_string()],
        "i32",
        true,
    );
    let fd_val = call(
        &open_fn,
        &[
            json!(path.to_str().unwrap()),
            json!(0), // O_RDONLY
        ],
    )
    .expect("open should succeed");
    let fd = fd_val.as_i64().expect("fd should be a number") as i32;
    assert!(fd >= 0);

    // close(fd) via a "handle" arg descriptor with a bare number.
    let close_fn = FfiOpFunction::new(
        "libc.so.6",
        "close",
        vec!["handle".to_string()],
        "i32",
        true,
    );
    let result = call(&close_fn, &[json!(fd)]).expect("close should succeed");
    assert_eq!(result.as_i64(), Some(0));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn with_finalizer_attaches_finalizer() {
    // Verify the builder method stores the finalizer. We don't call a real
    // cleanup function here — just check the field is set.
    unsafe extern "C" fn dummy_finalizer(_ptr: *mut std::ffi::c_void) {
        // no-op for testing
    }
    let func = FfiOpFunction::new("libc.so.6", "getpid", vec![], "i32", false);
    assert!(func.finalizer().is_none());
    let func = unsafe { func.with_finalizer(dummy_finalizer) };
    assert!(func.finalizer().is_some());
}

#[test]
fn handle_return_unmarshal_unit() {
    // Direct unit test: unmarshal a CRet::Int as "handle" produces the
    // {"__resource__":{"ptr":N}} JSON form.
    use crate::marshal::CRet;
    use crate::marshal::unmarshal_return;
    let ret = CRet::Int(0x1234);
    let v = unmarshal_return("handle", &ret).expect("handle unmarshal");
    let obj = v.as_object().expect("object form");
    let ptr = obj
        .get("__resource__")
        .and_then(|r| r.get("ptr"))
        .and_then(|v| v.as_i64());
    assert_eq!(ptr, Some(0x1234));
}

// -------------------------------------------------------------------------
// Phase 8.1: Cranelift trampolines — no arity ceiling
//
// These tests define `#[unsafe(no_mangle)] extern "C"` functions in the test
// binary and call them through the FFI dispatcher via the "self" library
// (Library::this()), exercising signatures the old arity-0..=6 macro
// dispatch could not express.
// -------------------------------------------------------------------------

/// An 8-argument integer C function (exceeds the old 6-arg ceiling).
#[unsafe(no_mangle)]
pub extern "C" fn ffi_test_sum8(
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    e: i64,
    f: i64,
    g: i64,
    h: i64,
) -> i64 {
    a + b * 2 + c * 3 + d * 4 + e * 5 + f * 6 + g * 7 + h * 8
}

/// A 9-argument mixed int/float C function — verifies that args beyond the
/// 6th still land in the right registers/stack slots on both the integer
/// and floating-point sides.
#[unsafe(no_mangle)]
pub extern "C" fn ffi_test_mixed9(
    a: i64,
    x: f64,
    b: i64,
    y: f64,
    c: i64,
    z: f64,
    d: i64,
    w: f64,
    e: i64,
) -> i64 {
    a + b + c + d + e + (x + y + z + w) as i64
}

#[test]
fn ffi_call_with_eight_args_exceeds_old_ceiling() {
    let func = FfiOpFunction::new(
        "self",
        "ffi_test_sum8",
        vec![
            "i64".to_string(),
            "i64".to_string(),
            "i64".to_string(),
            "i64".to_string(),
            "i64".to_string(),
            "i64".to_string(),
            "i64".to_string(),
            "i64".to_string(),
        ],
        "i64",
        false,
    );
    // 1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8 = 204
    let result = call(
        &func,
        &[
            json!(1),
            json!(2),
            json!(3),
            json!(4),
            json!(5),
            json!(6),
            json!(7),
            json!(8),
        ],
    )
    .expect("8-arg FFI call should succeed");
    assert_eq!(result.as_i64(), Some(204));
}

#[test]
fn ffi_call_with_nine_mixed_args() {
    let func = FfiOpFunction::new(
        "self",
        "ffi_test_mixed9",
        vec![
            "i64".to_string(),
            "f64".to_string(),
            "i64".to_string(),
            "f64".to_string(),
            "i64".to_string(),
            "f64".to_string(),
            "i64".to_string(),
            "f64".to_string(),
            "i64".to_string(),
        ],
        "i64",
        false,
    );
    // ints: 1+3+5+7+9 = 25; floats: 0.5+1.5+2.5+3.5 = 8.0 → 33
    let result = call(
        &func,
        &[
            json!(1),
            json!(0.5),
            json!(3),
            json!(1.5),
            json!(5),
            json!(2.5),
            json!(7),
            json!(3.5),
            json!(9),
        ],
    )
    .expect("9-arg mixed FFI call should succeed");
    assert_eq!(result.as_i64(), Some(33));
}

#[test]
fn cranelift_shim_is_cached_per_signature() {
    use crate::cranelift_shim::{ArgClass, resolve_shim};
    use crate::marshal::CRetKind;

    // A distinctive signature unlikely to collide with other tests.
    // (Note: the shim cache is process-global and shared with the parallel
    // tests above, so we assert on Arc identity, not global cache counts.)
    let args = vec![
        CArg::Int(1),
        CArg::Float(2.0),
        CArg::Int(3),
        CArg::Float(4.0),
    ];
    let s1 = resolve_shim(CRetKind::Int64, &args).expect("shim compile");
    let s2 = resolve_shim(CRetKind::Int64, &args).expect("shim reuse");
    assert!(
        std::sync::Arc::ptr_eq(&s1, &s2),
        "same signature should reuse the cached shim"
    );

    // A different signature compiles a distinct shim.
    let other_args = vec![CArg::Int(1), CArg::Float(2.0), CArg::Int(3), CArg::Int(4)];
    let s3 = resolve_shim(CRetKind::Int64, &other_args).expect("distinct shim");
    assert!(!std::sync::Arc::ptr_eq(&s1, &s3));

    // A different return kind is a different signature too.
    let s4 = resolve_shim(CRetKind::Float, &args).expect("different-ret shim");
    assert!(!std::sync::Arc::ptr_eq(&s1, &s4));

    // ArgClass classification sanity check.
    assert_eq!(ArgClass::from(&CArg::Int(0)), ArgClass::Int);
    assert_eq!(ArgClass::from(&CArg::Float(0.0)), ArgClass::Float);
}

// -------------------------------------------------------------------------
// out_struct descriptor (ADR-0083 §4)
// -------------------------------------------------------------------------

#[test]
fn out_struct_clock_gettime_returns_record() {
    // clock_gettime(CLOCK_REALTIME, &ts) — first arg is bound to 0,
    // second is an out_struct. The result is { ret, buf } where buf is
    // a record { tv_sec, tv_nsec }.
    let func = FfiOpFunction::new(
        "libc.so.6",
        "clock_gettime",
        vec![
            "i32".to_string(),
            "out_struct:{tv_sec:i64,tv_nsec:i64}".to_string(),
        ],
        "i32",
        false,
    );
    let result = call(&func, &[json!(0)]).expect("clock_gettime should succeed");
    let obj = result
        .as_object()
        .expect("clock_gettime should return a {ret, buf} record");
    assert_eq!(obj.get("ret").and_then(|v| v.as_i64()), Some(0));
    let buf = obj
        .get("buf")
        .and_then(|v| v.as_object())
        .expect("buf should be a record");
    let tv_sec = buf
        .get("tv_sec")
        .and_then(|v| v.as_i64())
        .expect("tv_sec should be present");
    assert!(tv_sec > 0, "tv_sec should be positive, got {tv_sec}");
    let tv_nsec = buf
        .get("tv_nsec")
        .and_then(|v| v.as_i64())
        .expect("tv_nsec should be present");
    assert!(
        (0..1_000_000_000).contains(&tv_nsec),
        "tv_nsec should be in [0, 1e9), got {tv_nsec}"
    );
}

#[test]
fn struct_descriptor_accepts_all_integer_field_types() {
    let st = marshal_struct_arg(
        "a:i8,b:i16,c:i32,d:i64,e:u8,f:u16,g:u32,h:u64,i:isize,j:usize",
        None,
    )
    .expect("parse struct fields");
    assert_eq!(st.fields.len(), 10);
    let sizes: Vec<usize> = st.fields.iter().map(|f| f.size).collect();
    assert_eq!(sizes, vec![1, 2, 4, 8, 1, 2, 4, 8, 8, 8]);
    let offsets: Vec<usize> = st.fields.iter().map(|f| f.offset).collect();
    assert_eq!(offsets, vec![0, 2, 4, 8, 16, 18, 20, 24, 32, 40]);
}

#[test]
fn struct_descriptor_explicit_offsets() {
    // `name:kind@off[:size]` — the backend emits libclang's real layout.
    let st = marshal_struct_arg("x:i32@4,y:i64@16:8", None).expect("explicit offsets");
    assert_eq!(st.fields[0].offset, 4);
    assert_eq!(st.fields[1].offset, 16);
    assert_eq!(st.buf.len(), 24);
}

#[test]
fn io_struct_unmarshal_u32_i16_fields() {
    let mut st = marshal_struct_arg("x:u32,y:i16", None).expect("parse struct fields");
    st.buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());
    st.buf[4..6].copy_from_slice(&(-12345i16).to_ne_bytes());
    let rec = unmarshal_io_struct(&st, 0);
    let obj = rec.as_object().expect("record");
    assert_eq!(
        obj.get("x").and_then(|v| v.as_i64()),
        Some(0xDEADBEEFu32 as i64)
    );
    assert_eq!(obj.get("y").and_then(|v| v.as_i64()), Some(-12345));
}

#[test]
fn io_struct_unmarshal_arr_and_rec_fields() {
    let mut st = marshal_struct_arg(
        "fam:u16@0,data:arr:14:u8@2,inner:rec:{tv_sec:i64@0,tv_nsec:i64@8}@16",
        None,
    )
    .expect("parse struct fields");
    st.buf[0..2].copy_from_slice(&2u16.to_ne_bytes());
    st.buf[2..6].copy_from_slice(&[127, 0, 0, 1]);
    st.buf[16..24].copy_from_slice(&1700000000i64.to_ne_bytes());
    let rec = unmarshal_io_struct(&st, 0);
    let obj = rec.as_object().expect("record");
    assert_eq!(obj.get("fam").and_then(|v| v.as_i64()), Some(2));
    let data: Vec<u8> = json_to_bytes(obj.get("data").unwrap()).unwrap();
    assert_eq!(&data[..4], &[127, 0, 0, 1]);
    let inner = obj.get("inner").and_then(|v| v.as_object()).unwrap();
    assert_eq!(
        inner.get("tv_sec").and_then(|v| v.as_i64()),
        Some(1700000000)
    );
}

// -------------------------------------------------------------------------
// in_ptr — bare Bytes pointer, no trailing size_t
// -------------------------------------------------------------------------

#[test]
fn marshal_in_ptr_expands_to_one_arg() {
    let res = marshal_call_args(&["in_ptr".to_string()], &[json!({"__bytes__": [1, 2, 3]})])
        .expect("in_ptr marshal should succeed");
    assert_eq!(res.c_args.len(), 1, "in_ptr expands to a single C arg");
    match res.c_args[0] {
        CArg::Int(p) => assert_ne!(p, 0),
        other => panic!("expected Int pointer, got {other:?}"),
    }
    assert_eq!(res.byte_keep_alive.len(), 1);
}

#[test]
fn in_ptr_real_call_memcmp() {
    // memcmp(a, b, n) — (const void*, const void*, size_t): two in_ptr
    // args plus an explicit length.
    let func = FfiOpFunction::new(
        "libc.so.6",
        "memcmp",
        vec![
            "in_ptr".to_string(),
            "in_ptr".to_string(),
            "i64".to_string(),
        ],
        "i32",
        false,
    );
    let eq = call(&func, &[json!([1, 2, 3]), json!([1, 2, 3]), json!(3)])
        .expect("memcmp call should succeed");
    assert_eq!(eq.as_i64(), Some(0));
    let ne = call(&func, &[json!([1, 2, 3]), json!([1, 2, 4]), json!(3)])
        .expect("memcmp call should succeed");
    assert!(ne.as_i64().unwrap() != 0);
}

// -------------------------------------------------------------------------
// ffi.load / ffi.sym — non-raising probes (Result-tag returns)
// -------------------------------------------------------------------------

#[test]
fn ffi_load_returns_ok_tag_for_libc() {
    let mut ctx = EvalContext::new();
    let r = FfiLoad
        .call(&[json!("libc.so.6")], &mut ctx)
        .expect("ffi.load should not raise");
    assert_eq!(r["__tag__"], "ok");
}

#[test]
fn ffi_load_returns_err_tag_for_missing_lib() {
    let mut ctx = EvalContext::new();
    let r = FfiLoad
        .call(&[json!("libdefinitely-not-present.so")], &mut ctx)
        .expect("ffi.load should not raise");
    assert_eq!(r["__tag__"], "err");
    assert!(r["value"].as_str().unwrap().contains("dlopen"));
}

#[test]
fn ffi_sym_resolves_getpid_and_reports_missing() {
    let mut ctx = EvalContext::new();
    let ok = FfiSym
        .call(&[json!("libc.so.6"), json!("getpid")], &mut ctx)
        .expect("ffi.sym should not raise");
    assert_eq!(ok["__tag__"], "ok");
    assert_ne!(ok["value"].as_i64(), Some(0));

    let err = FfiSym
        .call(&[json!("libc.so.6"), json!("no_such_symbol_xyz")], &mut ctx)
        .expect("ffi.sym should not raise");
    assert_eq!(err["__tag__"], "err");
}

// -------------------------------------------------------------------------
// ffi.read — generic pointer dereference
// -------------------------------------------------------------------------

#[test]
fn ffi_read_scalar_sign_extends() {
    use crate::FfiRead;
    static CELL: i32 = -42;
    let mut ctx = EvalContext::new();
    let v = FfiRead
        .call(&[json!(&CELL as *const i32 as i64), json!("i32")], &mut ctx)
        .expect("ffi.read i32");
    assert_eq!(v.as_i64(), Some(-42));
}

#[test]
fn ffi_read_bytes_and_arr() {
    use crate::FfiRead;
    static BUF: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];
    static WORDS: [u16; 3] = [1, 2, 0x8000];
    let mut ctx = EvalContext::new();
    let b = FfiRead
        .call(&[json!(BUF.as_ptr() as i64), json!("bytes:4")], &mut ctx)
        .expect("ffi.read bytes");
    assert_eq!(b["__bytes__"], json!([0xde, 0xad, 0xbe, 0xef]));
    let a = FfiRead
        .call(
            &[json!(WORDS.as_ptr() as i64), json!("arr:3:u16")],
            &mut ctx,
        )
        .expect("ffi.read arr");
    assert_eq!(a, json!([1, 2, 0x8000]));
}

#[test]
fn ffi_read_rec_decodes_fields_and_ptr() {
    use crate::FfiRead;
    #[repr(C)]
    struct Node {
        flags: i32,
        family: i32,
        next: *const Node,
    }
    // The statics below are read-only test fixtures.
    unsafe impl Sync for Node {}
    static TAIL: Node = Node {
        flags: 7,
        family: 2,
        next: std::ptr::null(),
    };
    static HEAD: Node = Node {
        flags: 1,
        family: 2,
        next: &TAIL,
    };
    let mut ctx = EvalContext::new();
    let desc = "rec:{flags:i32@0:4,family:i32@4:4,next:ptr@8:8}";
    let head = FfiRead
        .call(&[json!(&HEAD as *const Node as i64), json!(desc)], &mut ctx)
        .expect("ffi.read rec");
    assert_eq!(head["flags"], json!(1));
    assert_eq!(head["family"], json!(2));
    assert_eq!(head["next"].as_i64(), Some(&TAIL as *const Node as i64));

    // Chase the link: read the pointed-to node through the returned address.
    let tail = FfiRead
        .call(&[head["next"].clone(), json!(desc)], &mut ctx)
        .expect("ffi.read next");
    assert_eq!(tail["flags"], json!(7));
    assert_eq!(tail["next"].as_i64(), Some(0));
}

#[test]
fn ffi_read_cstr_and_resource_form() {
    use crate::FfiRead;
    static S: &[u8] = b"hello\0world";
    let mut ctx = EvalContext::new();
    let s = FfiRead
        .call(&[json!(S.as_ptr() as i64), json!("cstr")], &mut ctx)
        .expect("ffi.read cstr");
    assert_eq!(s, json!("hello"));

    // `{"__resource__": {"ptr": N}}` handles are accepted like numbers.
    let r = FfiRead
        .call(
            &[
                json!({"__resource__": {"ptr": S.as_ptr() as i64}}),
                json!("cstr"),
            ],
            &mut ctx,
        )
        .expect("ffi.read resource");
    assert_eq!(r, json!("hello"));
}

#[test]
fn ffi_read_null_pointer_errors() {
    use crate::FfiRead;
    let mut ctx = EvalContext::new();
    let err = FfiRead
        .call(&[json!(0), json!("i32")], &mut ctx)
        .expect_err("ffi.read null");
    assert!(err.to_string().contains("null"));
}

#[test]
fn in_ptr_bytes_content() {
    // `in_ptr` must pass the raw byte buffer — UTF-16 LE consumers (ICU
    // UChar* params) read it verbatim.
    let text = json!({"__bytes__": [97, 0, 46, 0, 98, 0]});
    let res = marshal_call_args(&["in_ptr".to_string()], &[text]).unwrap();
    let ptr = match res.c_args[0] {
        CArg::Int(p) => p as *const u8,
        _ => panic!(),
    };
    let bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(ptr, 6).to_vec() };
    assert_eq!(bytes, vec![97, 0, 46, 0, 98, 0]);
}
