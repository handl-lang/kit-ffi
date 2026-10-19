//! errno extraction for the FFI error clause.
//!
//! When an [`crate::FfiOpFunction`] is configured with `error_on_lt_zero`, a
//! negative integer return triggers an errno read. On Linux this calls
//! `__errno_location()` to obtain the thread-local errno pointer and reads
//! its value synchronously, before any other libc operation can clobber it.

use std::ffi::c_int;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn __errno_location() -> *mut c_int;
}

/// Extract the current C `errno` value.
///
/// The caller **must** invoke this synchronously, immediately after the C
/// call whose error is being checked — before any other libc operation
/// (including allocation) could overwrite errno.
pub(crate) fn last_errno() -> i32 {
    #[cfg(target_os = "linux")]
    {
        unsafe { *__errno_location() }
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}
