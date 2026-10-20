use crate::InitError;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// An initialized, independently owned, 64-byte-aligned allocation.
pub(crate) struct Buffer<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

/// Only plain numeric types may use all-zero initialization and trivial drop.
///
/// # Safety
/// Every bit pattern must be valid, all-zero bits must be a valid initialized
/// value, and the type must require no destructor or external resource cleanup.
pub(crate) unsafe trait Plain: Copy + Send + Sync {}
unsafe impl Plain for f32 {}
unsafe impl Plain for i8 {}
unsafe impl Plain for i16 {}
unsafe impl Plain for i32 {}
unsafe impl Plain for u16 {}

impl<T: Plain> Buffer<T> {
    pub fn new(len: usize) -> Result<Self, InitError> {
        let size = len
            .checked_mul(size_of::<T>())
            .ok_or(InitError::Allocation)?;
        let layout = Layout::from_size_align(size.max(1), 64).map_err(|_| InitError::Allocation)?;
        // SAFETY: nonzero valid layout; Plain permits zero initialization.
        let ptr =
            NonNull::new(unsafe { alloc_zeroed(layout) }.cast()).ok_or(InitError::Allocation)?;
        Ok(Self { ptr, len, layout })
    }
    pub fn region(&self) -> crate::MemoryRegion {
        crate::MemoryRegion {
            address: self.ptr.as_ptr().cast(),
            len: self.layout.size(),
        }
    }
    // Raw access does not create an intermediate slice reference. Kernel
    // callers must enforce exclusive access to every range being written.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}
impl<T> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: allocation is initialized and exclusively owned for its lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}
impl<T> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: exclusive access to the allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        // SAFETY: same pointer and layout returned by alloc_zeroed; only Plain is constructed.
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) }
    }
}
unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}
