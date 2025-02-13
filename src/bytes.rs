use core::iter::FromIterator;
use core::mem::{self, ManuallyDrop};
use core::ops::{Deref, RangeBounds};
use core::ptr::NonNull;
use core::{cmp, fmt, hash, ptr, slice, usize};

use alloc::{
    alloc::{dealloc, Layout},
    borrow::Borrow,
    boxed::Box,
    string::String,
    vec::Vec,
};

use crate::buf::IntoIter;
use crate::bytes_mut::RawElemsMut;
#[allow(unused)]
use crate::loom::sync::atomic::AtomicMut;
use crate::loom::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use crate::{offset_from, Buf, ElemsMut};

/// A cheaply cloneable and sliceable chunk of contiguous memory.
///
/// `Elems` is an efficient container for storing and operating on contiguous
/// slices of memory. It is intended for use primarily in networking code, but
/// could have applications elsewhere as well.
///
/// `Elems` values facilitate zero-copy network programming by allowing multiple
/// `Elems` objects to point to the same underlying memory.
///
/// `Elems` does not have a single implementation. It is an interface, whose
/// exact behavior is implemented through dynamic dispatch in several underlying
/// implementations of `Elems`.
///
/// All `Elems` implementations must fulfill the following requirements:
/// - They are cheaply cloneable and thereby shareable between an unlimited amount
///   of components, for example by modifying a reference count.
/// - Instances can be sliced to refer to a subset of the original buffer.
///
/// ```
/// use elems::Elems;
///
/// let mut mem = Elems::from("Hello world");
/// let a = mem.slice(0..5);
///
/// assert_eq!(a, "Hello");
///
/// let b = mem.split_to(6);
///
/// assert_eq!(mem, "world");
/// assert_eq!(b, "Hello ");
/// ```
///
/// # Memory layout
///
/// The `Elems` struct itself is fairly small, limited to 4 `usize` fields used
/// to track information about which segment of the underlying memory the
/// `Elems` handle has access to.
///
/// `Elems` keeps both a pointer to the shared state containing the full memory
/// slice and a pointer to the start of the region visible by the handle.
/// `Elems` also tracks the length of its view into the memory.
///
/// # Sharing
///
/// `Elems` contains a vtable, which allows implementations of `Elems` to define
/// how sharing/cloning is implemented in detail.
/// When `Elems::clone()` is called, `Elems` will call the vtable function for
/// cloning the backing storage in order to share it behind multiple `Elems`
/// instances.
///
/// For `Elems` implementations which refer to constant memory (e.g. created
/// via `Elems::from_static()`) the cloning implementation will be a no-op.
///
/// For `Elems` implementations which point to a reference counted shared storage
/// (e.g. an `Arc<[u8]>`), sharing will be implemented by increasing the
/// reference count.
///
/// Due to this mechanism, multiple `Elems` instances may point to the same
/// shared memory region.
/// Each `Elems` instance can point to different sections within that
/// memory region, and `Elems` instances may or may not have overlapping views
/// into the memory.
///
/// The following diagram visualizes a scenario where 2 `Elems` instances make
/// use of an `Arc`-based backing storage, and provide access to different views:
///
/// ```text
///
///    Arc ptrs                   ┌─────────┐
///    ________________________ / │ Elems 2 │
///   /                           └─────────┘
///  /          ┌───────────┐     |         |
/// |_________/ │  Elems 1  │     |         |
/// |           └───────────┘     |         |
/// |           |           | ___/ data     | tail
/// |      data |      tail |/              |
/// v           v           v               v
/// ┌─────┬─────┬───────────┬───────────────┬─────┐
/// │ Arc │     │           │               │     │
/// └─────┴─────┴───────────┴───────────────┴─────┘
/// ```
pub struct Elems<T: Pod = u8> {
    ptr: *const T,
    len: usize,
    // inlined "trait object"
    data: AtomicPtr<()>,
    vtable: &'static Vtable,
}

/// Alias to [`Elems<u8>`].
pub type Bytes = Elems<u8>;

pub unsafe trait Pod: Copy + Clone + Default + Send + Sync + 'static {}
macro_rules! impl_pod {
    (ty $($t:ty),*) => {
        $(unsafe impl Pod for $t {})*
    };
    (arr $($len:expr),*) => {
        $(unsafe impl<T: Pod> Pod for [T; $len] {})*
    };
    (tuple ($($t:ident),*)) => {
        unsafe impl<$($t: Pod),*> Pod for ($($t,)*) {}
    }
}
unsafe impl<T: Pod> Pod for core::num::Wrapping<T> {}
unsafe impl<T: Pod> Pod for core::num::Saturating<T> {}
impl_pod!(ty u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);
impl_pod!(arr 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
impl_pod!(tuple(A));
impl_pod!(tuple(A, B));
impl_pod!(tuple(A, B, C));
impl_pod!(tuple(A, B, C, D));
impl_pod!(tuple(A, B, C, D, E));
impl_pod!(tuple(A, B, C, D, E, F));
impl_pod!(tuple(A, B, C, D, E, F, G));
impl_pod!(tuple(A, B, C, D, E, F, G, H));
impl_pod!(tuple(A, B, C, D, E, F, G, H, I));
impl_pod!(tuple(A, B, C, D, E, F, G, H, I, J));
impl_pod!(tuple(A, B, C, D, E, F, G, H, I, J, K));
impl_pod!(tuple(A, B, C, D, E, F, G, H, I, J, K, L));

pub(crate) struct Vtable {
    /// fn(data, ptr, len)
    pub clone: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> RawElems,
    /// fn(data, ptr, len)
    ///
    /// takes `Elems` to value
    pub to_vec: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> (*mut u8, usize, usize),
    pub to_mut: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> RawElemsMut,
    /// fn(data)
    pub is_unique: unsafe fn(&AtomicPtr<()>) -> bool,
    /// fn(data, ptr, len)
    pub drop: unsafe fn(&mut AtomicPtr<()>, *const u8, usize),

    pub promotable: bool,
}

pub(crate) struct RawElems {
    ptr: *const u8,
    len: usize,
    // inlined "trait object"
    data: AtomicPtr<()>,
    vtable: &'static Vtable,
}
impl RawElems {
    pub fn erase<T: Pod>(bytes: Elems<T>) -> RawElems {
        let mut bytes = ManuallyDrop::new(bytes);
        RawElems {
            ptr: bytes.ptr.cast::<u8>(),
            len: bytes.len,
            data: AtomicPtr::new(*bytes.data.get_mut()),
            vtable: bytes.vtable,
        }
    }
    pub const fn recover<T: Pod>(self) -> Elems<T> {
        Elems {
            ptr: self.ptr.cast::<T>(),
            len: self.len,
            data: self.data,
            vtable: self.vtable,
        }
    }
}

impl<T: Pod> Elems<T> {
    /// Creates a new empty `Elems`.
    ///
    /// This will not allocate and the returned `Elems` handle will be empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Bytes;
    ///
    /// let b = Bytes::new();
    /// assert_eq!(&b[..], b"");
    /// ```
    #[inline]
    #[cfg(not(all(loom, test)))]
    pub const fn new() -> Self {
        Elems::from_static(&[])
    }

    /// Creates a new empty `Elems`.
    #[cfg(all(loom, test))]
    pub fn new() -> Self {
        Elems::from_static(&[])
    }

    /// Creates a new `Elems` from a static slice.
    ///
    /// The returned `Elems` will point directly to the static slice. There is
    /// no allocating or copying.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let b = Elems::from_static(b"hello");
    /// assert_eq!(&b[..], b"hello");
    /// ```
    #[inline]
    #[cfg(not(all(loom, test)))]
    pub const fn from_static(bytes: &'static [T]) -> Self {
        Elems {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            data: AtomicPtr::new(ptr::null_mut()),
            vtable: &VTables::<T>::STATIC_VTABLE,
        }
    }

    /// Creates a new `Elems` from a static slice.
    #[cfg(all(loom, test))]
    pub fn from_static(bytes: &'static [T]) -> Self {
        Elems {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            data: AtomicPtr::new(ptr::null_mut()),
            vtable: &VTables::<T>::STATIC_VTABLE,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a new `Elems` with length zero and the given pointer as the address.
    fn new_empty_with_ptr(ptr: *const T) -> Self {
        debug_assert!(!ptr.is_null());

        // Detach this pointer's provenance from whichever allocation it came from, and reattach it
        // to the provenance of the fake ZST [u8;0] at the same address.
        let ptr = without_provenance(ptr as usize);

        Elems {
            ptr,
            len: 0,
            data: AtomicPtr::new(ptr::null_mut()),
            vtable: &VTables::<T>::STATIC_VTABLE,
        }
    }

    /// Create [Elems] with a buffer whose lifetime is controlled
    /// via an explicit owner.
    ///
    /// A common use case is to zero-copy construct from mapped memory.
    ///
    /// ```
    /// # struct File;
    /// #
    /// # impl File {
    /// #     pub fn open(_: &str) -> Result<Self, ()> {
    /// #         Ok(Self)
    /// #     }
    /// # }
    /// #
    /// # mod memmap2 {
    /// #     pub struct Mmap;
    /// #
    /// #     impl Mmap {
    /// #         pub unsafe fn map(_file: &super::File) -> Result<Self, ()> {
    /// #             Ok(Self)
    /// #         }
    /// #     }
    /// #
    /// #     impl AsRef<[u8]> for Mmap {
    /// #         fn as_ref(&self) -> &[u8] {
    /// #             b"buf"
    /// #         }
    /// #     }
    /// # }
    /// use elems::Elems;
    /// use memmap2::Mmap;
    ///
    /// # fn main() -> Result<(), ()> {
    /// let file = File::open("upload_bundle.tar.gz")?;
    /// let mmap = unsafe { Mmap::map(&file) }?;
    /// let b = Elems::from_owner(mmap);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The `owner` will be transferred to the constructed [Elems] object, which
    /// will ensure it is dropped once all remaining clones of the constructed
    /// object are dropped. The owner will then be responsible for dropping the
    /// specified region of memory as part of its [Drop] implementation.
    ///
    /// Note that converting [Elems] constructed from an owner into a [ElemsMut]
    /// will always create a deep copy of the buffer into newly allocated memory.
    pub fn from_owner<O>(owner: O) -> Self
    where
        O: AsRef<[T]> + Send + 'static,
    {
        // Safety & Miri:
        // The ownership of `owner` is first transferred to the `Owned` wrapper and `Elems` object.
        // This ensures that the owner is pinned in memory, allowing us to call `.as_ref()` safely
        // since the lifetime of the owner is controlled by the lifetime of the new `Elems` object,
        // and the lifetime of the resulting borrowed `&[u8]` matches that of the owner.
        // Note that this remains safe so long as we only call `.as_ref()` once.
        //
        // There are some additional special considerations here:
        //   * We rely on Elems's Drop impl to clean up memory should `.as_ref()` panic.
        //   * Setting the `ptr` and `len` on the bytes object last (after moving the owner to
        //     Elems) allows Miri checks to pass since it avoids obtaining the `&[u8]` slice
        //     from a stack-owned Box.
        // More details on this: https://github.com/tokio-rs/bytes/pull/742/#discussion_r1813375863
        //                  and: https://github.com/tokio-rs/bytes/pull/742/#discussion_r1813316032

        let owned = Box::into_raw(Box::new(Owned {
            lifetime: OwnedLifetime {
                ref_cnt: AtomicUsize::new(1),
                drop: owned_box_and_drop::<O>,
            },
            owner,
        }));

        let mut ret = Elems {
            ptr: NonNull::dangling().as_ptr(),
            len: 0,
            data: AtomicPtr::new(owned.cast()),
            vtable: &VTables::<T>::OWNED_VTABLE,
        };

        let buf = unsafe { &*owned }.owner.as_ref();
        ret.ptr = buf.as_ptr();
        ret.len = buf.len();

        ret
    }

    /// Returns the number of bytes contained in this `Elems`.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let b = Elems::from(&b"hello"[..]);
    /// assert_eq!(b.len(), 5);
    /// ```
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the `Elems` has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Bytes;
    ///
    /// let b = Bytes::new();
    /// assert!(b.is_empty());
    /// ```
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if this is the only reference to the data and
    /// `Into<ElemsMut>` would avoid cloning the underlying buffer.
    ///
    /// Always returns false if the data is backed by a [static slice](Elems::from_static),
    /// or an [owner](Elems::from_owner).
    ///
    /// The result of this method may be invalidated immediately if another
    /// thread clones this value while this is being called. Ensure you have
    /// unique access to this value (`&mut Elems`) first if you need to be
    /// certain the result is valid (i.e. for safety reasons).
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let a = Elems::from(vec![1, 2, 3]);
    /// assert!(a.is_unique());
    /// let b = a.clone();
    /// assert!(!a.is_unique());
    /// ```
    pub fn is_unique(&self) -> bool {
        unsafe { (self.vtable.is_unique)(&self.data) }
    }

    /// Creates `Elems` instance from slice, by copying it.
    pub fn copy_from_slice(data: &[T]) -> Self {
        data.to_vec().into()
    }

    /// Returns a slice of self for the provided range.
    ///
    /// This will increment the reference count for the underlying memory and
    /// return a new `Elems` handle set to the slice.
    ///
    /// This operation is `O(1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let a = Elems::from(&b"hello world"[..]);
    /// let b = a.slice(2..5);
    ///
    /// assert_eq!(&b[..], b"llo");
    /// ```
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`, otherwise slicing
    /// will panic.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        use core::ops::Bound;

        let len = self.len();

        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).expect("out of range"),
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        assert!(
            begin <= end,
            "range start must not be greater than end: {:?} <= {:?}",
            begin,
            end,
        );
        assert!(
            end <= len,
            "range end out of bounds: {:?} <= {:?}",
            end,
            len,
        );

        if end == begin {
            return Elems::new();
        }

        let mut ret = self.clone();

        ret.len = end - begin;
        ret.ptr = unsafe { ret.ptr.add(begin) };

        ret
    }

    /// Returns a slice of self that is equivalent to the given `subset`.
    ///
    /// When processing a `Elems` buffer with other tools, one often gets a
    /// `&[u8]` which is in fact a slice of the `Elems`, i.e. a subset of it.
    /// This function turns that `&[u8]` into another `Elems`, as if one had
    /// called `self.slice()` with the offsets that correspond to `subset`.
    ///
    /// This operation is `O(1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let bytes = Elems::from(&b"012345678"[..]);
    /// let as_slice = bytes.as_ref();
    /// let subset = &as_slice[2..6];
    /// let subslice = bytes.slice_ref(&subset);
    /// assert_eq!(&subslice[..], b"2345");
    /// ```
    ///
    /// # Panics
    ///
    /// Requires that the given `sub` slice is in fact contained within the
    /// `Elems` buffer; otherwise this function will panic.
    pub fn slice_ref(&self, subset: &[u8]) -> Self {
        // Empty slice and empty Elems may have their pointers reset
        // so explicitly allow empty slice to be a subslice of any slice.
        if subset.is_empty() {
            return Elems::new();
        }

        let bytes_p = self.as_ptr() as usize;
        let bytes_len = self.len();

        let sub_p = subset.as_ptr() as usize;
        let sub_len = subset.len();

        assert!(
            sub_p >= bytes_p,
            "subset pointer ({:p}) is smaller than self pointer ({:p})",
            subset.as_ptr(),
            self.as_ptr(),
        );
        assert!(
            sub_p + sub_len <= bytes_p + bytes_len,
            "subset is out of bounds: self = ({:p}, {}), subset = ({:p}, {})",
            self.as_ptr(),
            bytes_len,
            subset.as_ptr(),
            sub_len,
        );

        let sub_offset = sub_p - bytes_p;

        self.slice(sub_offset..(sub_offset + sub_len))
    }

    /// Splits the bytes into two at the given index.
    ///
    /// Afterwards `self` contains elements `[0, at)`, and the returned `Elems`
    /// contains elements `[at, len)`. It's guaranteed that the memory does not
    /// move, that is, the address of `self` does not change, and the address of
    /// the returned slice is `at` bytes after that.
    ///
    /// This is an `O(1)` operation that just increases the reference count and
    /// sets a few indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let mut a = Elems::from(&b"hello world"[..]);
    /// let b = a.split_off(5);
    ///
    /// assert_eq!(&a[..], b"hello");
    /// assert_eq!(&b[..], b" world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    #[must_use = "consider Elems::truncate if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> Self {
        if at == self.len() {
            return Elems::new_empty_with_ptr(self.ptr.wrapping_add(at));
        }

        if at == 0 {
            return mem::replace(self, Elems::new_empty_with_ptr(self.ptr));
        }

        assert!(
            at <= self.len(),
            "split_off out of bounds: {:?} <= {:?}",
            at,
            self.len(),
        );

        let mut ret = self.clone();

        self.len = at;

        unsafe { ret.inc_start(at) };

        ret
    }

    /// Splits the bytes into two at the given index.
    ///
    /// Afterwards `self` contains elements `[at, len)`, and the returned
    /// `Elems` contains elements `[0, at)`.
    ///
    /// This is an `O(1)` operation that just increases the reference count and
    /// sets a few indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let mut a = Elems::from(&b"hello world"[..]);
    /// let b = a.split_to(5);
    ///
    /// assert_eq!(&a[..], b" world");
    /// assert_eq!(&b[..], b"hello");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    #[must_use = "consider Elems::advance if you don't need the other half"]
    pub fn split_to(&mut self, at: usize) -> Self {
        if at == self.len() {
            let end_ptr = self.ptr.wrapping_add(at);
            return mem::replace(self, Elems::new_empty_with_ptr(end_ptr));
        }

        if at == 0 {
            return Elems::new_empty_with_ptr(self.ptr);
        }

        assert!(
            at <= self.len(),
            "split_to out of bounds: {:?} <= {:?}",
            at,
            self.len(),
        );

        let mut ret = self.clone();

        unsafe { self.inc_start(at) };

        ret.len = at;
        ret
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    ///
    /// If `len` is greater than the buffer's current length, this has no
    /// effect.
    ///
    /// The [split_off](`Self::split_off()`) method can emulate `truncate`, but this causes the
    /// excess bytes to be returned instead of dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let mut buf = Elems::from(&b"hello world"[..]);
    /// buf.truncate(5);
    /// assert_eq!(buf, b"hello"[..]);
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            // The Vec "promotable" vtables do not store the capacity,
            // so we cannot truncate while using this repr. We *have* to
            // promote using `split_off` so the capacity can be stored.
            if self.vtable.promotable {
                drop(self.split_off(len));
            } else {
                self.len = len;
            }
        }
    }

    /// Clears the buffer, removing all data.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::Elems;
    ///
    /// let mut buf = Elems::from(&b"hello world"[..]);
    /// buf.clear();
    /// assert!(buf.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Try to convert self into `ElemsMut`.
    ///
    /// If `self` is unique for the entire original buffer, this will succeed
    /// and return a `ElemsMut` with the contents of `self` without copying.
    /// If `self` is not unique for the entire original buffer, this will fail
    /// and return self.
    ///
    /// This will also always fail if the buffer was constructed via either
    /// [from_owner](Elems::from_owner) or [from_static](Elems::from_static).
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::{Elems, ElemsMut};
    ///
    /// let bytes = Elems::from(b"hello".to_vec());
    /// assert_eq!(bytes.try_into_mut(), Ok(ElemsMut::from(&b"hello"[..])));
    /// ```
    pub fn try_into_mut(self) -> Result<ElemsMut<T>, Self> {
        if self.is_unique() {
            Ok(self.into())
        } else {
            Err(self)
        }
    }

    #[inline]
    pub(crate) unsafe fn with_vtable(
        ptr: *const T,
        len: usize,
        data: AtomicPtr<()>,
        vtable: &'static Vtable,
    ) -> Self {
        Elems {
            ptr,
            len,
            data,
            vtable,
        }
    }

    // private

    #[inline]
    fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    unsafe fn inc_start(&mut self, by: usize) {
        // should already be asserted, but debug assert for tests
        debug_assert!(self.len >= by, "internal: inc_start out of bounds");
        self.len -= by;
        self.ptr = self.ptr.add(by);
    }

    /// Advances the start of this buffer by `cnt` elements.
    #[inline]
    pub fn advance(&mut self, cnt: usize) {
        assert!(
            cnt <= self.len(),
            "cannot advance past `remaining`: {:?} <= {:?}",
            cnt,
            self.len(),
        );

        unsafe {
            self.inc_start(cnt);
        }
    }
}

// Vtable must enforce this behavior
unsafe impl Send for Elems {}
unsafe impl Sync for Elems {}

impl<T: Pod> Drop for Elems<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { (self.vtable.drop)(&mut self.data, self.ptr.cast::<u8>(), self.len) }
    }
}

impl<T: Pod> Clone for Elems<T> {
    #[inline]
    fn clone(&self) -> Self {
        let bytes = unsafe { (self.vtable.clone)(&self.data, self.ptr.cast::<u8>(), self.len) };
        bytes.recover::<T>()
    }
}

impl Buf for Elems {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(
            cnt <= self.len(),
            "cannot advance past `remaining`: {:?} <= {:?}",
            cnt,
            self.len(),
        );

        unsafe {
            self.inc_start(cnt);
        }
    }

    fn copy_to_bytes(&mut self, len: usize) -> Self {
        self.split_to(len)
    }
}

impl<T: Pod> Deref for Elems<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Pod> AsRef<[T]> for Elems<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Pod + hash::Hash> hash::Hash for Elems<T> {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.as_slice().hash(state);
    }
}

impl<T: Pod> Borrow<[T]> for Elems<T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl IntoIterator for Elems {
    type Item = u8;
    type IntoIter = IntoIter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T: Pod> IntoIterator for &'a Elems<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<T: Pod> FromIterator<T> for Elems<T> {
    fn from_iter<I: IntoIterator<Item = T>>(into_iter: I) -> Self {
        Vec::from_iter(into_iter).into()
    }
}

// impl Eq

impl<T: Pod + PartialEq> PartialEq for Elems<T> {
    fn eq(&self, other: &Elems<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Pod + PartialOrd> PartialOrd for Elems<T> {
    fn partial_cmp(&self, other: &Elems<T>) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Pod + Ord> Ord for Elems<T> {
    fn cmp(&self, other: &Elems<T>) -> cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: Pod + Eq> Eq for Elems<T> {}

impl<T: Pod + PartialEq> PartialEq<[T]> for Elems<T> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}

impl<T: Pod + PartialOrd> PartialOrd<[T]> for Elems<T> {
    fn partial_cmp(&self, other: &[T]) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other)
    }
}

impl<T: Pod + PartialEq> PartialEq<Elems<T>> for [T] {
    fn eq(&self, other: &Elems<T>) -> bool {
        *other == *self
    }
}

impl<T: Pod + PartialOrd> PartialOrd<Elems<T>> for [T] {
    fn partial_cmp(&self, other: &Elems<T>) -> Option<cmp::Ordering> {
        <[T] as PartialOrd<[T]>>::partial_cmp(self, other)
    }
}

impl PartialEq<str> for Elems {
    fn eq(&self, other: &str) -> bool {
        self.as_slice() == other.as_bytes()
    }
}

impl PartialOrd<str> for Elems {
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_bytes())
    }
}

impl PartialEq<Elems> for str {
    fn eq(&self, other: &Elems) -> bool {
        *other == *self
    }
}

impl PartialOrd<Elems> for str {
    fn partial_cmp(&self, other: &Elems) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self.as_bytes(), other)
    }
}

impl<T: Pod + PartialEq> PartialEq<Vec<T>> for Elems<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        *self == other[..]
    }
}

impl<T: Pod + PartialOrd> PartialOrd<Vec<T>> for Elems<T> {
    fn partial_cmp(&self, other: &Vec<T>) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(&other[..])
    }
}

impl<T: Pod + PartialEq> PartialEq<Elems<T>> for Vec<T> {
    fn eq(&self, other: &Elems<T>) -> bool {
        *other == *self
    }
}

impl<T: Pod + PartialOrd> PartialOrd<Elems<T>> for Vec<T> {
    fn partial_cmp(&self, other: &Elems<T>) -> Option<cmp::Ordering> {
        <[T] as PartialOrd<[T]>>::partial_cmp(self, other)
    }
}

impl PartialEq<String> for Elems {
    fn eq(&self, other: &String) -> bool {
        *self == other[..]
    }
}

impl PartialOrd<String> for Elems {
    fn partial_cmp(&self, other: &String) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_bytes())
    }
}

impl PartialEq<Elems> for String {
    fn eq(&self, other: &Elems) -> bool {
        *other == *self
    }
}

impl PartialOrd<Elems> for String {
    fn partial_cmp(&self, other: &Elems) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self.as_bytes(), other)
    }
}

impl<T: Pod + PartialEq> PartialEq<Elems<T>> for &[T] {
    fn eq(&self, other: &Elems<T>) -> bool {
        other.as_slice() == *self
    }
}

impl<T: Pod + PartialOrd> PartialOrd<Elems<T>> for &[T] {
    fn partial_cmp(&self, other: &Elems<T>) -> Option<cmp::Ordering> {
        <[T] as PartialOrd<[T]>>::partial_cmp(self, other)
    }
}

impl PartialEq<Elems> for &str {
    fn eq(&self, other: &Elems) -> bool {
        *other == *self
    }
}

impl PartialOrd<Elems> for &str {
    fn partial_cmp(&self, other: &Elems) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self.as_bytes(), other)
    }
}

// todo check or implement
impl<'a, T: ?Sized> PartialEq<&'a T> for Elems
where
    Elems: PartialEq<T>,
{
    fn eq(&self, other: &&'a T) -> bool {
        *self == **other
    }
}
// todo check or implement
impl<'a, T: ?Sized> PartialOrd<&'a T> for Elems
where
    Elems: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &&'a T) -> Option<cmp::Ordering> {
        self.partial_cmp(&**other)
    }
}

// impl From

impl<T: Pod> Default for Elems<T> {
    #[inline]
    fn default() -> Elems<T> {
        Elems::new()
    }
}

impl<T: Pod> From<&'static [T]> for Elems<T> {
    fn from(slice: &'static [T]) -> Elems<T> {
        Elems::from_static(slice)
    }
}

impl From<&'static str> for Elems {
    fn from(slice: &'static str) -> Elems {
        Elems::from_static(slice.as_bytes())
    }
}

impl<T: Pod> From<Vec<T>> for Elems<T> {
    fn from(vec: Vec<T>) -> Self {
        let mut vec = ManuallyDrop::new(vec);
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();

        // Avoid an extra allocation if possible.
        if len == cap {
            let vec = ManuallyDrop::into_inner(vec);
            return Elems::from(vec.into_boxed_slice());
        }

        let shared = Box::new(Shared {
            buf: ptr.cast::<u8>(), // TODO, FIXME: check if this is safe
            cap,
            ref_cnt: AtomicUsize::new(1),
        });

        let shared = Box::into_raw(shared);
        // The pointer should be aligned, so this assert should
        // always succeed.
        debug_assert!(
            0 == (shared as usize & KIND_MASK),
            "internal: Box<Shared> should have an aligned pointer",
        );
        Elems {
            ptr,
            len,
            data: AtomicPtr::new(shared as _),
            vtable: &VTables::<T>::SHARED_VTABLE, // TODO, FIXME: check if this is safe
        }
    }
}

impl<T: Pod> From<Box<[T]>> for Elems<T> {
    fn from(slice: Box<[T]>) -> Self {
        // Box<[u8]> doesn't contain a heap allocation for empty slices,
        // so the pointer isn't aligned enough for the KIND_VEC stashing to
        // work.
        if slice.is_empty() {
            return Elems::new();
        }

        let len = slice.len();
        let ptr = Box::into_raw(slice) as *mut T;

        if ptr as usize & 0x1 == 0 {
            let data = ptr_map(ptr, |addr| addr | KIND_VEC);
            Elems {
                ptr,
                len,
                data: AtomicPtr::new(data.cast()),
                vtable: &VTables::<T>::PROMOTABLE_EVEN_VTABLE,
            }
        } else {
            Elems {
                ptr,
                len,
                data: AtomicPtr::new(ptr.cast()),
                vtable: &VTables::<T>::PROMOTABLE_ODD_VTABLE,
            }
        }
    }
}

impl<T: Pod> From<Elems<T>> for ElemsMut<T> {
    /// Convert self into `ElemsMut`.
    ///
    /// If `bytes` is unique for the entire original buffer, this will return a
    /// `ElemsMut` with the contents of `bytes` without copying.
    /// If `bytes` is not unique for the entire original buffer, this will make
    /// a copy of `bytes` subset of the original buffer in a new `ElemsMut`.
    ///
    /// # Examples
    ///
    /// ```
    /// use elems::{Elems, ElemsMut};
    ///
    /// let bytes = Elems::from(b"hello".to_vec());
    /// assert_eq!(ElemsMut::from(bytes), ElemsMut::from(&b"hello"[..]));
    /// ```
    fn from(bytes: Elems<T>) -> Self {
        let bytes = ManuallyDrop::new(bytes);
        unsafe {
            (bytes.vtable.to_mut)(&bytes.data, bytes.ptr.cast::<u8>(), bytes.len).recover::<T>()
        }
    }
}

impl From<String> for Elems {
    fn from(s: String) -> Elems {
        Elems::from(s.into_bytes())
    }
}

//todo implement correct casts
impl<T: Pod> From<Elems<T>> for Vec<T> {
    fn from(bytes: Elems<T>) -> Vec<T> {
        let bytes = ManuallyDrop::new(bytes);
        unsafe {
            let (ptr, len, cap) =
                (bytes.vtable.to_vec)(&bytes.data, bytes.ptr.cast::<u8>(), bytes.len);
            Vec::from_raw_parts(ptr.cast::<T>(), len, cap)
        }
    }
}

// ===== impl Vtable =====

impl fmt::Debug for Vtable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vtable")
            .field("clone", &(self.clone as *const ()))
            .field("drop", &(self.drop as *const ()))
            .finish()
    }
}

// ===== impl StaticVtable =====

struct VTables<T: Pod>(core::marker::PhantomData<T>);
impl<T: Pod> VTables<T> {
    const STATIC_VTABLE: Vtable = Vtable {
        clone: static_clone::<T>,
        to_vec: static_to_vec::<T>,
        to_mut: static_to_mut::<T>,
        is_unique: static_is_unique,
        drop: static_drop,
        promotable: false,
    };

    const OWNED_VTABLE: Vtable = Vtable {
        clone: owned_clone::<T>,
        to_vec: owned_to_vec::<T>,
        to_mut: owned_to_mut::<T>,
        is_unique: owned_is_unique,
        drop: owned_drop,
        promotable: false,
    };

    const PROMOTABLE_EVEN_VTABLE: Vtable = Vtable {
        clone: promotable_even_clone::<T>,
        to_vec: promotable_even_to_vec::<T>,
        to_mut: promotable_even_to_mut::<T>,
        is_unique: promotable_is_unique,
        drop: promotable_even_drop,
        promotable: true,
    };

    const PROMOTABLE_ODD_VTABLE: Vtable = Vtable {
        clone: promotable_odd_clone::<T>,
        to_vec: promotable_odd_to_vec::<T>,
        to_mut: promotable_odd_to_mut::<T>,
        is_unique: promotable_is_unique,
        drop: promotable_odd_drop,
        promotable: true,
    };

    const SHARED_VTABLE: Vtable = Vtable {
        clone: shared_clone::<T>,
        to_vec: shared_to_vec::<T>,
        to_mut: shared_to_mut::<T>,
        is_unique: shared_is_unique,
        drop: shared_drop,
        promotable: false,
    };
}

unsafe fn static_clone<T: Pod>(_: &AtomicPtr<()>, ptr: *const u8, len: usize) -> RawElems {
    let slice = slice::from_raw_parts(ptr.cast::<T>(), len);
    RawElems::erase(Elems::from_static(slice))
}

unsafe fn static_to_vec<T: Pod>(
    _: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> (*mut u8, usize, usize) {
    let slice = slice::from_raw_parts(ptr.cast::<T>(), len);
    let mut v = ManuallyDrop::new(slice.to_vec());
    (v.as_mut_ptr().cast::<u8>(), v.len(), v.capacity())
}

unsafe fn static_to_mut<T: Pod>(_: &AtomicPtr<()>, ptr: *const u8, len: usize) -> RawElemsMut {
    let slice = slice::from_raw_parts(ptr.cast::<T>(), len);
    RawElemsMut::erase(ElemsMut::<T>::from(slice))
}

fn static_is_unique(_: &AtomicPtr<()>) -> bool {
    false
}

unsafe fn static_drop(_: &mut AtomicPtr<()>, _: *const u8, _: usize) {
    // nothing to drop for &'static [u8]
}

// ===== impl OwnedVtable =====

#[repr(C)]
struct OwnedLifetime {
    ref_cnt: AtomicUsize,
    drop: unsafe fn(*mut ()),
}

#[repr(C)]
struct Owned<T> {
    lifetime: OwnedLifetime,
    owner: T,
}

unsafe fn owned_box_and_drop<T>(ptr: *mut ()) {
    let b: Box<Owned<T>> = Box::from_raw(ptr as _);
    drop(b);
}

unsafe fn owned_clone<T: Pod>(data: &AtomicPtr<()>, ptr: *const u8, len: usize) -> RawElems {
    let owned = data.load(Ordering::Relaxed);
    let ref_cnt = &(*owned.cast::<OwnedLifetime>()).ref_cnt;
    let old_cnt = ref_cnt.fetch_add(1, Ordering::Relaxed);
    if old_cnt > usize::MAX >> 1 {
        crate::abort()
    }

    RawElems {
        ptr,
        len,
        data: AtomicPtr::new(owned as _),
        vtable: &VTables::<T>::OWNED_VTABLE,
    }
}

unsafe fn owned_to_vec<T: Pod>(
    _data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> (*mut u8, usize, usize) {
    let mut v = ManuallyDrop::new(slice::from_raw_parts(ptr.cast::<T>(), len).to_vec());
    (v.as_mut_ptr().cast::<u8>(), v.len(), v.capacity())
}

unsafe fn owned_to_mut<T: Pod>(data: &AtomicPtr<()>, ptr: *const u8, len: usize) -> RawElemsMut {
    let bytes_mut = ElemsMut::from_vec(slice::from_raw_parts(ptr.cast::<T>(), len).to_vec());
    owned_drop_impl(data.load(Ordering::Relaxed));
    RawElemsMut::erase(bytes_mut)
}

unsafe fn owned_is_unique(_data: &AtomicPtr<()>) -> bool {
    false
}

unsafe fn owned_drop_impl(owned: *mut ()) {
    let lifetime = owned.cast::<OwnedLifetime>();
    let ref_cnt = &(*lifetime).ref_cnt;

    let old_cnt = ref_cnt.fetch_sub(1, Ordering::Release);
    if old_cnt != 1 {
        return;
    }
    ref_cnt.load(Ordering::Acquire);

    let drop_fn = &(*lifetime).drop;
    drop_fn(owned)
}

unsafe fn owned_drop(data: &mut AtomicPtr<()>, _ptr: *const u8, _len: usize) {
    let owned = data.load(Ordering::Relaxed);
    owned_drop_impl(owned);
}

// ===== impl PromotableVtable =====

unsafe fn promotable_even_clone<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> RawElems {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        RawElems::erase(shallow_clone_arc::<T>(shared.cast(), ptr.cast::<T>(), len))
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        let buf = ptr_map(shared.cast(), |addr| addr & !KIND_MASK);
        RawElems::erase(shallow_clone_vec::<T>(
            data,
            shared,
            buf,
            ptr.cast::<T>(),
            len,
        ))
    }
}

unsafe fn promotable_to_vec<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const T,
    len: usize,
    f: fn(*mut ()) -> *mut T,
) -> (*mut u8, usize, usize) {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        shared_to_vec_impl(shared.cast(), ptr, len)
    } else {
        // If Elems holds a Vec, then the offset must be 0.
        debug_assert_eq!(kind, KIND_VEC);

        let buf = f(shared);

        let cap = offset_from(ptr, buf) + len;

        // Copy back buffer
        ptr::copy(ptr, buf, len);

        (buf.cast::<u8>(), len, cap)
    }
}

unsafe fn promotable_to_mut<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const T,
    len: usize,
    f: fn(*mut ()) -> *mut T,
) -> RawElemsMut {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        shared_to_mut_impl::<T>(shared.cast(), ptr, len)
    } else {
        // KIND_VEC is a view of an underlying buffer at a certain offset.
        // The ptr + len always represents the end of that buffer.
        // Before truncating it, it is first promoted to KIND_ARC.
        // Thus, we can safely reconstruct a Vec from it without leaking memory.
        debug_assert_eq!(kind, KIND_VEC);

        let buf = f(shared);
        let off = offset_from(ptr, buf);
        let cap = off + len;
        let v = Vec::from_raw_parts(buf, cap, cap);

        let mut b = ElemsMut::from_vec(v);
        b.advance_unchecked(off);
        RawElemsMut::erase(b)
    }
}

unsafe fn promotable_even_to_vec<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> (*mut u8, usize, usize) {
    promotable_to_vec(data, ptr.cast::<T>(), len, |shared| {
        ptr_map(shared.cast(), |addr| addr & !KIND_MASK)
    })
}

unsafe fn promotable_even_to_mut<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> RawElemsMut {
    promotable_to_mut::<T>(data, ptr.cast::<T>(), len, |shared| {
        ptr_map(shared.cast(), |addr| addr & !KIND_MASK)
    })
}

unsafe fn promotable_even_drop(data: &mut AtomicPtr<()>, ptr: *const u8, len: usize) {
    data.with_mut(|shared| {
        let shared = *shared;
        let kind = shared as usize & KIND_MASK;

        if kind == KIND_ARC {
            release_shared(shared.cast());
        } else {
            debug_assert_eq!(kind, KIND_VEC);
            let buf = ptr_map(shared.cast(), |addr| addr & !KIND_MASK);
            free_boxed_slice(buf, ptr, len);
        }
    });
}

unsafe fn promotable_odd_clone<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> RawElems {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        RawElems::erase(shallow_clone_arc::<T>(shared as _, ptr.cast::<T>(), len))
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        RawElems::erase(shallow_clone_vec::<T>(
            data,
            shared,
            shared.cast(),
            ptr.cast::<T>(),
            len,
        ))
    }
}

unsafe fn promotable_odd_to_vec<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> (*mut u8, usize, usize) {
    promotable_to_vec(data, ptr.cast::<T>(), len, |shared| shared.cast())
}

unsafe fn promotable_odd_to_mut<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> RawElemsMut {
    promotable_to_mut::<T>(data, ptr.cast::<T>(), len, |shared| shared.cast())
}

unsafe fn promotable_odd_drop(data: &mut AtomicPtr<()>, ptr: *const u8, len: usize) {
    data.with_mut(|shared| {
        let shared = *shared;
        let kind = shared as usize & KIND_MASK;

        if kind == KIND_ARC {
            release_shared(shared.cast());
        } else {
            debug_assert_eq!(kind, KIND_VEC);

            free_boxed_slice(shared.cast(), ptr, len);
        }
    });
}

unsafe fn promotable_is_unique(data: &AtomicPtr<()>) -> bool {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        let ref_cnt = (*shared.cast::<Shared>()).ref_cnt.load(Ordering::Relaxed);
        ref_cnt == 1
    } else {
        true
    }
}

unsafe fn free_boxed_slice(buf: *mut u8, offset: *const u8, len: usize) {
    let cap = offset_from(offset, buf) + len;
    dealloc(buf, Layout::from_size_align(cap, 1).unwrap())
}

// ===== impl SharedVtable =====

struct Shared {
    // Holds arguments to dealloc upon Drop, but otherwise doesn't use them
    buf: *mut u8,
    cap: usize,
    ref_cnt: AtomicUsize,
}

impl Drop for Shared {
    fn drop(&mut self) {
        unsafe { dealloc(self.buf, Layout::from_size_align(self.cap, 1).unwrap()) }
    }
}

// Assert that the alignment of `Shared` is divisible by 2.
// This is a necessary invariant since we depend on allocating `Shared` a
// shared object to implicitly carry the `KIND_ARC` flag in its pointer.
// This flag is set when the LSB is 0.
const _: [(); 0 - mem::align_of::<Shared>() % 2] = []; // Assert that the alignment of `Shared` is divisible by 2.

const KIND_ARC: usize = 0b0;
const KIND_VEC: usize = 0b1;
const KIND_MASK: usize = 0b1;

unsafe fn shared_clone<T: Pod>(data: &AtomicPtr<()>, ptr: *const u8, len: usize) -> RawElems {
    let shared = data.load(Ordering::Relaxed);
    RawElems::erase(shallow_clone_arc::<T>(shared as _, ptr.cast::<T>(), len))
}

unsafe fn shared_to_vec_impl<T: Pod>(
    shared: *mut Shared,
    ptr: *const T,
    len: usize,
) -> (*mut u8, usize, usize) {
    // Check that the ref_cnt is 1 (unique).
    //
    // If it is unique, then it is set to 0 with AcqRel fence for the same
    // reason in release_shared.
    //
    // Otherwise, we take the other branch and call release_shared.
    if (*shared)
        .ref_cnt
        .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        // Deallocate the `Shared` instance without running its destructor.
        let shared = *Box::from_raw(shared);
        let shared = ManuallyDrop::new(shared);
        let buf = shared.buf;
        let cap = shared.cap;

        // Copy back buffer
        ptr::copy(ptr, buf.cast::<T>(), len);

        (buf, len, cap)
    } else {
        let v = slice::from_raw_parts(ptr, len).to_vec();
        release_shared(shared);
        let mut v = ManuallyDrop::new(v);
        (v.as_mut_ptr().cast::<u8>(), v.len(), v.capacity())
    }
}

unsafe fn shared_to_vec<T: Pod>(
    data: &AtomicPtr<()>,
    ptr: *const u8,
    len: usize,
) -> (*mut u8, usize, usize) {
    shared_to_vec_impl::<T>(data.load(Ordering::Relaxed).cast(), ptr.cast::<T>(), len)
}

unsafe fn shared_to_mut_impl<T: Pod>(
    shared: *mut Shared,
    ptr: *const T,
    len: usize,
) -> RawElemsMut {
    // The goal is to check if the current handle is the only handle
    // that currently has access to the buffer. This is done by
    // checking if the `ref_cnt` is currently 1.
    //
    // The `Acquire` ordering synchronizes with the `Release` as
    // part of the `fetch_sub` in `release_shared`. The `fetch_sub`
    // operation guarantees that any mutations done in other threads
    // are ordered before the `ref_cnt` is decremented. As such,
    // this `Acquire` will guarantee that those mutations are
    // visible to the current thread.
    //
    // Otherwise, we take the other branch, copy the data and call `release_shared`.
    if (*shared).ref_cnt.load(Ordering::Acquire) == 1 {
        // Deallocate the `Shared` instance without running its destructor.
        let shared = *Box::from_raw(shared);
        let shared = ManuallyDrop::new(shared);
        let buf = shared.buf.cast::<T>();
        let cap = shared.cap;

        // Rebuild Vec
        let off = offset_from(ptr, buf);
        let v = Vec::from_raw_parts(buf, len + off, cap);

        let mut b = ElemsMut::from_vec(v);
        b.advance_unchecked(off);
        RawElemsMut::erase(b)
    } else {
        // Copy the data from Shared in a new Vec, then release it
        let v = slice::from_raw_parts(ptr, len).to_vec();
        release_shared(shared);
        RawElemsMut::erase(ElemsMut::<T>::from_vec(v))
    }
}

unsafe fn shared_to_mut<T: Pod>(data: &AtomicPtr<()>, ptr: *const u8, len: usize) -> RawElemsMut {
    shared_to_mut_impl::<T>(data.load(Ordering::Relaxed).cast(), ptr.cast::<T>(), len)
}

pub(crate) unsafe fn shared_is_unique(data: &AtomicPtr<()>) -> bool {
    let shared = data.load(Ordering::Acquire);
    let ref_cnt = (*shared.cast::<Shared>()).ref_cnt.load(Ordering::Relaxed);
    ref_cnt == 1
}

unsafe fn shared_drop(data: &mut AtomicPtr<()>, _ptr: *const u8, _len: usize) {
    data.with_mut(|shared| {
        release_shared(shared.cast());
    });
}

unsafe fn shallow_clone_arc<T: Pod>(shared: *mut Shared, ptr: *const T, len: usize) -> Elems<T> {
    let old_size = (*shared).ref_cnt.fetch_add(1, Ordering::Relaxed);

    if old_size > usize::MAX >> 1 {
        crate::abort();
    }

    Elems {
        ptr,
        len,
        data: AtomicPtr::new(shared as _),
        vtable: &VTables::<T>::SHARED_VTABLE,
    }
}

#[cold]
unsafe fn shallow_clone_vec<T: Pod>(
    atom: &AtomicPtr<()>,
    ptr: *const (),
    buf: *mut u8,
    offset: *const T,
    len: usize,
) -> Elems<T> {
    // If the buffer is still tracked in a `Vec<u8>`. It is time to
    // promote the vec to an `Arc`. This could potentially be called
    // concurrently, so some care must be taken.

    // First, allocate a new `Shared` instance containing the
    // `Vec` fields. It's important to note that `ptr`, `len`,
    // and `cap` cannot be mutated without having `&mut self`.
    // This means that these fields will not be concurrently
    // updated and since the buffer hasn't been promoted to an
    // `Arc`, those three fields still are the components of the
    // vector.
    let shared = Box::new(Shared {
        buf: buf.cast::<u8>(),
        cap: offset_from(offset, buf.cast::<T>()) + len,
        // Initialize refcount to 2. One for this reference, and one
        // for the new clone that will be returned from
        // `shallow_clone`.
        ref_cnt: AtomicUsize::new(2),
    });

    let shared = Box::into_raw(shared);

    // The pointer should be aligned, so this assert should
    // always succeed.
    debug_assert!(
        0 == (shared as usize & KIND_MASK),
        "internal: Box<Shared> should have an aligned pointer",
    );

    // Try compare & swapping the pointer into the `arc` field.
    // `Release` is used synchronize with other threads that
    // will load the `arc` field.
    //
    // If the `compare_exchange` fails, then the thread lost the
    // race to promote the buffer to shared. The `Acquire`
    // ordering will synchronize with the `compare_exchange`
    // that happened in the other thread and the `Shared`
    // pointed to by `actual` will be visible.
    match atom.compare_exchange(ptr as _, shared as _, Ordering::AcqRel, Ordering::Acquire) {
        Ok(actual) => {
            debug_assert!(actual as usize == ptr as usize);
            // The upgrade was successful, the new handle can be
            // returned.
            Elems {
                ptr: offset.cast::<T>(),
                len,
                data: AtomicPtr::new(shared as _),
                vtable: &VTables::<T>::SHARED_VTABLE,
            }
        }
        Err(actual) => {
            // The upgrade failed, a concurrent clone happened. Release
            // the allocation that was made in this thread, it will not
            // be needed.
            let shared = Box::from_raw(shared);
            mem::forget(*shared);

            // Buffer already promoted to shared storage, so increment ref
            // count.
            shallow_clone_arc(actual as _, offset, len)
        }
    }
}

unsafe fn release_shared(ptr: *mut Shared) {
    // `Shared` storage... follow the drop steps from Arc.
    if (*ptr).ref_cnt.fetch_sub(1, Ordering::Release) != 1 {
        return;
    }

    // This fence is needed to prevent reordering of use of the data and
    // deletion of the data.  Because it is marked `Release`, the decreasing
    // of the reference count synchronizes with this `Acquire` fence. This
    // means that use of the data happens before decreasing the reference
    // count, which happens before this fence, which happens before the
    // deletion of the data.
    //
    // As explained in the [Boost documentation][1],
    //
    // > It is important to enforce any possible access to the object in one
    // > thread (through an existing reference) to *happen before* deleting
    // > the object in a different thread. This is achieved by a "release"
    // > operation after dropping a reference (any access to the object
    // > through this reference must obviously happened before), and an
    // > "acquire" operation before deleting the object.
    //
    // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
    //
    // Thread sanitizer does not support atomic fences. Use an atomic load
    // instead.
    (*ptr).ref_cnt.load(Ordering::Acquire);

    // Drop the data
    drop(Box::from_raw(ptr));
}

// Ideally we would always use this version of `ptr_map` since it is strict
// provenance compatible, but it results in worse codegen. We will however still
// use it on miri because it gives better diagnostics for people who test bytes
// code with miri.
//
// See https://github.com/tokio-rs/bytes/pull/545 for more info.
#[cfg(miri)]
fn ptr_map<F, T>(ptr: *mut T, f: F) -> *mut T
where
    F: FnOnce(usize) -> usize,
{
    let old_addr = ptr as usize;
    let new_addr = f(old_addr);
    let diff = new_addr.wrapping_sub(old_addr);
    ptr.wrapping_add(diff)
}

#[cfg(not(miri))]
fn ptr_map<F, T>(ptr: *mut T, f: F) -> *mut T
where
    F: FnOnce(usize) -> usize,
{
    let old_addr = ptr as usize;
    let new_addr = f(old_addr);
    new_addr as *mut T
}

fn without_provenance<T>(ptr: usize) -> *const T {
    core::ptr::null::<u8>().wrapping_add(ptr).cast::<T>()
}

// compile-fails

/// ```compile_fail
/// use elems::Elems;
/// #[deny(unused_must_use)]
/// {
///     let mut b1 = Elems::from("hello world");
///     b1.split_to(6);
/// }
/// ```
fn _split_to_must_use() {}

/// ```compile_fail
/// use elems::Elems;
/// #[deny(unused_must_use)]
/// {
///     let mut b1 = Elems::from("hello world");
///     b1.split_off(6);
/// }
/// ```
fn _split_off_must_use() {}

// fuzz tests
#[cfg(all(test, loom))]
mod fuzz {
    use loom::sync::Arc;
    use loom::thread;

    use super::Elems;
    #[test]
    fn bytes_cloning_vec() {
        loom::model(|| {
            let a = Elems::from(b"abcdefgh".to_vec());
            let addr = a.as_ptr() as usize;

            // test the Elems::clone is Sync by putting it in an Arc
            let a1 = Arc::new(a);
            let a2 = a1.clone();

            let t1 = thread::spawn(move || {
                let b: Elems = (*a1).clone();
                assert_eq!(b.as_ptr() as usize, addr);
            });

            let t2 = thread::spawn(move || {
                let b: Elems = (*a2).clone();
                assert_eq!(b.as_ptr() as usize, addr);
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }
}
