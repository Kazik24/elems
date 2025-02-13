macro_rules! fmt_impl {
    ($tr:ident, $ty:ty) => {
        impl $tr for $ty {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                $tr::fmt(&ElemsRef(self.as_ref()), f)
            }
        }
    };
}

mod debug;
mod hex;

/// `ElemsRef` is not a part of public API of bytes crate.
struct ElemsRef<'a>(&'a [u8]);
