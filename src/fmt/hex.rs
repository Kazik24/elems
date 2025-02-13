use core::fmt::{Formatter, LowerHex, Result, UpperHex};

use super::ElemsRef;
use crate::{Elems, ElemsMut};

impl LowerHex for ElemsRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for &b in self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl UpperHex for ElemsRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for &b in self.0 {
            write!(f, "{:02X}", b)?;
        }
        Ok(())
    }
}

fmt_impl!(LowerHex, Elems);
fmt_impl!(LowerHex, ElemsMut);
fmt_impl!(UpperHex, Elems);
fmt_impl!(UpperHex, ElemsMut);
