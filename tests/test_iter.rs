#![warn(rust_2018_idioms)]

use elems::{buf::IntoIter, Elems};

#[test]
fn iter_len() {
    let buf = Elems::from_static(b"hello world");
    let iter = IntoIter::new(buf);

    assert_eq!(iter.size_hint(), (11, Some(11)));
    assert_eq!(iter.len(), 11);
}

#[test]
fn empty_iter_len() {
    let buf = Elems::new();
    let iter = IntoIter::new(buf);

    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.len(), 0);
}
