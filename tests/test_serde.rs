#![cfg(feature = "serde")]
#![warn(rust_2018_idioms)]

use serde_test::{assert_tokens, Token};

#[test]
fn test_ser_de_empty() {
    let b = elems::Elems::new();
    assert_tokens(&b, &[Token::Elems(b"")]);
    let b = elems::ElemsMut::with_capacity(0);
    assert_tokens(&b, &[Token::Elems(b"")]);
}

#[test]
fn test_ser_de() {
    let b = elems::Elems::from(&b"bytes"[..]);
    assert_tokens(&b, &[Token::Elems(b"bytes")]);
    let b = elems::ElemsMut::from(&b"bytes"[..]);
    assert_tokens(&b, &[Token::Elems(b"bytes")]);
}
