//! Byte inputs from JavaScript, realm-independent.
//!
//! `instanceof Uint8Array` is false for a typed array created in another realm
//! (a worker, an iframe, Node's `vm`, or the WASM glue's own global), so the
//! checks here use `ArrayBuffer.isView` and `Symbol.toStringTag`, which hold
//! in every realm.

use js_sys::{ArrayBuffer, Reflect, Symbol, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};

/// Copy a JS byte source into a `Vec<u8>`.
///
/// Accepts any `ArrayBufferView` (`Uint8Array`, `DataView`, `Buffer`, …,
/// using its own offset and length) or an `ArrayBuffer`. When `allow_text` is
/// set, a string is accepted and encoded as UTF-8. Anything else is an error
/// naming what was received.
pub(crate) fn bytes_from_js(value: &JsValue, allow_text: bool) -> Result<Vec<u8>, JsValue> {
    if allow_text {
        if let Some(text) = value.as_string() {
            return Ok(text.into_bytes());
        }
    }
    if ArrayBuffer::is_view(value) {
        let buffer = Reflect::get(value, &JsValue::from_str("buffer"))?;
        let offset = number_prop(value, "byteOffset")?;
        let length = number_prop(value, "byteLength")?;
        let buffer: ArrayBuffer = buffer.unchecked_into();
        return Ok(Uint8Array::new_with_byte_offset_and_length(&buffer, offset, length).to_vec());
    }
    if is_array_buffer(value) {
        return Ok(Uint8Array::new(value).to_vec());
    }
    let expected = if allow_text {
        "a string, an ArrayBuffer or an ArrayBufferView"
    } else {
        "an ArrayBuffer or an ArrayBufferView"
    };
    Err(JsValue::from_str(&format!(
        "Expected {}, got {}",
        expected,
        type_tag(value)
    )))
}

/// Like [`bytes_from_js`], but `undefined`/`null` mean "no body".
pub(crate) fn optional_bytes_from_js(value: &JsValue) -> Result<Vec<u8>, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Ok(Vec::new());
    }
    bytes_from_js(value, true)
}

fn number_prop(value: &JsValue, name: &str) -> Result<u32, JsValue> {
    let n = Reflect::get(value, &JsValue::from_str(name))?
        .as_f64()
        .ok_or_else(|| JsValue::from_str(&format!("{} is not a number", name)))?;
    if n < 0.0 || n > f64::from(u32::MAX) || n.fract() != 0.0 {
        return Err(JsValue::from_str(&format!("{} out of range: {}", name, n)));
    }
    Ok(n as u32)
}

fn is_array_buffer(value: &JsValue) -> bool {
    value.is_object() && to_string_tag(value).as_deref() == Some("ArrayBuffer")
}

/// `value[Symbol.toStringTag]`, which built-ins define identically in every realm.
fn to_string_tag(value: &JsValue) -> Option<String> {
    Reflect::get(value, &Symbol::to_string_tag())
        .ok()
        .and_then(|tag| tag.as_string())
}

fn type_tag(value: &JsValue) -> String {
    if value.is_object() {
        return to_string_tag(value).unwrap_or_else(|| "object".into());
    }
    value
        .js_typeof()
        .as_string()
        .unwrap_or_else(|| "unknown".into())
}
