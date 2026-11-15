use std::collections::HashSet;

use pyo3::{
    IntoPyObjectExt, PyResult, exceptions,
    prelude::*,
    pybacked::PyBackedStr,
    types::{PyBytes, PyDict, PyList},
};
use rustc_hash::FxHashMap as HashMap;

use crate::{CoreBPE, Rank, byte_pair_encode};

fn is_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn bytes_repr(bytes: &[u8]) -> String {
    let mut out = String::from("b'");
    for &byte in bytes {
        match byte {
            b'\'' => out.push_str("\\'"),
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push('\'');
    out
}

fn parse_bpe_error(line: &[u8], source: &str) -> PyErr {
    exceptions::PyValueError::new_err(format!(
        "Error parsing line {} in {source}",
        bytes_repr(line)
    ))
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn decode_base64(input: &[u8]) -> Option<Vec<u8>> {
    if input.is_empty() || input.len() % 4 != 0 {
        return None;
    }

    let padding = input.iter().rev().take_while(|&&byte| byte == b'=').count();
    if padding > 2 {
        return None;
    }

    let output_len = input.len() / 4 * 3 - padding;
    let mut output = Vec::with_capacity(output_len);

    for (chunk_index, chunk) in input.chunks_exact(4).enumerate() {
        let is_last = chunk_index == input.len() / 4 - 1;
        if !is_last && chunk.contains(&b'=') {
            return None;
        }

        let a = base64_value(chunk[0])?;
        let b = base64_value(chunk[1])?;
        output.push((a << 2) | (b >> 4));

        match (chunk[2], chunk[3]) {
            (b'=', b'=') if is_last => {}
            (b'=', _) => return None,
            (c, b'=') if is_last => {
                let c = base64_value(c)?;
                output.push((b << 4) | (c >> 2));
            }
            (c, d) => {
                let c = base64_value(c)?;
                let d = base64_value(d)?;
                output.push((b << 4) | (c >> 2));
                output.push((c << 6) | d);
            }
        }
    }

    Some(output)
}

fn split_bpe_line(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let mut fields = line
        .split(|&byte| is_ascii_whitespace(byte))
        .filter(|field| !field.is_empty());
    let token = fields.next()?;
    let rank = fields.next()?;
    if fields.next().is_some() {
        return None;
    }
    Some((token, rank))
}

fn parse_rank(bytes: &[u8]) -> Option<Rank> {
    if bytes.is_empty() {
        return None;
    }

    let mut rank: u64 = 0;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        rank = rank.checked_mul(10)?.checked_add(u64::from(byte - b'0'))?;
        if rank > u64::from(Rank::MAX) {
            return None;
        }
    }
    Some(rank as Rank)
}

fn for_each_bpe_entry(
    contents: &[u8],
    source: &str,
    mut f: impl FnMut(Vec<u8>, Rank) -> PyResult<()>,
) -> PyResult<()> {
    for mut line in contents.split(|&byte| byte == b'\n') {
        if line.ends_with(b"\r") {
            line = &line[..line.len() - 1];
        }
        if line.is_empty() {
            continue;
        }

        let (token, rank) = split_bpe_line(line).ok_or_else(|| parse_bpe_error(line, source))?;
        let token = decode_base64(token).ok_or_else(|| parse_bpe_error(line, source))?;
        let rank = parse_rank(rank).ok_or_else(|| parse_bpe_error(line, source))?;
        f(token, rank)?;
    }

    Ok(())
}

#[pyfunction]
fn load_tiktoken_bpe(py: Python, contents: &[u8], source: &str) -> PyResult<Py<PyDict>> {
    let ret = PyDict::new(py);

    for_each_bpe_entry(contents, source, |token, rank| {
        ret.set_item(PyBytes::new(py, &token), rank)?;
        Ok(())
    })?;

    Ok(ret.into())
}

#[pyfunction]
fn load_tiktoken_bpe_core(
    contents: &[u8],
    source: &str,
    special_tokens_encoder: HashMap<String, Rank>,
    pattern: &str,
) -> PyResult<(CoreBPE, usize, Rank)> {
    let mut encoder = HashMap::default();
    let mut max_rank = 0;

    for_each_bpe_entry(contents, source, |token, rank| {
        max_rank = max_rank.max(rank);
        encoder.insert(token, rank);
        Ok(())
    })?;

    let n_mergeable_ranks = encoder.len();
    let core_bpe = CoreBPE::new_internal(encoder, special_tokens_encoder, pattern)
        .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((core_bpe, n_mergeable_ranks, max_rank))
}

fn decode_token_bytes(core_bpe: &CoreBPE, token: Rank) -> Result<&[u8], Rank> {
    if let Some(bytes) = core_bpe.decoder.get(&token) {
        return Ok(bytes);
    }
    if let Some(bytes) = core_bpe.special_tokens_decoder.get(&token) {
        return Ok(bytes);
    }
    Err(token)
}

fn decode_with_offsets(core_bpe: &CoreBPE, tokens: &[Rank]) -> Result<(Vec<u8>, Vec<usize>), Rank> {
    let mut text = Vec::with_capacity(tokens.len() * 4);
    let mut offsets = Vec::with_capacity(tokens.len());
    let mut text_len = 0usize;

    for &token in tokens {
        let token_bytes = decode_token_bytes(core_bpe, token)?;
        let starts_with_continuation = token_bytes
            .first()
            .is_some_and(|&byte| (0x80..0xC0).contains(&byte));
        offsets.push(text_len.saturating_sub(starts_with_continuation as usize));
        text_len += token_bytes
            .iter()
            .filter(|&&byte| !(0x80..0xC0).contains(&byte))
            .count();
        text.extend_from_slice(token_bytes);
    }

    Ok((text, offsets))
}

#[pymethods]
impl CoreBPE {
    #[new]
    fn py_new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> PyResult<Self> {
        Self::new_internal(encoder, special_tokens_encoder, pattern)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))
    }

    // ====================
    // Encoding
    // ====================

    #[pyo3(name = "encode_ordinary")]
    fn py_encode_ordinary(&self, py: Python, text: &str) -> Vec<Rank> {
        py.detach(|| self.encode_ordinary(text))
    }

    #[pyo3(name = "encode_ordinary_batch")]
    fn py_encode_ordinary_batch(&self, py: Python, text: Vec<PyBackedStr>) -> Vec<Vec<Rank>> {
        py.detach(|| {
            text.iter()
                .map(|text| self.encode_ordinary(text.as_ref()))
                .collect()
        })
    }

    #[pyo3(name = "encode")]
    fn py_encode(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Vec<Rank>> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            match self.encode(text, &allowed_special) {
                Ok((tokens, _)) => Ok(tokens),
                Err(e) => Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
            }
        })
    }

    #[pyo3(name = "encode_batch")]
    fn py_encode_batch(
        &self,
        py: Python,
        text: Vec<PyBackedStr>,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Vec<Vec<Rank>>> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            text.iter()
                .map(|text| match self.encode(text.as_ref(), &allowed_special) {
                    Ok((tokens, _)) => Ok(tokens),
                    Err(e) => Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
                })
                .collect()
        })
    }

    fn encode_to_tiktoken_buffer(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Py<PyAny>> {
        let tokens_res = py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode(text, &allowed_special)
        });

        let tokens = match tokens_res {
            Ok((tokens, _)) => tokens,
            Err(e) => return Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
        };

        let buffer = TiktokenBuffer { tokens };
        buffer.into_py_any(py)
    }

    fn _encode_bytes(&self, py: Python, bytes: &[u8]) -> Vec<Rank> {
        py.detach(|| {
            match std::str::from_utf8(bytes) {
                // Straightforward case
                Ok(text) => self.encode_ordinary(text),
                // Oops, don't actually have UTF-8. But we need to do the regex splitting in
                // Unicode space, so we make our best guess at where we would have splits
                Err(e) => {
                    let text = unsafe { std::str::from_utf8_unchecked(&bytes[..e.valid_up_to()]) };
                    let (tokens, last_piece_token_len) =
                        self.encode(text, &HashSet::new()).unwrap();
                    let (mut tokens, last_piece_token_len) =
                        self._increase_last_piece_token_len(tokens, last_piece_token_len);

                    let mut unstable_bytes;
                    if !tokens.is_empty() && last_piece_token_len > 0 {
                        // Lop off the tokens from the last piece and run BPE on the remaining bytes
                        // This likely matches what models see better, e.g. if you assume we're
                        // dealing with truncated UTF-8 bytes.
                        // Niche, but note this may not be correct if we'd have had a regex
                        // split between the valid UTF-8 and the invalid bytes.
                        unstable_bytes = self
                            .decode_bytes(&tokens[tokens.len() - last_piece_token_len..])
                            .unwrap();
                        unstable_bytes.extend_from_slice(&bytes[e.valid_up_to()..]);

                        tokens.truncate(tokens.len() - last_piece_token_len);
                    } else {
                        unstable_bytes = bytes[e.valid_up_to()..].to_vec();
                    }

                    if !unstable_bytes.is_empty() {
                        match self.encoder.get(&unstable_bytes) {
                            Some(token) => tokens.push(*token),
                            None => {
                                tokens.extend(&byte_pair_encode(&unstable_bytes, &self.encoder))
                            }
                        }
                    }
                    tokens
                }
            }
        })
    }

    #[pyo3(name = "encode_with_unstable")]
    fn py_encode_with_unstable(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<(Vec<Rank>, Py<PyList>)> {
        let (tokens, completions): (Vec<Rank>, HashSet<Vec<Rank>>) = py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self._encode_unstable_native(text, &allowed_special)
        });
        let py_completions = PyList::new(py, completions.into_iter())?;
        Ok((tokens, py_completions.into()))
    }

    fn encode_single_token(&self, piece: &[u8]) -> PyResult<Rank> {
        if let Some(token) = self.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(piece.to_owned()))
    }

    fn encode_single_piece(&self, piece: &[u8]) -> Vec<Rank> {
        if let Some(token) = self.encoder.get(piece) {
            return vec![*token];
        }
        byte_pair_encode(piece, &self.encoder)
    }

    // ====================
    // Decoding
    // ====================

    #[pyo3(name = "decode_bytes")]
    fn py_decode_bytes(&self, py: Python, tokens: Vec<Rank>) -> Result<Py<PyBytes>, PyErr> {
        match py.detach(|| self.decode_bytes(&tokens)) {
            Ok(bytes) => Ok(PyBytes::new(py, &bytes).into()),
            Err(e) => Err(pyo3::exceptions::PyKeyError::new_err(format!("{}", e))),
        }
    }

    #[pyo3(name = "decode_bytes_batch")]
    fn py_decode_bytes_batch(
        &self,
        py: Python,
        batch: Vec<Vec<Rank>>,
    ) -> Result<Vec<Py<PyBytes>>, PyErr> {
        match py.detach(|| {
            batch
                .iter()
                .map(|tokens| self.decode_bytes(tokens))
                .collect::<Result<Vec<_>, _>>()
        }) {
            Ok(bytes_batch) => Ok(bytes_batch
                .iter()
                .map(|bytes| PyBytes::new(py, bytes).into())
                .collect()),
            Err(e) => Err(pyo3::exceptions::PyKeyError::new_err(format!("{}", e))),
        }
    }

    #[pyo3(name = "decode_tokens_bytes")]
    fn py_decode_tokens_bytes(
        &self,
        py: Python,
        tokens: Vec<Rank>,
    ) -> Result<Vec<Py<PyBytes>>, PyErr> {
        tokens
            .iter()
            .map(|&token| match decode_token_bytes(self, token) {
                Ok(bytes) => Ok(PyBytes::new(py, bytes).into()),
                Err(token) => Err(pyo3::exceptions::PyKeyError::new_err(token.to_string())),
            })
            .collect()
    }

    #[pyo3(name = "decode_with_offsets")]
    fn py_decode_with_offsets(
        &self,
        py: Python,
        tokens: Vec<Rank>,
    ) -> Result<(Py<PyBytes>, Vec<usize>), PyErr> {
        match py.detach(|| decode_with_offsets(self, &tokens)) {
            Ok((text, offsets)) => Ok((PyBytes::new(py, &text).into(), offsets)),
            Err(token) => Err(pyo3::exceptions::PyKeyError::new_err(token.to_string())),
        }
    }

    fn decode_single_token_bytes(&self, py: Python, token: Rank) -> PyResult<Py<PyBytes>> {
        match decode_token_bytes(self, token) {
            Ok(bytes) => Ok(PyBytes::new(py, bytes).into()),
            Err(token) => Err(PyErr::new::<exceptions::PyKeyError, _>(token.to_string())),
        }
    }

    // ====================
    // Miscellaneous
    // ====================

    fn token_byte_values(&self, py: Python) -> Vec<Py<PyBytes>> {
        self.sorted_token_bytes()
            .iter()
            .map(|x| PyBytes::new(py, x).into())
            .collect()
    }

    fn mergeable_ranks(&self, py: Python) -> PyResult<Py<PyDict>> {
        let ret = PyDict::new(py);
        for (token, rank) in &self.encoder {
            ret.set_item(PyBytes::new(py, token), *rank)?;
        }
        Ok(ret.into())
    }
}

#[pyclass(frozen)]
struct TiktokenBuffer {
    tokens: Vec<Rank>,
}

#[pymethods]
impl TiktokenBuffer {
    // Based on https://github.com/PyO3/pyo3/blob/v0.22.2/tests/test_buffer_protocol.rs#L25
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Object is not writable",
            ));
        }
        unsafe {
            let view_ref = &mut *view;
            view_ref.obj = slf.clone().into_any().into_ptr();

            let data = &slf.borrow().tokens;
            view_ref.buf = data.as_ptr() as *mut std::os::raw::c_void;
            view_ref.len = (data.len() * std::mem::size_of::<Rank>()) as isize;
            view_ref.readonly = 1;
            view_ref.itemsize = std::mem::size_of::<Rank>() as isize;
            view_ref.format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
                let msg = std::ffi::CString::new("I").unwrap();
                msg.into_raw()
            } else {
                std::ptr::null_mut()
            };
            view_ref.ndim = 1;
            view_ref.shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
                &mut view_ref.len
            } else {
                std::ptr::null_mut()
            };
            view_ref.strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
                &mut view_ref.itemsize
            } else {
                std::ptr::null_mut()
            };
            view_ref.suboffsets = std::ptr::null_mut();
            view_ref.internal = std::ptr::null_mut();
        }

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // Note that Py_buffer doesn't have a Drop impl
        unsafe {
            let view_ref = &mut *view;
            if !view_ref.format.is_null() {
                std::mem::drop(std::ffi::CString::from_raw(view_ref.format));
            }
        }
    }
}

#[pymodule(gil_used = false)]
fn _tiktoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CoreBPE>()?;
    m.add_function(wrap_pyfunction!(load_tiktoken_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(load_tiktoken_bpe_core, m)?)?;
    Ok(())
}
