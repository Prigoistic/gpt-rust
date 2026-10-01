use std::path::PathBuf;

use pyo3::prelude::*;
use tokeniser_core::Tokeniser as CoreTokeniser;

/// Python-facing wrapper around `tokeniser_core::Tokeniser`. Holds no logic of its own;
/// it exists so the Rust tokeniser can be tested and benchmarked against tiktoken.
#[pyclass]
struct Tokeniser {
    inner: CoreTokeniser,
}

#[pymethods]
impl Tokeniser {
    #[staticmethod]
    fn from_gpt2_files(data_dir: String) -> Self {
        Tokeniser {
            inner: CoreTokeniser::from_gpt2_files(&PathBuf::from(data_dir)),
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }
}

#[pymodule]
fn tokeniser_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokeniser>()?;
    Ok(())
}
