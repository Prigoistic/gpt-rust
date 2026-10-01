use std::path::Path;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use tokeniser_core::Tokeniser as CoreTokeniser;

#[pyclass]
struct Tokeniser {
    inner: CoreTokeniser,
}

#[pymethods]
impl Tokeniser {
    #[staticmethod]
    fn from_gpt2_files(data_dir: &str) -> PyResult<Self> {
        let inner = CoreTokeniser::from_gpt2_files(Path::new(data_dir))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymodule]
fn tokeniser_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokeniser>()?;
    Ok(())
}
