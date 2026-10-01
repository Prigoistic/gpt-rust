use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tokeniser_core::Tokeniser;

const DOC_BYTES: usize = 16 * 1024;

fn split_docs(text: &str, size: usize) -> Vec<&str> {
    let mut docs = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let mut cut = size.min(rest.len());
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        let (doc, tail) = rest.split_at(cut);
        docs.push(doc);
        rest = tail;
    }
    docs
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn encode(c: &mut Criterion) {
    let root = repo_root();
    let tokeniser = Tokeniser::from_gpt2_files(&root.join("data")).expect("load vocab");
    let mut group = c.benchmark_group("encode");
    group.sample_size(10);

    for name in ["varied", "repetitive"] {
        let path = root.join("bench-data").join(format!("{name}.txt"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipping {name}: run scripts/make_corpus.py first");
            continue;
        };
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function(name, |b| b.iter(|| tokeniser.encode(&text)));

        let docs = split_docs(&text, DOC_BYTES);
        group.bench_function(format!("{name}_batch"), |b| {
            b.iter(|| tokeniser.encode_batch(&docs))
        });
    }
    group.finish();
}

criterion_group!(benches, encode);
criterion_main!(benches);
