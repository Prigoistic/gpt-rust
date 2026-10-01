use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tokeniser_core::Tokeniser;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn encode(c: &mut Criterion) {
    let root = repo_root();
    let tokeniser = Tokeniser::from_gpt2_files(&root.join("data"));
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
    }
    group.finish();
}

criterion_group!(benches, encode);
criterion_main!(benches);
