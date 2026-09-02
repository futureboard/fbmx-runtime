//! Run a Performer over a deterministic note sequence and print the result.
//!
//! ```text
//! cargo run -p fbmx-runtime --release --bin fbmx-performer-probe -- <model.fbmx> [notes] [seed]
//! ```
//!
//! This exists so Python and Rust can be compared on the same numbers. The
//! input sequence is generated from a seed with a generator simple enough to
//! reimplement exactly in NumPy, because a parity test whose two sides disagree
//! about the *input* measures nothing.
//!
//! Output is one line per note: whitespace-separated `f32` values, printed with
//! enough digits to round-trip.

use std::time::Instant;

use fbmx_runtime::FbmxModel;

/// A 64-bit LCG. Chosen for being trivially reproducible in any language, not
/// for statistical quality — the sequence only has to be identical on both
/// sides, and this one is fully specified by two constants.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes constants.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[-1, 1)` with 24 bits of mantissa.
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // top 24 bits
        (bits as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: fbmx-performer-probe <model.fbmx> [notes] [seed]");
        std::process::exit(2);
    };
    let notes: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(128);
    let seed: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(7);

    let model = match FbmxModel::load(&path) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("load failed: {error}");
            std::process::exit(1);
        }
    };
    let performer = match model.instantiate_performer() {
        Ok(performer) => performer,
        Err(error) => {
            eprintln!("instantiate failed: {error}");
            std::process::exit(1);
        }
    };

    let mut rng = Lcg(seed);
    let mut input = vec![0.0_f32; notes * performer.input_size()];
    for value in input.iter_mut() {
        *value = rng.next_f32();
    }

    let started = Instant::now();
    let output = match performer.run(&input) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("run failed: {error}");
            std::process::exit(1);
        }
    };
    let elapsed = started.elapsed();

    eprintln!(
        "notes={notes} input_size={} output_size={} bidirectional={} elapsed_us={:.1}",
        performer.input_size(),
        performer.output_size(),
        performer.is_bidirectional(),
        elapsed.as_secs_f64() * 1e6
    );
    let width = performer.output_size();
    for note in 0..notes {
        let row = &output[note * width..(note + 1) * width];
        let text: Vec<String> = row.iter().map(|v| format!("{v:.9}")).collect();
        println!("{}", text.join(" "));
    }
}
