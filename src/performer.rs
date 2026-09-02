//! Runtime for the FBMX Performer: a score in, a way of playing it out.
//!
//! This is not an audio model and deliberately does not implement
//! [`crate::AudioModel`]. The audio runtimes in this crate take one sample and
//! return one sample; a Performer takes a *sequence of notes* and returns one
//! performance vector per note. Sharing the container is right — it is a header
//! and some named tensors, none of which is audio-specific — but sharing the
//! execution interface would mean pretending a note is a sample.
//!
//! It runs off the realtime thread by construction. A bidirectional pass cannot
//! start until the last note of the phrase is known, so there is nothing to be
//! gained by making it allocation-free in a callback; the host generates a
//! performance when the notes change and hands the engine the result.

use serde::Deserialize;

use crate::container::FbmxModel;
use crate::error::{FbmxError, Result};

/// Gates per step, in PyTorch's order: reset, update, new.
const GATES: usize = 3;

#[derive(Debug, Clone, Deserialize)]
struct PerformerHparams {
    hidden_size: usize,
    #[serde(default = "one")]
    num_layers: usize,
    #[serde(default)]
    bidirectional: bool,
    input_size: usize,
    output_size: usize,
    #[serde(default)]
    rnn: String,
}

fn one() -> usize {
    1
}

/// One direction's GRU weights.
#[derive(Debug, Clone)]
struct Direction {
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
}

/// A loaded Performer, ready to run over note sequences.
#[derive(Debug, Clone)]
pub struct PerformerRuntime {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    bidirectional: bool,
    forward: Direction,
    backward: Option<Direction>,
    /// One row of `trunk` weights plus a bias per output.
    head_weights: Vec<Vec<f32>>,
    head_biases: Vec<f32>,
    input_mean: Vec<f32>,
    input_std: Vec<f32>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Container model type of the Performer.
pub const PERFORMER_MODEL_TYPE: &str = "performer-gru";

/// Container model type of the Accent Analyzer.
///
/// A distinct type for a distinct model, sharing this execution path because
/// the architecture is genuinely the same — a bidirectional GRU over a note
/// sequence with one linear head per output. What differs is the feature
/// vocabulary and what the outputs mean, neither of which this file interprets.
/// Letting them share a *type* string would let a host load one where the other
/// belongs and read a prominence as an onset deviation.
pub const ACCENT_MODEL_TYPE: &str = "accent-gru";

impl PerformerRuntime {
    /// Load a Performer from a parsed container. Allocates.
    pub fn load(model: &FbmxModel) -> Result<Self> {
        Self::load_typed(model, PERFORMER_MODEL_TYPE)
    }

    /// Load a note-sequence GRU of a stated model type.
    pub fn load_typed(model: &FbmxModel, expected_type: &str) -> Result<Self> {
        let info = model.info();
        if info.model_type.as_str() != expected_type {
            return Err(FbmxError::UnsupportedModelType(
                info.model_type.as_str().to_string(),
            ));
        }
        let hp: PerformerHparams = serde_json::from_value(model.header().model.hparams.clone())?;
        if !hp.rnn.is_empty() && hp.rnn != "gru" {
            return Err(FbmxError::UnsupportedArchitecture(format!(
                "performer rnn = {:?}; only gru is implemented",
                hp.rnn
            )));
        }
        if hp.num_layers != 1 {
            return Err(FbmxError::UnsupportedArchitecture(format!(
                "performer num_layers = {}; only a single layer is implemented",
                hp.num_layers
            )));
        }
        if hp.hidden_size == 0 || hp.hidden_size > 1024 {
            return Err(FbmxError::UnsupportedArchitecture(format!(
                "hidden_size = {} is outside the supported range 1..=1024",
                hp.hidden_size
            )));
        }

        let gate = GATES * hp.hidden_size;
        let direction = |suffix: &str| -> Result<Direction> {
            Ok(Direction {
                weight_ih: model
                    .tensor(&format!("rnn.weight_ih_l0{suffix}"))?
                    .expect_shape(&[gate, hp.input_size])?
                    .to_vec(),
                weight_hh: model
                    .tensor(&format!("rnn.weight_hh_l0{suffix}"))?
                    .expect_shape(&[gate, hp.hidden_size])?
                    .to_vec(),
                bias_ih: model
                    .tensor(&format!("rnn.bias_ih_l0{suffix}"))?
                    .expect_shape(&[gate])?
                    .to_vec(),
                bias_hh: model
                    .tensor(&format!("rnn.bias_hh_l0{suffix}"))?
                    .expect_shape(&[gate])?
                    .to_vec(),
            })
        };

        let forward = direction("")?;
        let backward = if hp.bidirectional {
            Some(direction("_reverse")?)
        } else {
            None
        };

        let trunk = hp.hidden_size * if hp.bidirectional { 2 } else { 1 };
        let mut head_weights = Vec::with_capacity(hp.output_size);
        let mut head_biases = Vec::with_capacity(hp.output_size);
        for index in 0..hp.output_size {
            head_weights.push(
                model
                    .tensor(&format!("heads.{index}.weight"))?
                    .expect_shape(&[1, trunk])?
                    .to_vec(),
            );
            head_biases.push(
                model
                    .tensor(&format!("heads.{index}.bias"))?
                    .expect_shape(&[1])?[0],
            );
        }

        Ok(Self {
            input_size: hp.input_size,
            hidden_size: hp.hidden_size,
            output_size: hp.output_size,
            bidirectional: hp.bidirectional,
            forward,
            backward,
            head_weights,
            head_biases,
            input_mean: model
                .tensor("input_mean")?
                .expect_shape(&[hp.input_size])?
                .to_vec(),
            input_std: model
                .tensor("input_std")?
                .expect_shape(&[hp.input_size])?
                .to_vec(),
        })
    }

    pub const fn input_size(&self) -> usize {
        self.input_size
    }

    pub const fn output_size(&self) -> usize {
        self.output_size
    }

    pub const fn is_bidirectional(&self) -> bool {
        self.bidirectional
    }

    /// Run one GRU direction over the whole sequence.
    ///
    /// `out` is filled with `notes * hidden` values at `stride`-spaced offsets,
    /// so the forward and backward passes can write into the two halves of one
    /// interleaved trunk buffer without a second allocation.
    fn run_direction(
        &self,
        direction: &Direction,
        normalized: &[f32],
        notes: usize,
        out: &mut [f32],
        offset: usize,
        stride: usize,
        reverse: bool,
    ) {
        let hidden = self.hidden_size;
        let mut h = vec![0.0_f32; hidden];
        let mut gi = vec![0.0_f32; GATES * hidden];
        let mut gh = vec![0.0_f32; GATES * hidden];

        for step in 0..notes {
            let index = if reverse { notes - 1 - step } else { step };
            let x = &normalized[index * self.input_size..(index + 1) * self.input_size];

            for gate in 0..GATES * hidden {
                let row =
                    &direction.weight_ih[gate * self.input_size..(gate + 1) * self.input_size];
                let mut sum = direction.bias_ih[gate];
                for (weight, value) in row.iter().zip(x) {
                    sum += weight * value;
                }
                gi[gate] = sum;

                let row = &direction.weight_hh[gate * hidden..(gate + 1) * hidden];
                let mut sum = direction.bias_hh[gate];
                for (weight, value) in row.iter().zip(&h) {
                    sum += weight * value;
                }
                gh[gate] = sum;
            }

            for unit in 0..hidden {
                let r = sigmoid(gi[unit] + gh[unit]);
                let z = sigmoid(gi[hidden + unit] + gh[hidden + unit]);
                // The new gate scales the *hidden* contribution by the reset
                // gate, which is why the two biases cannot be folded together.
                let n = (gi[2 * hidden + unit] + r * gh[2 * hidden + unit]).tanh();
                h[unit] = (1.0 - z) * n + z * h[unit];
            }
            out[index * stride + offset..index * stride + offset + hidden].copy_from_slice(&h);
        }
    }

    /// Predict a performance for a sequence of notes.
    ///
    /// `notes` is `note_count * input_size`, row-major, in score order. The
    /// result is `note_count * output_size` in the same order.
    pub fn run(&self, notes: &[f32]) -> Result<Vec<f32>> {
        if self.input_size == 0 || !notes.len().is_multiple_of(self.input_size) {
            return Err(FbmxError::UnsupportedArchitecture(format!(
                "note buffer of {} values is not a multiple of the {} input features",
                notes.len(),
                self.input_size
            )));
        }
        let count = notes.len() / self.input_size;
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut normalized = vec![0.0_f32; notes.len()];
        for (index, value) in notes.iter().enumerate() {
            let feature = index % self.input_size;
            normalized[index] = (value - self.input_mean[feature]) / self.input_std[feature];
        }

        let trunk = self.hidden_size * if self.bidirectional { 2 } else { 1 };
        let mut encoded = vec![0.0_f32; count * trunk];
        self.run_direction(
            &self.forward,
            &normalized,
            count,
            &mut encoded,
            0,
            trunk,
            false,
        );
        if let Some(backward) = self.backward.as_ref() {
            self.run_direction(
                backward,
                &normalized,
                count,
                &mut encoded,
                self.hidden_size,
                trunk,
                true,
            );
        }

        let mut out = vec![0.0_f32; count * self.output_size];
        for note in 0..count {
            let row = &encoded[note * trunk..(note + 1) * trunk];
            for output in 0..self.output_size {
                let weights = &self.head_weights[output];
                let mut sum = self.head_biases[output];
                for (weight, value) in weights.iter().zip(row) {
                    sum += weight * value;
                }
                out[note * self.output_size + output] = sum;
            }
        }
        Ok(out)
    }
}

impl FbmxModel {
    /// Build a Performer from this container.
    ///
    /// Separate from [`FbmxModel::instantiate`] because a Performer is not an
    /// audio model: `instantiate` returns something that processes samples, and
    /// there is no sensible way for this to be that.
    pub fn instantiate_performer(&self) -> Result<PerformerRuntime> {
        PerformerRuntime::load(self)
    }

    /// Build an Accent Analyzer from this container.
    ///
    /// Same runtime, different model type: see [`ACCENT_MODEL_TYPE`].
    pub fn instantiate_accent_analyzer(&self) -> Result<PerformerRuntime> {
        PerformerRuntime::load_typed(self, ACCENT_MODEL_TYPE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single GRU step, worked out by hand from PyTorch's definition, so the
    /// gate order and the reset-scaling of the hidden contribution are pinned
    /// by something other than the implementation being tested.
    ///
    /// With all weights zero and all biases zero the gates are
    /// `r = z = sigmoid(0) = 0.5` and `n = tanh(0) = 0`, so
    /// `h' = (1 - 0.5) * 0 + 0.5 * 0 = 0`. Give the new gate an input bias of
    /// `a` and it becomes `h' = 0.5 * tanh(a)`.
    #[test]
    fn one_gru_step_matches_the_definition() {
        let hidden = 1;
        let input = 1;
        let mut runtime = PerformerRuntime {
            input_size: input,
            hidden_size: hidden,
            output_size: 1,
            bidirectional: false,
            forward: Direction {
                weight_ih: vec![0.0; GATES * hidden * input],
                weight_hh: vec![0.0; GATES * hidden * hidden],
                bias_ih: vec![0.0, 0.0, 0.75],
                bias_hh: vec![0.0; GATES * hidden],
            },
            backward: None,
            head_weights: vec![vec![1.0]],
            head_biases: vec![0.0],
            input_mean: vec![0.0],
            input_std: vec![1.0],
        };

        let out = runtime.run(&[0.0]).expect("one note");
        let expected = 0.5 * 0.75_f32.tanh();
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );

        // Second step: the hidden state now carries `expected`, and with zero
        // recurrent weights the update gate blends it against the same new
        // candidate: h'' = 0.5 * tanh(0.75) + 0.5 * h'.
        let out = runtime.run(&[0.0, 0.0]).expect("two notes");
        let second = 0.5 * 0.75_f32.tanh() + 0.5 * expected;
        assert!(
            (out[1] - second).abs() < 1e-6,
            "got {}, expected {second}",
            out[1]
        );

        // Normalisation is applied, not ignored.
        runtime.input_mean = vec![1.0];
        runtime.input_std = vec![2.0];
        let normalized = runtime.run(&[1.0]).expect("one note");
        assert!(
            (normalized[0] - expected).abs() < 1e-6,
            "an input equal to the mean must behave like a zero input"
        );
    }

    /// The backward direction has to read the sequence in reverse and land its
    /// state on the right note. A one-note sequence cannot tell the two apart,
    /// so this uses two and checks that reversing the input reverses which
    /// note carries the accumulated state.
    #[test]
    fn the_backward_direction_reads_the_sequence_in_reverse() {
        let hidden = 1;
        let input = 1;
        let direction = |bias: f32| Direction {
            weight_ih: vec![0.0, 0.0, 1.0],
            weight_hh: vec![0.0; GATES * hidden * hidden],
            bias_ih: vec![0.0, 0.0, bias],
            bias_hh: vec![0.0; GATES * hidden],
        };
        let runtime = PerformerRuntime {
            input_size: input,
            hidden_size: hidden,
            output_size: 1,
            bidirectional: true,
            forward: direction(0.0),
            backward: Some(direction(0.0)),
            // Read only the backward half of the trunk.
            head_weights: vec![vec![0.0, 1.0]],
            head_biases: vec![0.0],
            input_mean: vec![0.0],
            input_std: vec![1.0],
        };

        // Backward starts at the last note, so note 1 sees only itself while
        // note 0 sees note 1 first and then itself.
        let out = runtime.run(&[0.0, 2.0]).expect("two notes");
        let last = 0.5 * 2.0_f32.tanh();
        assert!(
            (out[1] - last).abs() < 1e-6,
            "the final note must be the backward pass's first step, got {}",
            out[1]
        );
        assert!(
            out[0].abs() > 1e-6,
            "the first note must carry state accumulated from the last"
        );
    }

    #[test]
    fn a_note_buffer_that_is_not_a_whole_number_of_notes_is_rejected() {
        let runtime = PerformerRuntime {
            input_size: 3,
            hidden_size: 1,
            output_size: 1,
            bidirectional: false,
            forward: Direction {
                weight_ih: vec![0.0; 3 * 3],
                weight_hh: vec![0.0; 3],
                bias_ih: vec![0.0; 3],
                bias_hh: vec![0.0; 3],
            },
            backward: None,
            head_weights: vec![vec![1.0]],
            head_biases: vec![0.0],
            input_mean: vec![0.0; 3],
            input_std: vec![1.0; 3],
        };
        assert!(runtime.run(&[0.0, 0.0]).is_err());
        assert!(runtime.run(&[]).expect("empty is fine").is_empty());
    }
}
