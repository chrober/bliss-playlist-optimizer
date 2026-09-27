//! Host for optional optimizer guidance addons.
//!
//! Addons are deliberately process-isolated. The optimizer sends versioned
//! JSONL requests to a trusted executable and treats timeouts, malformed
//! responses, and provider failures as neutral guidance. This keeps the core
//! Bliss route search independent of network clients, LMS APIs, and provider
//! implementation details.

use bliss_playlist_guidance_spi::{
    encode, Anchor, ArtifactDescriptor, Candidate, GuidanceRequest, GuidanceResponse,
    GuidanceScope, GuidanceSignal, Manifest, ResourceDescriptor, ScoreContext, PROTOCOL_NAME,
    SPI_VERSION,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use super::GuidanceAddonConfig;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(2);
#[allow(dead_code)] // Used by the Task 5 shared planner boundary.
const GUIDANCE_CAP: f64 = 0.75;

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct AddonDiagnostic {
    pub configured_id: String,
    pub provider_id: Option<String>,
    pub state: &'static str,
    pub message: Option<String>,
    pub prepared: bool,
    pub score_batches: u64,
    pub returned_signals: u64,
    pub accepted_signals: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ProviderPreparation {
    pub artifacts: Vec<ArtifactDescriptor>,
    pub resources: Vec<ResourceDescriptor>,
    pub anchors: Vec<Anchor>,
}

#[allow(dead_code)] // Constructed by the Task 5 shared planner boundary.
#[derive(Clone, Debug, Default)]
pub(crate) struct GuidanceWeights {
    by_provider_channel: BTreeMap<(String, String), f64>,
    target_by_provider_channel: BTreeMap<(String, String), u8>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TargetShareDiagnostic {
    pub provider_id: String,
    pub channel: String,
    pub target_percent: u8,
    pub supported_candidate_count: usize,
    pub multiplier: f64,
}

struct TargetShareDetails {
    provider_id: String,
    channel: String,
    target_percent: u8,
    supported: BTreeSet<usize>,
    multiplier: f64,
}

fn has_positive_support(
    items: Option<&Vec<AppliedGuidanceContribution>>,
    provider_id: &str,
    channel: &str,
) -> bool {
    items.is_some_and(|items| {
        items.iter().any(|item| {
            item.provider_id == provider_id && item.channel == channel && item.contribution > 0.0
        })
    })
}

impl GuidanceWeights {
    pub(crate) fn from_policy(policy: &[super::GuidancePolicyEntry]) -> Self {
        Self {
            by_provider_channel: policy
                .iter()
                .map(|entry| {
                    (
                        (entry.provider_id.clone(), entry.channel.clone()),
                        entry.weight.clamp(-1.0, 1.0),
                    )
                })
                .collect(),
            target_by_provider_channel: policy
                .iter()
                .filter_map(|entry| {
                    entry
                        .target_percent
                        .filter(|target| *target > 0)
                        .map(|target| ((entry.provider_id.clone(), entry.channel.clone()), target))
                })
                .collect(),
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.by_provider_channel
            .values()
            .any(|weight| *weight != 0.0)
    }

    pub(crate) fn has_target_shares(&self) -> bool {
        !self.target_by_provider_channel.is_empty()
    }

    /// Finds the smallest Bliss-ranked prefix that gives every configured
    /// target channel enough supported alternatives to plausibly meet its
    /// requested share. The provider can still be unavailable or have too few
    /// matches; in that case the complete supplied discovery pool is retained
    /// and the caller reports that limitation explicitly.
    pub(crate) fn target_share_minimum_pool_limit(
        &self,
        candidates: &[usize],
        contributions: &BTreeMap<usize, Vec<AppliedGuidanceContribution>>,
        requested_selection_count: usize,
        minimum_pool_limit: usize,
    ) -> usize {
        let minimum_pool_limit = minimum_pool_limit.min(candidates.len());
        if candidates.is_empty() || self.target_by_provider_channel.is_empty() {
            return minimum_pool_limit;
        }

        let required = self
            .target_by_provider_channel
            .iter()
            .map(|(channel, target_percent)| {
                let target_count = requested_selection_count
                    .saturating_mul(usize::from(*target_percent))
                    .div_ceil(100);
                (channel.clone(), target_count)
            })
            .collect::<BTreeMap<_, _>>();
        let mut supported = required
            .keys()
            .cloned()
            .map(|channel| (channel, 0_usize))
            .collect::<BTreeMap<_, _>>();

        for (index, candidate) in candidates.iter().copied().enumerate() {
            let items = contributions.get(&candidate);
            for channel in required.keys() {
                let has_support = has_positive_support(items, &channel.0, &channel.1);
                if has_support {
                    *supported.entry(channel.clone()).or_default() += 1;
                }
            }
            let prefix_length = index + 1;
            if prefix_length >= minimum_pool_limit
                && required.iter().all(|(channel, target)| {
                    supported.get(channel).copied().unwrap_or_default() >= *target
                })
            {
                return prefix_length;
            }
        }
        candidates.len()
    }

    fn weight(&self, provider_id: &str, channel: &str) -> f64 {
        self.by_provider_channel
            .get(&(provider_id.to_owned(), channel.to_owned()))
            .copied()
            .unwrap_or(0.0)
    }

    fn target_share_details(
        &self,
        candidates: &[usize],
        contributions: &BTreeMap<usize, Vec<AppliedGuidanceContribution>>,
    ) -> Vec<TargetShareDetails> {
        let pool_size = candidates.len();
        self.target_by_provider_channel
            .iter()
            .map(|((provider_id, channel), target_percent)| {
                let mut supported_base_weight = 0.0_f64;
                let mut other_base_weight = 0.0_f64;
                let mut supported = BTreeSet::new();
                for (rank, candidate) in candidates.iter().copied().enumerate() {
                    let rank_fraction = if pool_size > 1 {
                        rank as f64 / (pool_size - 1) as f64
                    } else {
                        0.0
                    };
                    // Identical to BlissMixer's candidate-selection base curve:
                    // the least acoustically suitable member of the pool retains
                    // one tenth of the top member's base weight.
                    let base_weight = (-std::f64::consts::LN_10 * rank_fraction).exp();
                    if has_positive_support(contributions.get(&candidate), provider_id, channel) {
                        supported.insert(candidate);
                        supported_base_weight += base_weight;
                    } else {
                        other_base_weight += base_weight;
                    }
                }
                let multiplier = if supported.is_empty() || other_base_weight <= 0.0 {
                    1.0
                } else {
                    let target = f64::from(*target_percent) / 100.0;
                    if target >= 1.0 {
                        1_000_000.0
                    } else {
                        ((target * other_base_weight) / ((1.0 - target) * supported_base_weight))
                            .max(0.000_001)
                    }
                };
                TargetShareDetails {
                    provider_id: provider_id.clone(),
                    channel: channel.clone(),
                    target_percent: *target_percent,
                    supported,
                    multiplier,
                }
            })
            .collect()
    }

    pub(crate) fn target_share_diagnostics(
        &self,
        candidates: &[usize],
        contributions: &BTreeMap<usize, Vec<AppliedGuidanceContribution>>,
    ) -> Vec<TargetShareDiagnostic> {
        self.target_share_details(candidates, contributions)
            .into_iter()
            .map(|details| TargetShareDiagnostic {
                provider_id: details.provider_id,
                channel: details.channel,
                target_percent: details.target_percent,
                supported_candidate_count: details.supported.len(),
                multiplier: details.multiplier,
            })
            .collect()
    }

    /// Returns DSTM-style per-candidate multipliers for channels configured
    /// as best-effort target shares.  The caller supplies an already
    /// Bliss-ranked, bounded candidate pool: a target may change selection
    /// inside this pool but can never nominate a candidate outside it.
    pub(crate) fn target_share_weights(
        &self,
        candidates: &[usize],
        contributions: &BTreeMap<usize, Vec<AppliedGuidanceContribution>>,
    ) -> BTreeMap<usize, f64> {
        let mut weights = candidates
            .iter()
            .copied()
            .map(|candidate| (candidate, 1.0))
            .collect::<BTreeMap<_, _>>();
        if candidates.is_empty() || self.target_by_provider_channel.is_empty() {
            return weights;
        }

        for details in self.target_share_details(candidates, contributions) {
            for candidate in details.supported {
                *weights.entry(candidate).or_insert(1.0) *= details.multiplier;
            }
        }
        weights
    }
}

#[derive(Clone, Debug)]
struct ProviderSignal {
    provider_id: String,
    signal: GuidanceSignal,
}

/// A provider signal that survived policy weighting and changed an already
/// Bliss-qualified candidate's score. This is emitted with selected results so
/// hosts can explain an actual selection without recreating provider logic.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct AppliedGuidanceContribution {
    pub provider_id: String,
    pub channel: String,
    pub scope: GuidanceScope,
    pub signal_score: f64,
    pub confidence: f64,
    pub policy_weight: f64,
    pub contribution: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

#[allow(dead_code)] // Consumed by the Task 5 shared planner boundary.
#[derive(Clone, Debug, Default)]
pub(crate) struct GuidanceBatch {
    pub adjustment_by_candidate: BTreeMap<usize, f64>,
    pub contributions_by_candidate: BTreeMap<usize, Vec<AppliedGuidanceContribution>>,
    pub observed: u64,
    pub applied: u64,
    pub diagnostics: Vec<AddonDiagnostic>,
}

struct Session {
    configured_id: String,
    options: Value,
    timeout: Duration,
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<Result<String, String>>,
    manifest: Option<Manifest>,
    preparation: ProviderPreparation,
    prepared: bool,
    score_batches: u64,
    returned_signals: u64,
    accepted_signals: u64,
    disabled: bool,
}

impl Session {
    fn start(config: &GuidanceAddonConfig) -> Result<Self, String> {
        let mut child = Command::new(&config.program)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("cannot start '{}': {error}", config.program))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "guidance addon stdin was not piped".to_owned())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "guidance addon stdout was not piped".to_owned())?;
        let (sender, responses) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if sender.send(Ok(line.trim_end().to_owned())).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        });
        Ok(Self {
            configured_id: config.id.clone(),
            options: config.options.clone(),
            preparation: ProviderPreparation {
                artifacts: config.artifacts.clone(),
                resources: config.resources.clone(),
                anchors: Vec::new(),
            },
            timeout: config
                .timeout_ms
                .map(Duration::from_millis)
                .unwrap_or(DEFAULT_TIMEOUT),
            child,
            stdin,
            responses,
            manifest: None,
            prepared: false,
            score_batches: 0,
            returned_signals: 0,
            accepted_signals: 0,
            disabled: false,
        })
    }

    fn request(&mut self, request: &GuidanceRequest) -> Result<GuidanceResponse, String> {
        if self.disabled {
            return Err("guidance addon session is disabled after an earlier failure".to_owned());
        }
        let line =
            encode(request).map_err(|error| format!("cannot encode guidance request: {error}"))?;
        if let Err(error) = writeln!(self.stdin, "{line}").and_then(|_| self.stdin.flush()) {
            self.disable();
            return Err(format!("cannot write guidance request: {error}"));
        }
        let response = match self.responses.recv_timeout(self.timeout) {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                self.disable();
                return Err(error);
            }
            Err(error) => {
                self.disable();
                return Err(format!("guidance addon response timeout: {error}"));
            }
        };
        serde_json::from_str(&response).map_err(|error| {
            self.disable();
            format!("cannot decode guidance addon response: {error}")
        })
    }

    fn disable(&mut self) {
        self.disabled = true;
        let _ = self.child.kill();
    }

    fn close(&mut self) {
        if !self.disabled {
            let _ = self.request(&GuidanceRequest::Close {
                spi_version: SPI_VERSION,
            });
        }
        self.disabled = true;
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }

    fn describe(&mut self) -> Result<Manifest, String> {
        let response = self.request(&GuidanceRequest::Describe {
            spi_version: SPI_VERSION,
        })?;
        let GuidanceResponse::Manifest(manifest) = response else {
            return Err("guidance addon did not return a manifest".to_owned());
        };
        if manifest.spi_version != SPI_VERSION {
            return Err(format!(
                "guidance addon reports unsupported SPI version {}",
                manifest.spi_version
            ));
        }
        if manifest.protocol != PROTOCOL_NAME {
            return Err(format!(
                "guidance addon reports unsupported protocol '{}'",
                manifest.protocol
            ));
        }
        self.manifest = Some(manifest.clone());
        Ok(manifest)
    }

    fn prepare(&mut self, job_id: &str, anchors: Vec<Anchor>) -> Result<(), String> {
        let mut preparation = self.preparation.clone();
        preparation.anchors = anchors;
        let response = self.request(&GuidanceRequest::Prepare {
            spi_version: SPI_VERSION,
            job_id: job_id.to_owned(),
            options: self.options.clone(),
            artifacts: preparation.artifacts,
            resources: preparation.resources,
            anchors: preparation.anchors,
        })?;
        match response {
            GuidanceResponse::Prepared { .. } => {
                self.prepared = true;
                Ok(())
            }
            GuidanceResponse::Error { message, .. } => Err(message),
            other => Err(format!(
                "guidance addon returned unexpected response {other:?}"
            )),
        }
    }

    #[allow(dead_code)] // Activated by the shared planner boundary in Task 5.
    fn score(
        &mut self,
        request_id: &str,
        context: ScoreContext,
        candidates: Vec<Candidate>,
    ) -> Result<Vec<ProviderSignal>, String> {
        let candidate_ids = candidates
            .iter()
            .map(|candidate| candidate.candidate_id.clone())
            .collect::<BTreeSet<_>>();
        let response = self.request(&GuidanceRequest::Score {
            spi_version: SPI_VERSION,
            request_id: request_id.to_owned(),
            context,
            candidates,
        })?;
        match response {
            GuidanceResponse::Scores { signals, .. } => {
                let accepted = validate_batch_signals(&candidate_ids, signals)?;
                let manifest = self.manifest.as_ref().ok_or_else(|| {
                    "guidance addon was scored before manifest negotiation".to_owned()
                })?;
                validate_manifest_signals(manifest, &accepted.1)?;
                self.score_batches += 1;
                self.returned_signals += accepted.0;
                self.accepted_signals += accepted.1.len() as u64;
                Ok(accepted
                    .1
                    .into_iter()
                    .map(|signal| ProviderSignal {
                        provider_id: manifest.provider_id.clone(),
                        signal,
                    })
                    .collect())
            }
            GuidanceResponse::Error { message, .. } => Err(message),
            other => Err(format!(
                "guidance addon returned unexpected response {other:?}"
            )),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.close();
    }
}

/// A best-effort collection of optional guidance addon sessions.
pub(crate) struct GuidanceHost {
    sessions: Vec<Session>,
    pub diagnostics: Vec<AddonDiagnostic>,
}

impl GuidanceHost {
    pub(crate) fn start(configs: &[GuidanceAddonConfig]) -> Self {
        let mut host = Self {
            sessions: Vec::new(),
            diagnostics: Vec::new(),
        };
        for config in configs {
            match Session::start(config) {
                Ok(session) => host.sessions.push(session),
                Err(message) => host.diagnostics.push(AddonDiagnostic {
                    configured_id: config.id.clone(),
                    provider_id: None,
                    state: "failed",
                    message: Some(message),
                    prepared: false,
                    score_batches: 0,
                    returned_signals: 0,
                    accepted_signals: 0,
                }),
            }
        }
        host
    }

    pub(crate) fn prepare(&mut self, job_id: &str, anchors: Vec<Anchor>) {
        for session in &mut self.sessions {
            if session.disabled {
                continue;
            }
            let manifest = match session.describe() {
                Ok(manifest) => manifest,
                Err(message) => {
                    session.disable();
                    host_failure(&mut self.diagnostics, session, message);
                    continue;
                }
            };
            if manifest.provider_id != session.configured_id {
                session.disable();
                host_failure(
                    &mut self.diagnostics,
                    session,
                    format!(
                        "manifest provider id '{}' does not match configured id '{}'",
                        manifest.provider_id, session.configured_id
                    ),
                );
                continue;
            }
            match session.prepare(job_id, anchors.clone()) {
                Ok(()) => self.diagnostics.push(AddonDiagnostic {
                    configured_id: session.configured_id.clone(),
                    provider_id: Some(manifest.provider_id),
                    state: "prepared",
                    message: None,
                    prepared: session.prepared,
                    score_batches: session.score_batches,
                    returned_signals: session.returned_signals,
                    accepted_signals: session.accepted_signals,
                }),
                Err(message) => {
                    session.disable();
                    host_failure(&mut self.diagnostics, session, message);
                }
            }
        }
    }

    pub(crate) fn accepted_signal_count(&self) -> usize {
        self.diagnostics
            .iter()
            .map(|diagnostic| diagnostic.accepted_signals as usize)
            .sum()
    }

    #[allow(dead_code)] // Activated by the shared planner boundary in Task 5.
    pub(crate) fn score(
        &mut self,
        request_id: &str,
        context: ScoreContext,
        candidates: Vec<Candidate>,
        weights: &GuidanceWeights,
        candidate_index: &BTreeMap<String, usize>,
    ) -> GuidanceBatch {
        let mut signals = Vec::new();
        for session in &mut self.sessions {
            if session.disabled {
                continue;
            }
            match session.score(request_id, context.clone(), candidates.clone()) {
                Ok(mut values) => {
                    refresh_prepared_diagnostic(
                        &mut self.diagnostics,
                        &session.configured_id,
                        session.score_batches,
                        session.returned_signals,
                        session.accepted_signals,
                    );
                    signals.append(&mut values);
                }
                Err(message) => {
                    session.disable();
                    host_failure(&mut self.diagnostics, session, message);
                }
            }
        }
        let mut batch = aggregate_batch(signals, weights, candidate_index);
        batch.diagnostics = self.diagnostics.clone();
        batch
    }
}

fn refresh_prepared_diagnostic(
    diagnostics: &mut [AddonDiagnostic],
    configured_id: &str,
    score_batches: u64,
    returned_signals: u64,
    accepted_signals: u64,
) {
    if let Some(diagnostic) = diagnostics.iter_mut().rev().find(|diagnostic| {
        diagnostic.configured_id == configured_id && diagnostic.state == "prepared"
    }) {
        diagnostic.score_batches = score_batches;
        diagnostic.returned_signals = returned_signals;
        diagnostic.accepted_signals = accepted_signals;
    }
}

fn host_failure(diagnostics: &mut Vec<AddonDiagnostic>, session: &Session, message: String) {
    diagnostics.push(AddonDiagnostic {
        configured_id: session.configured_id.clone(),
        provider_id: session
            .manifest
            .as_ref()
            .map(|manifest| manifest.provider_id.clone()),
        state: "failed",
        message: Some(message),
        prepared: session.prepared,
        score_batches: session.score_batches,
        returned_signals: session.returned_signals,
        accepted_signals: session.accepted_signals,
    });
}

#[allow(dead_code)] // Called by GuidanceHost::score once a planner requests guidance.
fn validate_batch_signals(
    candidate_ids: &BTreeSet<String>,
    signals: Vec<GuidanceSignal>,
) -> Result<(u64, Vec<GuidanceSignal>), String> {
    let returned = signals.len() as u64;
    if let Some(signal) = signals
        .iter()
        .find(|signal| !candidate_ids.contains(&signal.candidate_id))
    {
        return Err(format!(
            "guidance addon returned candidate '{}' outside the score batch",
            signal.candidate_id
        ));
    }
    let mut accepted = signals
        .into_iter()
        .filter(|signal| {
            signal.scope == GuidanceScope::Global || signal.scope == GuidanceScope::Edge
        })
        .map(GuidanceSignal::bounded)
        .collect::<Vec<_>>();
    accepted.sort_by(|left, right| {
        left.candidate_id
            .cmp(&right.candidate_id)
            .then_with(|| left.channel.cmp(&right.channel))
    });
    Ok((returned, accepted))
}

fn validate_manifest_signals(
    manifest: &Manifest,
    signals: &[GuidanceSignal],
) -> Result<(), String> {
    for signal in signals {
        let Some(channel) = manifest
            .channels
            .iter()
            .find(|channel| channel.channel == signal.channel)
        else {
            return Err(format!(
                "guidance addon returned undeclared channel '{}'",
                signal.channel
            ));
        };
        if !channel.scopes.contains(&signal.scope) {
            return Err(format!(
                "guidance addon returned channel '{}' for undeclared scope {:?}",
                signal.channel, signal.scope
            ));
        }
    }
    Ok(())
}

#[allow(dead_code)] // Called by GuidanceHost::score at the planner boundary.
fn aggregate_batch(
    mut signals: Vec<ProviderSignal>,
    weights: &GuidanceWeights,
    candidate_index: &BTreeMap<String, usize>,
) -> GuidanceBatch {
    signals.sort_by(|left, right| {
        left.provider_id
            .cmp(&right.provider_id)
            .then_with(|| left.signal.candidate_id.cmp(&right.signal.candidate_id))
            .then_with(|| left.signal.channel.cmp(&right.signal.channel))
    });
    let observed = signals.len() as u64;
    let mut adjustment_by_candidate = BTreeMap::<usize, f64>::new();
    let mut applied = 0_u64;
    let mut contributions_by_candidate = BTreeMap::<usize, Vec<AppliedGuidanceContribution>>::new();
    for provider_signal in signals {
        let ProviderSignal {
            provider_id,
            signal,
        } = provider_signal;
        let Some(candidate) = candidate_index.get(&signal.candidate_id).copied() else {
            continue;
        };
        let policy_weight = weights.weight(&provider_id, &signal.channel);
        let contribution = policy_weight * signal.score * signal.confidence;
        if contribution != 0.0 {
            applied += 1;
            *adjustment_by_candidate.entry(candidate).or_default() += contribution;
            contributions_by_candidate
                .entry(candidate)
                .or_default()
                .push(AppliedGuidanceContribution {
                    provider_id,
                    channel: signal.channel,
                    scope: signal.scope,
                    signal_score: signal.score,
                    confidence: signal.confidence,
                    policy_weight,
                    contribution,
                    rationale: signal.rationale,
                });
        }
    }
    for adjustment in adjustment_by_candidate.values_mut() {
        *adjustment = adjustment.clamp(-GUIDANCE_CAP, GUIDANCE_CAP);
    }
    GuidanceBatch {
        adjustment_by_candidate,
        contributions_by_candidate,
        observed,
        applied,
        diagnostics: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_host_is_neutral() {
        let mut host = GuidanceHost::start(&[]);
        let batch = host.score(
            "test",
            ScoreContext {
                scope: GuidanceScope::Global,
                left_anchor_id: None,
                right_anchor_id: None,
                context_track_ids: vec![],
            },
            vec![],
            &GuidanceWeights::default(),
            &BTreeMap::new(),
        );
        assert!(batch.adjustment_by_candidate.is_empty());
        assert!(host.diagnostics.is_empty());
    }

    #[test]
    fn out_of_batch_signal_is_rejected_as_a_provider_failure() {
        let candidates = BTreeSet::from(["shortlist-a".to_owned(), "shortlist-b".to_owned()]);
        let error = validate_batch_signals(
            &candidates,
            vec![GuidanceSignal {
                candidate_id: "not-in-shortlist".to_owned(),
                channel: "playcount".to_owned(),
                scope: GuidanceScope::Global,
                score: 1.0,
                confidence: 1.0,
                rationale: None,
                observed_at: None,
            }],
        )
        .unwrap_err();
        assert!(error.contains("outside the score batch"));
    }

    #[test]
    fn aggregation_combines_weighted_channels_with_a_stable_cap() {
        let signals = vec![
            ProviderSignal {
                provider_id: "lastfm-guidance".to_owned(),
                signal: GuidanceSignal {
                    candidate_id: "bliss-row-2".to_owned(),
                    channel: "lastfm_artist".to_owned(),
                    scope: GuidanceScope::Global,
                    score: 0.5,
                    confidence: 1.0,
                    rationale: None,
                    observed_at: None,
                },
            },
            ProviderSignal {
                provider_id: "lastfm-guidance".to_owned(),
                signal: GuidanceSignal {
                    candidate_id: "bliss-row-2".to_owned(),
                    channel: "lastfm_track".to_owned(),
                    scope: GuidanceScope::Global,
                    score: 1.0,
                    confidence: 0.5,
                    rationale: None,
                    observed_at: None,
                },
            },
        ];
        let weights = GuidanceWeights::from_policy(&[
            crate::GuidancePolicyEntry {
                provider_id: "lastfm-guidance".to_owned(),
                channel: "lastfm_track".to_owned(),
                weight: 0.8,
                target_percent: None,
            },
            crate::GuidancePolicyEntry {
                provider_id: "lastfm-guidance".to_owned(),
                channel: "lastfm_artist".to_owned(),
                weight: 0.4,
                target_percent: None,
            },
        ]);
        let index = BTreeMap::from([("bliss-row-2".to_owned(), 2_usize)]);
        let batch = aggregate_batch(signals, &weights, &index);
        assert_eq!(batch.observed, 2);
        assert_eq!(batch.applied, 2);
        assert!((batch.adjustment_by_candidate[&2] - 0.6).abs() < f64::EPSILON);
        let contributions = batch
            .contributions_by_candidate
            .get(&2)
            .expect("applied guidance is retained for result provenance");
        assert_eq!(contributions.len(), 2);
        assert_eq!(contributions[0].provider_id, "lastfm-guidance");
        assert_eq!(contributions[0].channel, "lastfm_artist");
        assert!((contributions[0].contribution - 0.2).abs() < f64::EPSILON);
        assert_eq!(contributions[1].channel, "lastfm_track");
        assert!((contributions[1].contribution - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn target_share_weights_promote_supported_candidates_from_the_same_bliss_pool() {
        let weights = GuidanceWeights::from_policy(&[crate::GuidancePolicyEntry {
            provider_id: "lastfm-guidance".to_owned(),
            channel: "lastfm_track".to_owned(),
            weight: 1.0,
            target_percent: Some(75),
        }]);
        let contributions = BTreeMap::from([(
            2_usize,
            vec![AppliedGuidanceContribution {
                provider_id: "lastfm-guidance".to_owned(),
                channel: "lastfm_track".to_owned(),
                scope: GuidanceScope::Global,
                signal_score: 1.0,
                confidence: 1.0,
                policy_weight: 1.0,
                contribution: 1.0,
                rationale: None,
            }],
        )]);

        let candidate_weights = weights.target_share_weights(&[1, 2, 3, 4], &contributions);

        assert!(candidate_weights[&2] > 1.0);
        assert_eq!(candidate_weights[&1], 1.0);
        assert_eq!(candidate_weights[&3], 1.0);
        assert_eq!(candidate_weights[&4], 1.0);
    }

    #[test]
    fn zero_target_does_not_change_the_bliss_order_weight() {
        let weights = GuidanceWeights::from_policy(&[crate::GuidancePolicyEntry {
            provider_id: "lastfm-guidance".to_owned(),
            channel: "lastfm_track".to_owned(),
            weight: 1.0,
            target_percent: Some(0),
        }]);
        let contributions = BTreeMap::from([(
            2_usize,
            vec![AppliedGuidanceContribution {
                provider_id: "lastfm-guidance".to_owned(),
                channel: "lastfm_track".to_owned(),
                scope: GuidanceScope::Global,
                signal_score: 1.0,
                confidence: 1.0,
                policy_weight: 1.0,
                contribution: 1.0,
                rationale: None,
            }],
        )]);

        assert_eq!(
            weights.target_share_weights(&[1, 2], &contributions),
            BTreeMap::from([(1, 1.0), (2, 1.0)])
        );
    }

    #[test]
    fn target_share_diagnostics_explain_supported_candidates_and_multiplier() {
        let weights = GuidanceWeights::from_policy(&[crate::GuidancePolicyEntry {
            provider_id: "lastfm-guidance".to_owned(),
            channel: "lastfm_artist".to_owned(),
            weight: 1.0,
            target_percent: Some(75),
        }]);
        let contributions = BTreeMap::from([(
            2_usize,
            vec![AppliedGuidanceContribution {
                provider_id: "lastfm-guidance".to_owned(),
                channel: "lastfm_artist".to_owned(),
                scope: GuidanceScope::Global,
                signal_score: 1.0,
                confidence: 1.0,
                policy_weight: 1.0,
                contribution: 1.0,
                rationale: None,
            }],
        )]);

        let diagnostics = weights.target_share_diagnostics(&[1, 2, 3], &contributions);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].provider_id, "lastfm-guidance");
        assert_eq!(diagnostics[0].channel, "lastfm_artist");
        assert_eq!(diagnostics[0].target_percent, 75);
        assert_eq!(diagnostics[0].supported_candidate_count, 1);
        assert!(diagnostics[0].multiplier > 1.0);
    }

    #[test]
    fn aggregation_uses_provider_and_channel_as_the_policy_key() {
        let signals = vec![ProviderSignal {
            provider_id: "future-provider".to_owned(),
            signal: GuidanceSignal {
                candidate_id: "bliss-row-2".to_owned(),
                channel: "preference".to_owned(),
                scope: GuidanceScope::Global,
                score: 1.0,
                confidence: 1.0,
                rationale: None,
                observed_at: None,
            },
        }];
        let weights = GuidanceWeights::from_policy(&[crate::GuidancePolicyEntry {
            provider_id: "future-provider".to_owned(),
            channel: "preference".to_owned(),
            weight: 0.5,
            target_percent: None,
        }]);
        let index = BTreeMap::from([("bliss-row-2".to_owned(), 2_usize)]);

        let batch = aggregate_batch(signals, &weights, &index);

        assert!((batch.adjustment_by_candidate[&2] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn refreshes_prepared_diagnostic_with_final_score_counts() {
        let mut diagnostics = vec![AddonDiagnostic {
            configured_id: "lastfm-guidance".to_owned(),
            provider_id: Some("lastfm-guidance".to_owned()),
            state: "prepared",
            message: None,
            prepared: true,
            score_batches: 0,
            returned_signals: 0,
            accepted_signals: 0,
        }];

        refresh_prepared_diagnostic(&mut diagnostics, "lastfm-guidance", 3, 12, 10);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].score_batches, 3);
        assert_eq!(diagnostics[0].returned_signals, 12);
        assert_eq!(diagnostics[0].accepted_signals, 10);
    }
}
