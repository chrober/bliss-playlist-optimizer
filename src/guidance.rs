//! Host for optional optimizer guidance addons.
//!
//! Addons are deliberately process-isolated. The optimizer sends versioned
//! JSONL requests to a trusted executable and treats timeouts, malformed
//! responses, and provider failures as neutral guidance. This keeps the core
//! Bliss route search independent of network clients, LMS APIs, and provider
//! implementation details.

use bliss_playlist_guidance_spi::{
    encode, Anchor, Candidate, GuidanceRequest, GuidanceResponse, GuidanceScope, GuidanceSignal,
    Manifest, ScoreContext, PROTOCOL_NAME, SPI_VERSION,
};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use super::GuidanceAddonConfig;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct AddonDiagnostic {
    pub configured_id: String,
    pub provider_id: Option<String>,
    pub state: &'static str,
    pub message: Option<String>,
}

struct Session {
    configured_id: String,
    options: Value,
    timeout: Duration,
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<Result<String, String>>,
    manifest: Option<Manifest>,
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
            timeout: config
                .timeout_ms
                .map(Duration::from_millis)
                .unwrap_or(DEFAULT_TIMEOUT),
            child,
            stdin,
            responses,
            manifest: None,
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

    fn prepare(
        &mut self,
        job_id: &str,
        candidates: Vec<Candidate>,
        anchors: Vec<Anchor>,
    ) -> Result<(), String> {
        let response = self.request(&GuidanceRequest::Prepare {
            spi_version: SPI_VERSION,
            job_id: job_id.to_owned(),
            options: self.options.clone(),
            candidates,
            anchors,
        })?;
        match response {
            GuidanceResponse::Prepared { .. } => Ok(()),
            GuidanceResponse::Error { message, .. } => Err(message),
            other => Err(format!(
                "guidance addon returned unexpected response {other:?}"
            )),
        }
    }

    fn score(
        &mut self,
        request_id: &str,
        context: ScoreContext,
        candidates: Vec<Candidate>,
    ) -> Result<Vec<GuidanceSignal>, String> {
        let response = self.request(&GuidanceRequest::Score {
            spi_version: SPI_VERSION,
            request_id: request_id.to_owned(),
            context,
            candidates,
        })?;
        match response {
            GuidanceResponse::Scores { signals, .. } => Ok(signals
                .into_iter()
                .filter(|signal| {
                    signal.scope == GuidanceScope::Global || signal.scope == GuidanceScope::Edge
                })
                .map(GuidanceSignal::bounded)
                .collect()),
            GuidanceResponse::Error { message, .. } => Err(message),
            other => Err(format!(
                "guidance addon returned unexpected response {other:?}"
            )),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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
                }),
            }
        }
        host
    }

    pub(crate) fn prepare(
        &mut self,
        job_id: &str,
        candidates: Vec<Candidate>,
        anchors: Vec<Anchor>,
    ) {
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
            match session.prepare(job_id, candidates.clone(), anchors.clone()) {
                Ok(()) => self.diagnostics.push(AddonDiagnostic {
                    configured_id: session.configured_id.clone(),
                    provider_id: Some(manifest.provider_id),
                    state: "prepared",
                    message: None,
                }),
                Err(message) => host_failure(&mut self.diagnostics, session, message),
            }
        }
    }

    pub(crate) fn score(
        &mut self,
        request_id: &str,
        context: ScoreContext,
        candidates: Vec<Candidate>,
    ) -> Vec<GuidanceSignal> {
        let mut signals = Vec::new();
        for session in &mut self.sessions {
            if session.disabled {
                continue;
            }
            match session.score(request_id, context.clone(), candidates.clone()) {
                Ok(mut values) => signals.append(&mut values),
                Err(message) => host_failure(&mut self.diagnostics, session, message),
            }
        }
        signals
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
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_host_is_neutral() {
        let mut host = GuidanceHost::start(&[]);
        let signals = host.score(
            "test",
            ScoreContext {
                scope: GuidanceScope::Global,
                left_anchor_id: None,
                right_anchor_id: None,
                context_track_ids: vec![],
            },
            vec![],
        );
        assert!(signals.is_empty());
        assert!(host.diagnostics.is_empty());
    }
}
