//! TLS configuration for secure HTTPS transport
//!
//! Provides enterprise-grade TLS configuration using rustls.
//! Supports certificate and key loading from PEM files.

use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::ServerConfig;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use tokio_rustls::TlsAcceptor;

/// TLS configuration error
#[derive(Debug, Error)]
pub enum TlsError {
    #[error("Failed to read certificate file '{path}': {source}")]
    CertificateRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to read key file '{path}': {source}")]
    KeyRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("No certificates found in '{path}'")]
    NoCertificates { path: String },

    #[error("No private key found in '{path}'")]
    NoPrivateKey { path: String },

    #[error("Failed to build TLS config: {0}")]
    ConfigBuild(String),
}

/// TLS configuration for HTTPS server
///
/// # Example
/// ```ignore
/// let tls_config = TlsConfig::from_pem_files(
///     "/path/to/cert.pem",
///     "/path/to/key.pem",
/// )?;
/// let transport = HyperTransport::with_tls("0.0.0.0:6334", tls_config);
/// ```
/// TLS configuration wrapper
#[derive(Clone)]
pub struct TlsConfig {
    acceptor: TlsAcceptor,
}

impl TlsConfig {
    /// Create TLS config from PEM certificate and key files
    ///
    /// # Arguments
    /// * `cert_path` - Path to PEM-encoded certificate chain file
    /// * `key_path` - Path to PEM-encoded private key file
    ///
    /// # Errors
    /// Returns error if files cannot be read or parsed
    pub fn from_pem_files(
        cert_path: impl AsRef<Path>,
        key_path: impl AsRef<Path>,
    ) -> Result<Self, TlsError> {
        let cert_path = cert_path.as_ref();
        let key_path = key_path.as_ref();

        // Load certificates
        let certs = load_certs(cert_path)?;
        if certs.is_empty() {
            return Err(TlsError::NoCertificates {
                path: cert_path.display().to_string(),
            });
        }

        // Load private key
        let key = load_key(key_path)?;

        // Build server config with safe defaults
        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| TlsError::ConfigBuild(e.to_string()))?;

        Ok(Self {
            acceptor: TlsAcceptor::from(Arc::new(config)),
        })
    }

    /// Get the TLS acceptor for accepting connections
    pub fn acceptor(&self) -> &TlsAcceptor {
        &self.acceptor
    }
}

/// Load certificates from PEM file
fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, TlsError> {
    let file = File::open(path).map_err(|e| TlsError::CertificateRead {
        path: path.display().to_string(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);

    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut reader)
        .filter_map(|result| result.ok())
        .collect();

    Ok(certs)
}

/// Load private key from PEM file
fn load_key(path: &Path) -> Result<PrivateKeyDer<'static>, TlsError> {
    let file = File::open(path).map_err(|e| TlsError::KeyRead {
        path: path.display().to_string(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);

    // Try to read any supported key format (PKCS#8, RSA, EC)
    loop {
        match rustls_pemfile::read_one(&mut reader) {
            Ok(Some(rustls_pemfile::Item::Pkcs1Key(key))) => {
                return Ok(PrivateKeyDer::Pkcs1(key));
            }
            Ok(Some(rustls_pemfile::Item::Pkcs8Key(key))) => {
                return Ok(PrivateKeyDer::Pkcs8(key));
            }
            Ok(Some(rustls_pemfile::Item::Sec1Key(key))) => {
                return Ok(PrivateKeyDer::Sec1(key));
            }
            Ok(Some(_)) => continue, // Skip other items
            Ok(None) => break,       // End of file
            Err(_) => break,
        }
    }

    Err(TlsError::NoPrivateKey {
        path: path.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_error_display() {
        let err = TlsError::NoCertificates {
            path: "/test/cert.pem".to_string(),
        };
        assert!(err.to_string().contains("/test/cert.pem"));
    }

    #[test]
    fn test_missing_cert_file() {
        let result = TlsConfig::from_pem_files("/nonexistent/cert.pem", "/nonexistent/key.pem");
        match result {
            Err(TlsError::CertificateRead { .. }) => {}
            _ => panic!("Expected CertificateRead error"),
        }
    }
}
