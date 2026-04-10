use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct ArmCommand {
    pub edge_time_ns: u64,
    pub q: Vec<f32>,
    pub tau: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct DexCommand {
    pub edge_time_ns: u64,
    pub q: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum PublishedMessage {
    Arm(ArmCommand),
    Dex3Left(DexCommand),
    Dex3Right(DexCommand),
    Dex1Left(DexCommand),
    Dex1Right(DexCommand),
}

#[derive(Debug, Error)]
pub enum DdsPublishError {
    #[error("DDS 发布失败: {0}")]
    PublishFailed(String),
}

#[async_trait]
pub trait UnitreeClient: Send + Sync {
    async fn publish_arm(&self, cmd: &ArmCommand) -> Result<(), DdsPublishError>;
    async fn publish_dex3_left(&self, cmd: &DexCommand) -> Result<(), DdsPublishError>;
    async fn publish_dex3_right(&self, cmd: &DexCommand) -> Result<(), DdsPublishError>;
    async fn publish_dex1_left(&self, cmd: &DexCommand) -> Result<(), DdsPublishError>;
    async fn publish_dex1_right(&self, cmd: &DexCommand) -> Result<(), DdsPublishError>;
}

#[derive(Default)]
struct MockInner {
    published: Vec<PublishedMessage>,
    fail_publish: bool,
}

/// Mock Unitree DDS 客户端：用于本地 demo 与测试。
#[derive(Clone, Default)]
pub struct MockUnitreeClient {
    inner: Arc<Mutex<MockInner>>,
}

impl MockUnitreeClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_fail_publish(&self, fail: bool) {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        inner.fail_publish = fail;
    }

    pub fn published_count(&self) -> usize {
        let inner = self.inner.lock().expect("mock lock poisoned");
        inner.published.len()
    }

    pub fn take_published(&self) -> Vec<PublishedMessage> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        std::mem::take(&mut inner.published)
    }
}

#[async_trait]
impl UnitreeClient for MockUnitreeClient {
    async fn publish_arm(&self, cmd: &ArmCommand) -> Result<(), DdsPublishError> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        if inner.fail_publish {
            return Err(DdsPublishError::PublishFailed("mock fail".to_string()));
        }
        inner.published.push(PublishedMessage::Arm(cmd.clone()));
        Ok(())
    }

    async fn publish_dex3_left(&self, cmd: &DexCommand) -> Result<(), DdsPublishError> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        if inner.fail_publish {
            return Err(DdsPublishError::PublishFailed("mock fail".to_string()));
        }
        inner
            .published
            .push(PublishedMessage::Dex3Left(cmd.clone()));
        Ok(())
    }

    async fn publish_dex3_right(&self, cmd: &DexCommand) -> Result<(), DdsPublishError> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        if inner.fail_publish {
            return Err(DdsPublishError::PublishFailed("mock fail".to_string()));
        }
        inner
            .published
            .push(PublishedMessage::Dex3Right(cmd.clone()));
        Ok(())
    }

    async fn publish_dex1_left(&self, cmd: &DexCommand) -> Result<(), DdsPublishError> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        if inner.fail_publish {
            return Err(DdsPublishError::PublishFailed("mock fail".to_string()));
        }
        inner
            .published
            .push(PublishedMessage::Dex1Left(cmd.clone()));
        Ok(())
    }

    async fn publish_dex1_right(&self, cmd: &DexCommand) -> Result<(), DdsPublishError> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        if inner.fail_publish {
            return Err(DdsPublishError::PublishFailed("mock fail".to_string()));
        }
        inner
            .published
            .push(PublishedMessage::Dex1Right(cmd.clone()));
        Ok(())
    }
}
