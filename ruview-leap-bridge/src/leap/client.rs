use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use thiserror::Error;

use crate::bridge::types::{HandSide, LeapCommandFrame};

/// LEAP 单手运行态（MVP：仅保留 bridge 决策所需字段）。
#[derive(Clone, Debug, Default)]
pub struct LeapHandStatus {
    pub online: bool,
    pub temperature_c: f32,
    pub error_code: Option<String>,
}

/// LEAP 双手运行态快照。
#[derive(Clone, Debug, Default)]
pub struct LeapDeviceStatus {
    pub left: LeapHandStatus,
    pub right: LeapHandStatus,
}

#[derive(Debug, Error)]
pub enum LeapClientError {
    #[error("LEAP 客户端内部错误: {0}")]
    Internal(String),
}

/// LEAP SDK/驱动包装层（抽象接口，便于 mock 与后续替换真实 SDK）。
#[async_trait]
pub trait LeapClient: Send + Sync {
    /// 获取硬件状态（在线/温度/错误码）。
    async fn get_status(&self) -> Result<LeapDeviceStatus, LeapClientError>;

    /// 发布一次双手命令。
    async fn publish(&self, frame: &LeapCommandFrame) -> Result<(), LeapClientError>;
}

#[derive(Default)]
struct MockInner {
    status: LeapDeviceStatus,
    published: Vec<LeapCommandFrame>,
}

/// Mock LEAP 客户端：用于本地 demo 与测试。
#[derive(Clone, Default)]
pub struct MockLeapClient {
    inner: Arc<Mutex<MockInner>>,
}

impl MockLeapClient {
    pub fn new() -> Self {
        let mut status = LeapDeviceStatus::default();
        // 默认在线且不过温，避免 demo 模式“一启动就 not-ready”。
        status.left.online = true;
        status.right.online = true;
        Self {
            inner: Arc::new(Mutex::new(MockInner {
                status,
                published: Vec::new(),
            })),
        }
    }

    pub fn set_online(&self, side: HandSide, online: bool) {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        match side {
            HandSide::Left => inner.status.left.online = online,
            HandSide::Right => inner.status.right.online = online,
        }
    }

    pub fn set_temperature_c(&self, side: HandSide, temperature_c: f32) {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        match side {
            HandSide::Left => inner.status.left.temperature_c = temperature_c,
            HandSide::Right => inner.status.right.temperature_c = temperature_c,
        }
    }

    pub fn set_error_code(&self, side: HandSide, error_code: Option<String>) {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        match side {
            HandSide::Left => inner.status.left.error_code = error_code,
            HandSide::Right => inner.status.right.error_code = error_code,
        }
    }

    pub fn published_count(&self) -> usize {
        let inner = self.inner.lock().expect("mock lock poisoned");
        inner.published.len()
    }

    pub fn take_published(&self) -> Vec<LeapCommandFrame> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        std::mem::take(&mut inner.published)
    }
}

#[async_trait]
impl LeapClient for MockLeapClient {
    async fn get_status(&self) -> Result<LeapDeviceStatus, LeapClientError> {
        let inner = self.inner.lock().expect("mock lock poisoned");
        Ok(inner.status.clone())
    }

    async fn publish(&self, frame: &LeapCommandFrame) -> Result<(), LeapClientError> {
        let mut inner = self.inner.lock().expect("mock lock poisoned");
        inner.published.push(frame.clone());
        Ok(())
    }
}
