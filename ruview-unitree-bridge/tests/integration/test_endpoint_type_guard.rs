use ruview_unitree_bridge::bridge::validator::{DexKind, EndpointGuard};

#[test]
fn integration_endpoint_guard_matrix() {
    let leap = EndpointGuard::new("LEAP_V2");
    assert!(!leap.allow_dex(DexKind::Dex3));
    assert!(!leap.allow_dex(DexKind::Dex1));

    let dex3 = EndpointGuard::new("DEX3");
    assert!(dex3.allow_dex(DexKind::Dex3));
    assert!(!dex3.allow_dex(DexKind::Dex1));

    let dex1 = EndpointGuard::new("DEX1");
    assert!(!dex1.allow_dex(DexKind::Dex3));
    assert!(dex1.allow_dex(DexKind::Dex1));
}
