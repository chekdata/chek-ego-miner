use std::path::{Component, Path, PathBuf};

pub fn validate_path_component(raw: &str, field: &str) -> Result<String, String> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(format!("{field} 不能为空"));
    }
    if matches!(value, "." | "..") {
        return Err(format!("{field} 不能为 . 或 .."));
    }
    if value
        .chars()
        .any(|ch| matches!(ch, '/' | '\\' | '\0' | '\n' | '\r'))
    {
        return Err(format!("{field} 不能包含路径分隔符或控制字符"));
    }
    Ok(value.to_string())
}

pub fn validate_relative_path(raw: &str, field: &str) -> Result<PathBuf, String> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(format!("{field} 不能为空"));
    }
    let path = Path::new(value);
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(format!("{field} 必须是受限的相对路径"));
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(format!("{field} 不能为空"));
    }
    Ok(normalized)
}

pub fn join_relative(root: &Path, raw: &str, field: &str) -> Result<PathBuf, String> {
    Ok(root.join(validate_relative_path(raw, field)?))
}

pub fn session_base_dir(data_dir: &Path, session_id: &str) -> Result<PathBuf, String> {
    let safe_session_id = validate_path_component(session_id, "session_id")?;
    Ok(data_dir.join("session").join(safe_session_id))
}

pub fn ensure_session_dir_path(base_dir: &Path) -> Result<(), String> {
    let parent_name = base_dir
        .parent()
        .and_then(|path| path.file_name())
        .and_then(|value| value.to_str());
    if parent_name != Some("session") {
        return Err(format!("非法 session 目录: {}", base_dir.display()));
    }
    let session_id = base_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("非法 session 目录: {}", base_dir.display()))?;
    validate_path_component(session_id, "session_id").map(|_| ())
}

pub fn validate_fs_path(raw: &str, field: &str) -> Result<String, String> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(format!("{field} 不能为空"));
    }
    let path = Path::new(value);
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(format!("{field} 不能包含 . 或 .. 路径段"));
    }
    Ok(value.to_string())
}

pub fn ensure_no_relative_escape(path: &Path, field: &str) -> Result<(), String> {
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(format!("{field} 不能包含 . 或 .. 路径段"));
    }
    Ok(())
}
