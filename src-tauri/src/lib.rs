use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;

struct BackendProcess(Mutex<Option<Child>>);

fn is_backend_healthy() -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(
        &"127.0.0.1:8000".parse().unwrap(),
        Duration::from_millis(500),
    ) else {
        return false;
    };

    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));

    let request = "GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nConnection: close\r\n\r\n";
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }

    let mut buf = [0u8; 256];
    if let Ok(n) = stream.read(&mut buf) {
        let resp = String::from_utf8_lossy(&buf[..n]);
        return resp.contains("200 OK") || resp.contains("HTTP/1.1 200") || resp.contains("HTTP/1.0 200");
    }
    false
}

fn get_project_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if dir.ends_with("src-tauri") {
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        }
    }
    dir
}

fn find_python(root: &PathBuf) -> PathBuf {
    let candidates = [
        root.join(".venv/bin/python3"),
        root.join("backend/.venv/bin/python3"),
        root.join("../.venv/bin/python3"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    PathBuf::from("python3")
}

fn spawn_backend(root: &PathBuf) -> Option<Child> {
    let server_script = root.join("backend/run_server.py");
    let python_bin = find_python(root);

    println!("[Tauri] Project root: {:?}", root);
    println!("[Tauri] Using Python: {:?}", python_bin);
    println!("[Tauri] Server script: {:?}", server_script);

    let child = Command::new(&python_bin)
        .current_dir(root)
        .arg(&server_script)
        .arg("--port")
        .arg("8000")
        .spawn();

    match child {
        Ok(c) => {
            println!("[Tauri] Backend spawned successfully with PID: {}", c.id());
            Some(c)
        }
        Err(e) => {
            eprintln!("[Tauri] Failed to spawn backend: {}", e);
            None
        }
    }
}

#[derive(serde::Serialize)]
pub struct SaveResult {
    pub success: bool,
    pub path: Option<String>,
    pub cancelled: bool,
    pub error: Option<String>,
}

#[tauri::command]
fn save_file_dialog(
    default_name: String,
    filter_name: String,
    extensions: Vec<String>,
    content: String,
    is_base64: bool,
) -> SaveResult {
    let exts: Vec<&str> = extensions.iter().map(|s| s.as_str()).collect();
    let mut dialog = rfd::FileDialog::new()
        .set_file_name(&default_name)
        .add_filter(&filter_name, &exts);

    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let dl = PathBuf::from(home).join("Downloads");
        if dl.exists() {
            dialog = dialog.set_directory(&dl);
        }
    }

    if let Some(path) = dialog.save_file() {
        let write_res = if is_base64 {
            use base64::Engine;
            let clean_b64 = if let Some(idx) = content.find(',') {
                &content[idx + 1..]
            } else {
                &content
            };
            match base64::engine::general_purpose::STANDARD.decode(clean_b64.trim()) {
                Ok(bytes) => std::fs::write(&path, bytes),
                Err(e) => {
                    return SaveResult {
                        success: false,
                        path: None,
                        cancelled: false,
                        error: Some(format!("Base64 decode error: {}", e)),
                    }
                }
            }
        } else {
            std::fs::write(&path, content.as_bytes())
        };

        match write_res {
            Ok(_) => {
                #[cfg(target_os = "macos")]
                {
                    let _ = Command::new("open").arg("-R").arg(&path).spawn();
                }
                SaveResult {
                    success: true,
                    path: Some(path.to_string_lossy().to_string()),
                    cancelled: false,
                    error: None,
                }
            }
            Err(e) => SaveResult {
                success: false,
                path: None,
                cancelled: false,
                error: Some(format!("Failed to write file: {}", e)),
            },
        }
    } else {
        SaveResult {
            success: false,
            path: None,
            cancelled: true,
            error: None,
        }
    }
}

#[tauri::command]
fn copy_file_dialog(
    default_name: String,
    filter_name: String,
    extensions: Vec<String>,
    source_path: String,
) -> SaveResult {
    let exts: Vec<&str> = extensions.iter().map(|s| s.as_str()).collect();
    let mut dialog = rfd::FileDialog::new()
        .set_file_name(&default_name)
        .add_filter(&filter_name, &exts);

    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let dl = PathBuf::from(home).join("Downloads");
        if dl.exists() {
            dialog = dialog.set_directory(&dl);
        }
    }

    if let Some(dest) = dialog.save_file() {
        let src = PathBuf::from(&source_path);
        if !src.exists() {
            return SaveResult {
                success: false,
                path: None,
                cancelled: false,
                error: Some(format!("Source file does not exist: {}", source_path)),
            };
        }
        match std::fs::copy(&src, &dest) {
            Ok(_) => {
                #[cfg(target_os = "macos")]
                {
                    let _ = Command::new("open").arg("-R").arg(&dest).spawn();
                }
                SaveResult {
                    success: true,
                    path: Some(dest.to_string_lossy().to_string()),
                    cancelled: false,
                    error: None,
                }
            }
            Err(e) => SaveResult {
                success: false,
                path: None,
                cancelled: false,
                error: Some(format!("Failed to copy file: {}", e)),
            },
        }
    } else {
        SaveResult {
            success: false,
            path: None,
            cancelled: true,
            error: None,
        }
    }
}

#[tauri::command]
fn pick_folder_dialog(default_path: Option<String>) -> Option<String> {
    let mut dialog = rfd::FileDialog::new();
    if let Some(ref dp) = default_path {
        let p = PathBuf::from(dp);
        if p.exists() {
            if p.is_dir() {
                dialog = dialog.set_directory(&p);
            } else if let Some(parent) = p.parent() {
                dialog = dialog.set_directory(parent);
            }
        }
    } else if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let dl = PathBuf::from(home).join("Downloads");
        if dl.exists() {
            dialog = dialog.set_directory(&dl);
        }
    }
    dialog.pick_folder().map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn open_folder(path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Err(format!("Path does not exist: {}", path));
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("explorer").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(&path).spawn().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            save_file_dialog,
            copy_file_dialog,
            pick_folder_dialog,
            open_folder
        ])
        .setup(|app| {
            let root = get_project_root();
            if !is_backend_healthy() {
                if let Some(c) = spawn_backend(&root) {
                    app.manage(BackendProcess(Mutex::new(Some(c))));
                }
            } else {
                app.manage(BackendProcess(Mutex::new(None)));
                println!("[Tauri] Backend is already running and healthy on port 8000.");
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<BackendProcess>() {
                    if let Ok(mut lock) = state.0.lock() {
                        if let Some(mut child) = lock.take() {
                            let pid = child.id();
                            println!("[Tauri] Terminating backend process (PID: {})...", pid);
                            let kill_res = child.kill();
                            let wait_res = child.wait();
                            println!(
                                "[Tauri] Backend process (PID: {}) terminated. kill_result: {:?}, exit_status: {:?}",
                                pid, kill_res, wait_res
                            );
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
