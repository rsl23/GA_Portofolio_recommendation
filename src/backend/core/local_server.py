"""Foreground development entry point with a clear occupied-port message."""
import socket

import uvicorn


def run_local_backend():
    try:
        connection = socket.create_connection(("127.0.0.1", 8000), timeout=.5)
    except OSError:
        pass
    else:
        connection.close()
        print("Port 8000 sudah digunakan. Server kedua tidak dijalankan.")
        print("Periksa backend: .\\start_backend.ps1 -Action status")
        print("Untuk memakai terminal ini, hentikan backend terpisah terlebih dahulu:")
        print(".\\start_backend.ps1 -Action stop")
        print("Lalu jalankan kembali: python -m src.backend.app")
        return False
    # Foreground execution belongs to this terminal; avoid spawning an editor
    # reloader that can outlive Ctrl+C. Use restart after editing backend code.
    uvicorn.run("src.backend.app:app", host="127.0.0.1", port=8000, reload=False,
                timeout_graceful_shutdown=5)
    return True
