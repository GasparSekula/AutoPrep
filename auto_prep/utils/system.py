import platform

import psutil


def get_system_info():
    """Collect system stats"""

    # Collect platform information
    system_info = {
        "System": platform.system(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Architecture": platform.architecture()[0],
        "Python Version": platform.python_version(),
    }

    # Collect CPU information
    system_info.update(
        {
            "Physical Cores": psutil.cpu_count(logical=False),
            "Logical Cores": psutil.cpu_count(logical=True),
            "CPU Frequency (MHz)": (
                psutil.cpu_freq().current if psutil.cpu_freq() else None
            ),
        }
    )

    # Collect Memory (RAM) information
    virtual_memory = psutil.virtual_memory()
    system_info.update(
        {
            "Total RAM (GB)": round(virtual_memory.total / (1024**3), 2),
            "Available RAM (GB)": round(virtual_memory.available / (1024**3), 2),
        }
    )

    # Collect Disk information
    disk_usage = psutil.disk_usage("/")
    system_info.update(
        {
            "Total Disk Space (GB)": round(disk_usage.total / (1024**3), 2),
            "Free Disk Space (GB)": round(disk_usage.free / (1024**3), 2),
        }
    )

    return system_info
