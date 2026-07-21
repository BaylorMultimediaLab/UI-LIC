"""
Unified Interface For Learned Image Compression (UI-LIC) - Docker VMAF Helper Module

This module (`vmaf_docker.py`) provides an OS-agnostic interface for calculating VMAF (Video Multi-Method Assessment Fusion)
perceptual quality scores using Docker. By mounting local reconstructed and reference images read-only into a pre-compiled
static FFmpeg container (`mwader/static-ffmpeg`), it avoids complex host system FFmpeg compilation requirements.
"""

import subprocess
import os
import re

def check_docker_availability():
    """
    Checks if Docker is installed and the Docker daemon is accessible.
    Returns: (bool, message)
    """
    try:
        # Check if 'docker' command exists and can talk to the daemon
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, "Docker is available."
        else:
            return False, "Docker is installed, but the daemon isn't running or you lack permissions (try adding your user to the 'docker' group)."
    except FileNotFoundError:
        return False, "Docker command not found. Please install Docker to use VMAF."
    except Exception as e:
        return False, f"Unexpected error checking Docker: {str(e)}"

def calculate_vmaf(dist_path, ref_path, docker_image="mwader/static-ffmpeg"):
    """
    Calculates VMAF score by mounting local files into a containerized FFmpeg runtime.
    
    Mounts reference and reconstructed images as read-only volumes (:ro) into the container,
    invokes the libvmaf filter via FFmpeg, and parses the resulting score from stderr.
    
    Returns:
        float: The calculated VMAF score, or 0.0 if file validation or container execution fails.
    """
    if not os.path.exists(dist_path) or not os.path.exists(ref_path):
        return 0.0

    # Get the clean, absolute path
    dist_abs = os.path.abspath(os.path.normpath(dist_path))
    ref_abs = os.path.abspath(os.path.normpath(ref_path))
    
    # Convert Windows backslashes to forward slashes for Docker CLI compatibility.
    # This is safe for Linux/Mac as they already use forward slashes.
    dist_abs = dist_abs.replace('\\', '/')
    ref_abs = ref_abs.replace('\\', '/')
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{dist_abs}:/dist.png:ro",
        "-v", f"{ref_abs}:/ref.png:ro",
        docker_image,
        "-i", "/dist.png", 
        "-i", "/ref.png",
        "-lavfi", "libvmaf",
        "-f", "null", "-"
    ]
    
    try:
        # FFmpeg outputs VMAF info to stderr
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stderr
        
        # Look for "VMAF score: <number>"
        match = re.search(r"VMAF score:\s+([\d\.]+)", output)
        if match:
            return float(match.group(1))
        else:
            match = re.search(r"VMAF score:\s*(\d+\.\d+)", output)
            if match:
                return float(match.group(1))
                
            print(f"[VMAF Warning] Could not find VMAF score in output for {os.path.basename(dist_path)}")
            return 0.0
            
    except subprocess.CalledProcessError as e:
        print(f"[VMAF Error] Docker command failed for {os.path.basename(dist_path)}")
        print(f"Error output: {e.stderr[:200]}...")
        return 0.0
    except Exception as e:
        print(f"[VMAF Error] Unexpected error: {e}")
        return 0.0

if __name__ == "__main__":
    # Quick test if run directly
    import sys
    if len(sys.argv) == 3:
        score = calculate_vmaf(sys.argv[1], sys.argv[2])
        print(f"VMAF Score: {score}")
    else:
        print("Usage: python vmaf_docker.py <distorted_image> <reference_image>")
