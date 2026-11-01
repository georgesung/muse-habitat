#!/usr/bin/env bash
# For containers whose NVIDIA driver mount has no EGL/GL libraries (habitat-sim: "libEGL.so.1: cannot open shared
# object file"): take them from the driver package of the running kernel driver's version, unpacked only (nothing is
# installed), into third_party/nvidia-egl: the libglvnd loaders (libEGL, libOpenGL, libGLdispatch, and libGL + libGLX so
# that OpenCV's libGL matches this libGLdispatch), the NVIDIA EGL vendor library and its GL cores, and the vendor json. scripts/env.sh points the loader at them.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
V=$(sed -n 's/.*Kernel Module for x86_64 *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version)
run=NVIDIA-Linux-x86_64-$V.run
dl="$root/third_party/nvidia-dl"
out="$root/third_party/nvidia-egl"
mkdir -p "$dl" "$out/lib"
cd "$dl"
[ -f "$run" ] || curl -sSfLO "https://us.download.nvidia.com/tesla/$V/$run"
[ -d x ] || sh "$run" --extract-only --target x >/dev/null
cp x/libEGL.so.1.1.0 x/libOpenGL.so.0 x/libGLdispatch.so.0 x/libGL.so.1.7.0 x/libGLX.so.0 "$out/lib/"
for l in libEGL_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-glvkspirv libnvidia-tls; do
    cp "x/$l.so.$V" "$out/lib/"
done
ln -sf libEGL.so.1.1.0 "$out/lib/libEGL.so.1"
ln -sf libGL.so.1.7.0 "$out/lib/libGL.so.1"
ln -sf "libEGL_nvidia.so.$V" "$out/lib/libEGL_nvidia.so.0"
sed "s#libEGL_nvidia.so.0#$out/lib/libEGL_nvidia.so.0#" x/10_nvidia.json > "$out/10_nvidia.json"
echo "driver $V EGL libraries in $out"
