# Security

Please report a vulnerability privately, through GitHub's private
vulnerability reporting (the repository's **Security** tab, "Report a
vulnerability"), not in a public issue.

kvmgr runs inside a vLLM engine and serves HTTP endpoints (README
"Endpoints") with no authentication of their own. The stats listener binds
127.0.0.1. The control plane (`kvmgr_control_plane=1`) shares the engine's own
port and can change knobs and drop the GPU cache, so anything that proxies that
port exposes it: enable it only behind a trusted network.

Only the latest release tag is supported.
