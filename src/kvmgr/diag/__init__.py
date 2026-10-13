"""kvmgr.diag -- opt-in diagnostics, isolated from everything functional.

Each module here is installed by ``kvmgr.register._install_diagnostics`` and
only when its knob is on; nothing else in kvmgr imports this package (a lint in
the tests enforces it). Diagnostics reach core through ``kvmgr.hooks`` and
never the other way round, so switching one off cannot change behaviour.

| module       | knob                    | default | why                                     |
|--------------|-------------------------|---------|-----------------------------------------|
| hangdump     | diag_hang_dump          | on      | zero cost; a hang with it absent cost two wedges |
| ledger       | diag_transfer_ledger    | on      | a few integers per transfer; the only thing an async Xid leaves |
| serializer   | diag_serializer_walk    | off     | names the producer of a numpy scalar; the COERCION is core |
| geometry     | diag_geometry_snapshot  | off     | writes the KV tensor geometry to a JSON file at handler build |
| tiering      | diag_tiering_shims      | off     | shims for upstream's TieringOffloadingSpec, which our spec does not use |
| reload       | diag_hot_reload         | off     | re-import kvmgr in a running engine; `/kv/reload` exists only when on |
"""
