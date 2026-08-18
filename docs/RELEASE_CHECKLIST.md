# Release checklist

- [ ] A fresh environment passes `bash scripts/smoke_test.sh`.
- [ ] User-facing commands contain no machine-specific absolute path.
- [ ] Dataset samples, tensors, checkpoints, outputs, and evaluations are
      untracked (the public split metadata under `data/splits/` is allowed).
- [ ] No tracked file is larger than the repository's release policy.
- [ ] Secret scanning reports no credentials or private keys.
- [ ] PNN contributor ownership and redistribution permission are confirmed.
- [ ] Apache-2.0 notices and third-party licenses are retained.
- [ ] Weight/data download URLs have SHA256 checksums.
- [ ] Open-loop and closed-loop instructions were verified from a clean clone.

Suggested audit:

```bash
find . -type f -size +50M -not -path './.git/*'
git ls-files | rg '(^outputs/|^evaluation/|^checkpoints/|\\.(pth|pt|pkl|npy)$)'
git ls-files data/ | rg -v '^data/splits/bench2drive_base_train_val_split.json$'
rg -n '(/data2/|/home/|/opt/data/)' --glob '!bench2drive/scenario_runner/**'
rg -n '(API_KEY|TOKEN|PASSWORD|SECRET|BEGIN .*PRIVATE KEY)' --hidden --glob '!.git/**'
```
