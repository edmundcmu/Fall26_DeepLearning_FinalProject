# UMA Cu/H — shared AWS + GitHub experiment starter

This starter includes the existing UMA-M Cu/H zero-shot baseline script; its GPU inference and real DFT errors have **not been executed or validated here**. This is not an implementation of FA-Joint or LoRA. The model `uma-m-1p1` and `oc20` task are configured in the baseline script. Cu/H DFT labels (FHI-aims DFT-SRP48) do **not** exactly match UMA `oc20` DFT settings (VASP RPBE).

## Suggested repository contents

- `scripts/uma_cuh_uma_m_baseline.py`: raw DFT vs unadapted UMA-M baseline, outputs MAE and plots.
- `scripts/run_baseline.sh`: runs and logs a baseline from a pinned Git SHA, optional S3 upload.
- `aws/deploy.sh`: creates immutable versioned Git worktrees; does **not** start a GPU experiment.
- `.github/workflows/sync-aws.yml`: AWS Systems Manager command on each approved push to `main`.
- `requirements-extras.txt`: helper dependencies; **pin installed versions after validation**.

## One-time AWS setup

1. Create GitHub private repository; give each teammate a GitHub collaborator role. Protect `main` with pull requests.
2. Create AWS budget alert (e.g. $100 and $300) and request On-Demand G GPU quota if it is zero. Budget alerts alone do not stop instances.
3. Launch an EC2 GPU instance (e.g. `g6e.xlarge`, 48 GB L40S; consider smaller `g6.xlarge`, 24 GB L4 for baseline only). Select a current AWS Deep Learning AMI for GPU/PyTorch on Ubuntu. Start with >=100 GB gp3 EBS and verify total pricing in your Region. Attach an IAM instance role with `AmazonSSMManagedInstanceCore` and least-privilege rights to your S3 results path. No inbound SSH port is required if using Session Manager. Instance needs outbound HTTPS.
4. Connect: AWS EC2 console > Instances > Connect > Session Manager. SSM Agent and IAM permissions must be in place.
5. On the EC2 instance, as `ubuntu`, run:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
# Activate the AMI's supported PyTorch environment if torch is not found (see AMI welcome message).
python -m pip install -r requirements-extras.txt  # after cloning the repo
hf auth login  # personal HF token with access to facebook/UMA; never commit it
```

6. In `ubuntu` account, set up **read-only GitHub deploy key** for this repo:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/uma_deploy_ed25519 -N '' -C 'uma-aws-readonly'
cat ~/.ssh/uma_deploy_ed25519.pub
```

Paste public key into GitHub repository > Settings > Deploy keys (leave write access unchecked). Add a verified GitHub `known_hosts` entry, then create `~/.ssh/config`:

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/uma_deploy_ed25519
  IdentitiesOnly yes
  StrictHostKeyChecking yes
```

```bash
chmod 600 ~/.ssh/config
# Replace ORG/REPO
# git clone git@github.com:ORG/REPO.git ~/uma-project
mkdir -p ~/bin ~/uma-data/runs ~/uma-releases
cp ~/uma-project/aws/deploy.sh ~/bin/deploy.sh
chmod +x ~/bin/deploy.sh
# On first setup:
~/bin/deploy.sh "$(git -C ~/uma-project rev-parse HEAD)"
```

NOTE: Get the repository checkout and install requirements before running the commands that require it. Verify GitHub SSH host key fingerprints against GitHub's published fingerprints before trusting the server. AWS Session Manager connects as `ssm-user` on many AMIs; use `sudo -iu ubuntu` for the one-time setup.

## Automatic GitHub → AWS synchronization

Create an AWS IAM OIDC identity provider for `https://token.actions.githubusercontent.com` (audience `sts.amazonaws.com`). Create an IAM role trusted only by your specific repo and `main` branch, and grant `ssm:SendCommand` on this one EC2 instance and the AWS-RunShellScript SSM document, plus `ssm:GetCommandInvocation` needed to check success. Refer to the latest AWS and GitHub docs for precise resource and trust policy configuration. Never use static AWS access keys in GitHub secrets.

GitHub repo Settings > Secrets and variables > Actions:
- Secret `AWS_SYNC_ROLE_ARN` → role ARN.
- Variable `AWS_REGION` → the instance Region (e.g. `us-east-1`).
- Variable `EC2_INSTANCE_ID` → the **running** GPU instance ID.

Every merged push to `main` now creates `/home/ubuntu/uma-releases/<commit>` and updates `/home/ubuntu/uma-current`. It does not install packages or start training. GitHub Actions fails to sync if the EC2 instance is stopped; rerun the workflow on startup. **Run experiments from the exact `/home/ubuntu/uma-releases/<commit>` directory**, not the moving `uma-current` symlink, to preserve reproducibility.

## Baseline experiment

```bash
# On AWS as ubuntu; activate working Python/Conda environment first
cd /home/ubuntu/uma-current
python scripts/uma_cuh_uma_m_baseline.py --inspect-only
bash scripts/run_baseline.sh
# Optional, before run: export S3_RESULTS_URI=s3://YOUR_PRIVATE_BUCKET/uma-results
```

Model access and GPU are prerequisites. Script downloads `colabfit/reactive_hydrogen_ACS_2023` (`co/co_0.parquet`) and uses `Cu54H2`, an H2-on-copper system, *not single-H Cu(111) diffusion*. It outputs force MAEs, raw and calibrated energy MAEs, plots, runtime, and a Section 5 text draft. Confirm data provenance, composition, units, group splitting, DFT task mismatch, and force constraints before treating values as final research results.

## Project experiment sequence

1. Confirm dataset provenance, label units, composition, constraints, and data splits by trajectory.
2. Baseline `U0` unadapted UMA-M vs DFT: fixed H/Cu force MAEs and calibrated energy errors.
3. Validate legal equivariant LoRA insertion, frozen backbone, zero-update equivalence, energy-gradient-derived forces, and rotation tests.
4. Establish 4–8 legal candidate adapter groups and fixed trainable-parameter budget.
5. Short-train each group independently; save H/Cu force and energy correction response signatures.
6. Compare FA-Joint greedy residual fit, FA-Independent, Force-Gradient, Random; validate summed proxy against actual combined model and final-train development masks.
7. Fresh-train the selected masks using identical data, rank, loss, schedule, and budget. Three paired seeds per method; evaluate on sealed test groups.
8. Report plots, MAEs, change from U0, total selection/training GPU-hours, variance, and failure cases. Reference DFT mismatch and potential UMA pretraining overlap.

Do not retrain automatically after every GitHub push. Training runs should be started deliberately from a pinned commit and saved to separate run directories.

## Caveats

- Baseline dataset metadata and availability: https://huggingface.co/datasets/colabfit/reactive_hydrogen_ACS_2023
- Model installation and inference: https://facebookresearch.github.io/fairchem/quickstart/
- Official fine-tuning tooling (standard fine-tuning, **not** your specialized adapter selector): https://facebookresearch.github.io/fairchem/fine-tuning/
- OIDC GitHub to AWS: https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws
- GitHub deploy keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys

No GPU training, AWS deployment, or GitHub Actions execution was performed when making this package.
