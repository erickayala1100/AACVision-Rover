# Publish to GitHub

## 1. Create the online repository

On GitHub, create a new empty repository named:

```text
AACVision-Rover
```

Do not add a README, license, or `.gitignore` online because they are already included here.

## 2. Initialize and commit locally

From Windows PowerShell after extracting this project:

```powershell
cd "$HOME\Downloads\AACVision-Rover"
git init
git add .
git commit -m "Initial AAC Vision rover release"
git branch -M main
```

## 3. Connect the remote

Copy the HTTPS repository address shown by GitHub, then run:

```powershell
git remote add origin YOUR_COPIED_GITHUB_REPOSITORY_URL
git push -u origin main
```

## 4. Clone it on the Raspberry Pi

```bash
cd /home/pi
git clone YOUR_COPIED_GITHUB_REPOSITORY_URL
cd AACVision-Rover
chmod +x scripts/*.sh
./scripts/check_environment.sh
./scripts/run_v8_3.sh
```
