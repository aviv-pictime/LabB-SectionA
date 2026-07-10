# How to push this repo to GitHub

Local git is fully prepared: one commit on branch `main`, remote `origin` set to
`https://github.com/avivrabi/LabB-SectionA.git`, clean working tree.
The only missing piece is the **empty repo existing on GitHub**. `git push` cannot
create it for you.

## 1. Create the empty repo (once)

Go to **https://github.com/new** and set:

- **Owner:** `avivrabi`
- **Repository name:** `LabB-SectionA` (must match exactly)
- **Visibility:** Private
- **Do NOT** add a README, .gitignore, or license — leave it completely empty
  (otherwise the first push conflicts with the auto-created commit)
- Click **Create repository**

## 2. Push

```powershell
cd C:\Users\AvivRabi\PycharmProjects\LabB-SectionA
git push -u origin main
```

If a GitHub login window appears, authenticate as **avivrabi**.

## Troubleshooting: still "Repository not found" after creating it

This means the terminal is authenticated as a **different GitHub account** that
can't see `avivrabi`'s private repo.

1. Confirm the credential helper:
   ```powershell
   git config credential.helper   # expected: manager
   ```
2. Remove the stale credential so git re-prompts:
   - Windows → search **"Credential Manager"** → **Windows Credentials**
   - Find and **Remove** any `git:https://github.com` entry
3. Re-run `git push -u origin main` and log in as `avivrabi`.

## What gets pushed

Tracked files (8): `strategy.py`, `run.py`, `utils.py`, `evaluation.py`,
`process.md`, `process_summary.md`, `README.md`, `.gitignore`.

Excluded (via `.gitignore`, kept local only): `data/`, `constants.yaml`,
`.idea/`, `.eval_cache/`, `.venv/`.
