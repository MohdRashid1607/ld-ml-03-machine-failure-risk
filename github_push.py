"""
github_push.py
--------------
Pushes the LD ML 03 project to GitHub.
Run from the project root in CMD:
    python github_push.py
"""

import subprocess
import sys


def run(cmd, capture=False):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True
    )
    return result


def main():
    print("=" * 55)
    print("  LD ML 03 — GitHub Push Script")
    print("=" * 55)

    # ── 1. Check existing remotes ──────────────────────────────
    check = run("git remote -v", capture=True)
    remote_output = check.stdout.strip()

    if remote_output:
        print("\n✅ Remote already configured:")
        print(remote_output)
    else:
        print("\n⚠️  No git remote found.")
        print("\nSteps to create your GitHub repo:")
        print("  1. Go to https://github.com/new")
        print("  2. Name it:  ld-ml-03-machine-failure-risk")
        print("  3. Set it to Public or Private — your choice")
        print("  4. Do NOT add README/gitignore (repo must be empty)")
        print("  5. Copy the repo URL (e.g. https://github.com/YourUsername/ld-ml-03-machine-failure-risk.git)")
        print()
        url = input("Paste your GitHub repo URL here: ").strip()
        if not url:
            print("❌ No URL provided. Exiting.")
            sys.exit(1)
        run(f'git remote add origin "{url}"')
        print(f"✅ Remote 'origin' set to: {url}")

    # ── 2. Stage all files ─────────────────────────────────────
    print("\n📦 Staging all files...")
    run("git add .")

    # Show what's being committed
    status = run("git status --short", capture=True)
    if status.stdout.strip():
        print("Files to be committed:")
        for line in status.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print("Nothing new to stage — everything already committed.")

    # ── 3. Commit ──────────────────────────────────────────────
    msg = "v0.1.0 - Complete LD ML 03: Isolation Forest anomaly detector, 6-tab Gradio Space, full docs"
    print(f"\n💬 Committing: \"{msg}\"")
    commit = run(f'git commit -m "{msg}"', capture=True)

    if "nothing to commit" in commit.stdout.lower():
        print("✅ Nothing new to commit — repo is up to date.")
    elif commit.returncode == 0:
        print("✅ Commit created successfully.")
    else:
        # May already be committed — not a hard error
        print(f"ℹ️  Commit output: {commit.stdout or commit.stderr}")

    # ── 4. Push ────────────────────────────────────────────────
    print("\n🚀 Pushing to GitHub (main branch)...")
    push = run("git push -u origin main", capture=True)

    if push.returncode == 0:
        print("✅ Push successful!")
    else:
        # Try 'master' branch as fallback
        print(f"⚠️  Push to 'main' failed: {push.stderr.strip()}")
        print("Trying 'master' branch...")
        push2 = run("git push -u origin master", capture=True)
        if push2.returncode == 0:
            print("✅ Push to 'master' successful!")
        else:
            print(f"❌ Push failed: {push2.stderr.strip()}")
            print("\nManual fix — run these in CMD:")
            print("  git push -u origin main")
            sys.exit(1)

    # ── 5. Done ────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  🎉 ALL DONE!")
    print("=" * 55)
    check_remote = run("git remote get-url origin", capture=True)
    repo_url = check_remote.stdout.strip().replace(".git", "")
    print(f"\n  GitHub repo : {repo_url}")
    print(f"  HF Space    : https://huggingface.co/spaces/khalidml65/lottery-machine-failure-risk")
    print(f"  HF Dataset  : https://huggingface.co/datasets/khalidml65/lottery-draw-machine-telemetry")
    print(f"  HF Model    : https://huggingface.co/khalidml65/lottery-isolation-forest-detector")
    print()


if __name__ == "__main__":
    main()
