from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


WORKFLOW = Path(__file__).parents[1] / ".github/workflows/release.yml"


@pytest.fixture
def workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def step(workflow: dict, job: str, name: str) -> dict:
    return next(item for item in workflow["jobs"][job]["steps"] if item["name"] == name)


def run_embedded_python(step: dict) -> None:
    code = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    exec(compile(code, str(WORKFLOW), "exec"), {})


def test_release_workflow_is_tag_driven_and_gated(workflow: dict) -> None:
    assert workflow["on"]["push"]["tags"] == ["v*"]
    jobs = workflow["jobs"]
    assert set(jobs) == {"validate-release", "tests", "python-package", "windows-portable", "publish"}
    assert workflow["permissions"] == {"contents": "read"}
    assert jobs["publish"]["permissions"] == {"contents": "write"}
    assert jobs["publish"]["needs"] == ["validate-release", "tests", "python-package", "windows-portable"]
    for name in ("tests", "python-package", "windows-portable"):
        assert jobs[name]["needs"] == ["validate-release"]
        assert "permissions" not in jobs[name]
    assert "permissions" not in jobs["validate-release"]
    checkout = step(workflow, "validate-release", "Check out repository")
    assert checkout["with"]["fetch-depth"] == "0"
    assert checkout["with"]["persist-credentials"] == "false"


def test_release_workflow_runs_the_supported_test_matrix(workflow: dict) -> None:
    assert workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12", "3.13"]
    assert step(workflow, "tests", "Run test suite")["run"] == "python -m pytest -vv"


def test_release_version_flows_into_build_install_upload_and_publish(workflow: dict) -> None:
    assert "0.1.0" not in WORKFLOW.read_text(encoding="utf-8")
    gate = step(workflow, "validate-release", "Validate main ancestry, tag and source metadata")
    assert gate["id"] == "release"
    assert workflow["jobs"]["validate-release"]["outputs"]["version"] == "${{ steps.release.outputs.version }}"
    for job in ("python-package", "publish"):
        assert workflow["jobs"][job]["env"]["RELEASE_VERSION"] == "${{ needs.validate-release.outputs.version }}"
    build = step(workflow, "python-package", "Build wheel and sdist")["run"]
    assert "python -m build" in build
    for kind, suffix in (("wheel", "py3-none-any.whl"), ("sdist", "tar.gz")):
        filename = f"nexusmind-${{RELEASE_VERSION}}{'-' if kind == 'wheel' else '.'}{suffix}"
        assert filename in build
        clean_install = step(workflow, "python-package", f"Clean-install and verify {kind}")["run"]
        assert f'venv-{kind}/bin/python" -m pip install "$GITHUB_WORKSPACE/dist/{filename}"' in clean_install
        assert f'venv-{kind}/bin/python" -m pip check' in clean_install
        assert f'venv-{kind}/bin/nexusmind" --help' in clean_install
        assert "nexusmind-kb" in clean_install
        assert filename in step(workflow, "publish", "Create GitHub Release")["run"]
    upload = step(workflow, "python-package", "Upload verified Python distributions")["with"]
    assert upload["path"].splitlines() == [
        "dist/nexusmind-${{ env.RELEASE_VERSION }}-py3-none-any.whl",
        "dist/nexusmind-${{ env.RELEASE_VERSION }}.tar.gz",
    ]
    assert upload["if-no-files-found"] == "error"
    publish = step(workflow, "publish", "Create GitHub Release")["run"]
    assert "gh release create" in publish
    assert "--verify-tag" in publish
    assert '--notes-file "docs/releases/${GITHUB_REF_NAME}.md"' in publish
    assert "release-artifacts/nexusmind-windows-portable.zip" in publish
    assert "./scripts/build-portable.ps1" in step(workflow, "windows-portable", "Build and E2E smoke-test portable ZIP")["run"]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


@pytest.fixture
def release_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    git(repo, "config", "user.name", "Release test")
    git(repo, "config", "user.email", "release-test@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "config", "tag.gpgsign", "false")
    git(repo, "commit", "--allow-empty", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-b", "feature")
    git(repo, "commit", "--allow-empty", "-m", "unmerged feature")
    feature = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "main")
    git(repo, "commit", "--allow-empty", "-m", "main tip")
    main = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/main", main)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_OUTPUT", str(repo / "outputs"))
    return repo, {"base": base, "feature": feature, "main": main}


def prepare_release(repo: Path, monkeypatch: pytest.MonkeyPatch, commit: str, version: str, *, annotated: bool = False) -> None:
    tag = f"v{version}"
    git(repo, "tag", *(["-a", "-m", "Release"] if annotated else []), tag, commit)
    git(repo, "checkout", "--detach", tag)
    monkeypatch.setenv("GITHUB_REF_NAME", tag)
    monkeypatch.setenv("GITHUB_SHA", commit)
    (repo / "pyproject.toml").write_text(
        f'[project]\nversion = "{version}"\nrequires-python = ">=3.11,<3.14"\nlicense = "MIT"\n',
        encoding="utf-8",
    )
    notes = repo / "docs/releases" / f"{tag}.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("Release notes\n", encoding="utf-8")


@pytest.mark.parametrize("version", ["0.1.0", "0.1.1"])
@pytest.mark.parametrize("target,annotated", [("main", False), ("base", True)])
def test_release_gate_accepts_tags_in_main_history(workflow: dict, release_repo, monkeypatch, version, target, annotated) -> None:
    repo, commits = release_repo
    prepare_release(repo, monkeypatch, commits[target], version, annotated=annotated)
    run_embedded_python(step(workflow, "validate-release", "Validate main ancestry, tag and source metadata"))
    assert (repo / "outputs").read_text(encoding="utf-8") == f"version={version}\n"


@pytest.mark.parametrize("failure", ["unmerged", "missing-main", "wrong-head", "wrong-sha", "wrong-version", "wrong-python", "wrong-license", "missing-notes"])
def test_release_gate_rejects_invalid_release(workflow: dict, release_repo, monkeypatch, failure) -> None:
    repo, commits = release_repo
    prepare_release(repo, monkeypatch, commits["feature" if failure == "unmerged" else "main"], "0.1.1")
    if failure == "missing-main":
        git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    elif failure == "wrong-head":
        git(repo, "checkout", "--detach", commits["base"])
    elif failure == "wrong-sha":
        monkeypatch.setenv("GITHUB_SHA", commits["base"])
    elif failure.startswith("wrong-"):
        old, new = {
            "wrong-version": ("0.1.1", "0.1.0"),
            "wrong-python": (">=3.11,<3.14", ">=3.10"),
            "wrong-license": ("MIT", "Apache-2.0"),
        }[failure]
        metadata = repo / "pyproject.toml"
        metadata.write_text(metadata.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    elif failure == "missing-notes":
        (repo / "docs/releases/v0.1.1.md").unlink()
    with pytest.raises((SystemExit, subprocess.CalledProcessError)):
        run_embedded_python(step(workflow, "validate-release", "Validate main ancestry, tag and source metadata"))
    assert not (repo / "outputs").exists()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("field,value,error", [
    ("Requires-Python", ">=3.11,<3.14", None),
    ("Requires-Python", "<3.14,>=3.11", None),
    ("Requires-Python", " <3.14, >=3.11 ", None),
    ("Requires-Python", None, "Requires-Python"),
    ("Requires-Python", ">=3.11", "Requires-Python"),
    ("Requires-Python", ">=3.10,<3.14", "Requires-Python"),
    ("Requires-Python", ">=3.11,<3.13", "Requires-Python"),
    ("Version", "0.1.0", "version"),
    ("License-Expression", "Apache-2.0", "license"),
])
def test_clean_install_verifies_final_metadata(workflow, tmp_path, monkeypatch, kind, field, value, error) -> None:
    metadata = {"Version": "0.1.1", "License-Expression": "MIT", "Requires-Python": ">=3.11,<3.14"}
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    monkeypatch.setenv("RELEASE_VERSION", "0.1.1")
    monkeypatch.setattr(importlib.metadata, "metadata", lambda name: metadata)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    scripts = []
    for name in ("nexusmind", "nexusmind-kb"):
        (tmp_path / name).touch()
        scripts.append(SimpleNamespace(name=name, load=lambda: lambda: None))
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: scripts)
    verification = step(workflow, "python-package", f"Clean-install and verify {kind}")
    if error is None:
        run_embedded_python(verification)
    else:
        with pytest.raises(SystemExit, match=error):
            run_embedded_python(verification)


@pytest.mark.parametrize("version", ["0.1.0", "0.1.1"])
@pytest.mark.parametrize("problem", [None, "missing", "extra", "stale"])
def test_publication_set_uses_derived_version(workflow, tmp_path, monkeypatch, version, problem) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_VERSION", version)
    artifacts = tmp_path / "release-artifacts"
    artifacts.mkdir()
    wheel = artifacts / f"nexusmind-{version}-py3-none-any.whl"
    for name in (wheel.name, f"nexusmind-{version}.tar.gz", "nexusmind-windows-portable.zip"):
        (artifacts / name).touch()
    if problem == "missing":
        wheel.unlink()
    elif problem == "extra":
        (artifacts / "unexpected.zip").touch()
    elif problem == "stale":
        wheel.rename(artifacts / "nexusmind-0.0.9-py3-none-any.whl")
    verification = step(workflow, "publish", "Verify publication set")
    if problem is None:
        run_embedded_python(verification)
    else:
        with pytest.raises(SystemExit, match="release artifact mismatch"):
            run_embedded_python(verification)


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("remote_change", [None, "moved", "deleted", "unavailable"])
def test_publication_rechecks_remote_tag(workflow, release_repo, monkeypatch, annotated, remote_change) -> None:
    repo, commits = release_repo
    prepare_release(repo, monkeypatch, commits["main"], "0.1.1", annotated=annotated)
    run_embedded_python(step(workflow, "validate-release", "Validate main ancestry, tag and source metadata"))
    remote = repo.parent / "remote.git"
    git(repo, "clone", "--bare", str(repo), str(remote))
    git(repo, "remote", "add", "origin", remote.as_uri())
    tag_ref = "refs/tags/v0.1.1"
    original_tag = git(repo, "rev-parse", tag_ref)
    if remote_change == "moved":
        # Change only the remote; the build checkout still has the original tag.
        if annotated:
            git(repo, "tag", "-a", "-m", "Replacement", "replacement", commits["feature"])
            git(repo, "push", "origin", f"refs/tags/replacement:{tag_ref}", "--force")
        else:
            git(remote, "update-ref", tag_ref, commits["feature"])
    elif remote_change == "deleted":
        git(remote, "update-ref", "-d", tag_ref)
    elif remote_change == "unavailable":
        git(repo, "remote", "set-url", "origin", (repo.parent / "missing.git").as_uri())
    assert git(repo, "rev-parse", tag_ref) == original_tag
    publication = step(workflow, "publish", "Create GitHub Release")
    assert publication["run"].startswith("set -euo pipefail\n")
    assert publication["run"].index("git") < publication["run"].index("gh release create")
    if remote_change is None:
        run_embedded_python(publication)
    else:
        with pytest.raises((SystemExit, subprocess.CalledProcessError)):
            run_embedded_python(publication)
