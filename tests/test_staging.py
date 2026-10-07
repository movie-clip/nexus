import pytest

from nexus.staging import Change, StagedBackend, apply_changes

CRLF_FILE = "class A {\r\n    int x = 1;\r\n}\r\n"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "A.cs").write_bytes(CRLF_FILE.encode())
    (root / "README.md").write_bytes(b"hello\n")
    return root


@pytest.fixture
def backend(project, tmp_path):
    return StagedBackend(project, tmp_path / "staging")


def raw(path) -> bytes:
    return path.read_bytes()


def test_edit_is_staged_and_project_is_untouched(project, backend):
    result = backend.edit("/src/A.cs", "int x = 1;", "int x = 2;")
    assert result.error is None and result.occurrences == 1
    assert raw(project / "src" / "A.cs") == CRLF_FILE.encode()  # nothing on disk changed
    assert "int x = 2;" in backend.read("/src/A.cs").file_data["content"]  # agent sees its edit


def test_edit_keeps_crlf_line_endings(backend):
    backend.edit("/src/A.cs", "int x = 1;", "int x = 2;\n    int y = 3;")
    (change,) = backend.changes()
    assert change.after == "class A {\r\n    int x = 2;\r\n    int y = 3;\r\n}\r\n"
    assert "\n" not in change.after.replace("\r\n", "")  # no bare LF anywhere


def test_old_string_with_crlf_still_matches(backend):
    result = backend.edit("/src/A.cs", "class A {\r\n    int x", "class B {\r\n    int x")
    assert result.error is None
    assert backend.changes()[0].after.startswith("class B {\r\n")


def test_repeated_edits_build_on_the_staged_version(backend):
    backend.edit("/src/A.cs", "int x = 1;", "int x = 2;")
    backend.edit("/src/A.cs", "int x = 2;", "int x = 3;")
    assert "int x = 3;" in backend.changes()[0].after


def test_new_file_takes_a_neighbours_line_ending(backend):
    backend.write("/src/B.cs", "class B {\n}\n")
    (change,) = backend.changes()
    assert change.before is None
    assert change.after == "class B {\r\n}\r\n"


def test_new_file_without_neighbours_uses_lf(backend):
    backend.write("/docs/new.md", "line\n")
    assert backend.changes()[0].after == "line\n"


def test_delete_is_staged(project, backend):
    assert backend.delete("/README.md").error is None
    assert (project / "README.md").exists()
    assert backend.read("/README.md").error == "File '/README.md' not found"
    assert backend.changes() == [Change("/README.md", "hello\n", None)]


def test_write_then_delete_of_a_new_file_is_no_change(backend):
    backend.write("/tmp.txt", "x")
    backend.delete("/tmp.txt")
    assert backend.changes() == []


def test_unchanged_rewrite_is_not_a_change(backend):
    backend.write("/README.md", "hello\n")
    assert backend.changes() == []


@pytest.mark.parametrize("path", ["/../outside.txt", "/src/../../outside.txt", "relative.txt"])
def test_writes_outside_the_project_are_refused(backend, path, tmp_path):
    assert backend.write(path, "x").error.startswith("Path traversal not allowed")
    assert not (tmp_path / "outside.txt").exists()


def test_edit_of_missing_file_reports_not_found(backend):
    assert backend.edit("/nope.cs", "a", "b").error == "Error: File '/nope.cs' not found"


def test_listings_show_the_project_as_it_was(backend):
    backend.write("/src/B.cs", "class B {}\n")
    names = [entry["path"] for entry in backend.ls("/src").entries]
    assert any(n.endswith("A.cs") for n in names)
    assert not any(n.endswith("B.cs") for n in names)


# --- applying an approved batch -----------------------------------------------


def test_apply_writes_edits_creates_and_deletes(project, backend):
    backend.edit("/src/A.cs", "int x = 1;", "int x = 2;")
    backend.write("/src/B.cs", "class B {}\n")
    backend.delete("/README.md")
    applied, conflicts = apply_changes(project, backend.changes())
    assert conflicts == []
    assert sorted(applied) == ["/README.md", "/src/A.cs", "/src/B.cs"]
    assert raw(project / "src" / "A.cs") == CRLF_FILE.replace("1", "2").encode()
    assert (project / "src" / "B.cs").exists()
    assert not (project / "README.md").exists()


def test_apply_refuses_files_changed_since_staging(project, backend):
    backend.edit("/src/A.cs", "int x = 1;", "int x = 2;")
    batch = backend.changes()
    (project / "src" / "A.cs").write_bytes(b"someone else edited this\r\n")
    applied, conflicts = apply_changes(project, batch)
    assert (applied, conflicts) == ([], ["/src/A.cs"])
    assert raw(project / "src" / "A.cs") == b"someone else edited this\r\n"


def test_apply_twice_is_safe(project, backend):
    backend.edit("/src/A.cs", "int x = 1;", "int x = 2;")
    batch = backend.changes()
    apply_changes(project, batch)
    assert apply_changes(project, batch) == (["/src/A.cs"], [])


# --- mixed line endings (found on a real Unity file) ---------------------------

MIXED = "a\r\nb\nc\r\nd\n"


def test_mixed_endings_survive_an_edit_line_by_line(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "m.cs").write_bytes(MIXED.encode())
    backend = StagedBackend(root, tmp_path / "s")
    backend.edit("/m.cs", "b", "B1\nB2")
    # Untouched lines keep their own ending; the replaced line's ending (LF)
    # goes to both lines that replace it.
    assert backend.changes()[0].after == "a\r\nB1\nB2\nc\r\nd\n"


def test_full_rewrite_keeps_endings_of_unchanged_lines(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "m.cs").write_bytes(MIXED.encode())
    backend = StagedBackend(root, tmp_path / "s")
    backend.write("/m.cs", "a\nb\nC\nd\n")
    assert backend.changes()[0].after == "a\r\nb\nC\r\nd\n"


def test_missing_final_newline_is_kept(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "m.cs").write_bytes(b"a\r\nb")
    backend = StagedBackend(root, tmp_path / "s")
    backend.edit("/m.cs", "a", "A")
    assert backend.changes()[0].after == "A\r\nb"


REAL_FILE = (
    "C:/projects/impact/Impact/Packages/athena.shared/Runtime/Components/Character/"
    "CharacterAnimationState.cs"
)


@pytest.mark.skipif(
    not __import__("pathlib").Path(REAL_FILE).exists(), reason="impact checkout not present"
)
def test_replay_of_the_real_mixed_ending_file_changes_only_the_edited_lines(tmp_path):
    import pathlib

    original = pathlib.Path(REAL_FILE).read_bytes()
    root = tmp_path / "p"
    root.mkdir()
    (root / "S.cs").write_bytes(original)
    backend = StagedBackend(root, tmp_path / "s")
    result = backend.edit(
        "/S.cs",
        "/// Is jump input active this frame?",
        "/// True on the landing frame.\n/// Not input.",
    )
    assert result.error is None
    before = original.decode().splitlines(keepends=True)
    after = backend.changes()[0].after.splitlines(keepends=True)
    changed_before = [line for line in before if line not in after]
    changed_after = [line for line in after if line not in before]
    assert [line.strip() for line in changed_before] == ["/// Is jump input active this frame?"]
    assert [line.strip() for line in changed_after] == [
        "/// True on the landing frame.",
        "/// Not input.",
    ]
