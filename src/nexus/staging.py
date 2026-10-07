"""A Deep Agents backend that stages writes instead of touching the project.

The engineer agent sees the project through ``StagedBackend``:

* ``read_file`` shows the staged version of a file once the agent changed it,
  otherwise the project's file;
* ``write_file`` / ``edit_file`` / ``delete`` only record the change, in a
  staging folder outside the project;
* ``ls`` / ``glob`` / ``grep`` show the project as it was before the agent's
  changes.

When the agent is done, ``changes()`` returns the batch as before/after text,
which the user approves or rejects as a whole; ``apply_changes`` writes an
approved batch to the project.

Line endings are kept line by line (``restore_endings``): files can mix CRLF
and LF, and the review diff doesn't show endings, so any ending the change
didn't need to touch must stay byte-identical. Deep Agents' own ``edit`` reads
with universal newlines and writes back with ``\\n``, which would turn a
one-line edit of a CRLF file into a whole-file change.
"""

import difflib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import perform_string_replacement


@dataclass(frozen=True)
class Change:
    """One file in a staged batch. ``None`` means the file does not exist."""

    path: str
    before: str | None
    after: str | None


class StagedBackend(BackendProtocol):
    def __init__(self, project: str | Path, staging_dir: str | Path) -> None:
        self.root = Path(project).resolve()
        self.stage_root = Path(staging_dir).resolve()
        self.base = FilesystemBackend(root_dir=self.root, virtual_mode=True)
        self.stage = FilesystemBackend(root_dir=self.stage_root, virtual_mode=True)
        self.touched: set[str] = set()  # staged writes and edits
        self.deleted: set[str] = set()

    # --- paths ----------------------------------------------------------------

    def _rel(self, file_path: str) -> PurePosixPath:
        path = PurePosixPath(file_path.replace("\\", "/"))
        if not path.is_absolute() or ".." in path.parts or "~" in path.parts:
            raise ValueError(f"Path traversal not allowed: {file_path}")
        return path.relative_to("/")

    def _project_file(self, file_path: str) -> Path:
        return self.root / self._rel(file_path)

    def _staged_file(self, file_path: str) -> Path:
        return self.stage_root / self._rel(file_path)

    # --- reads: staged version first ------------------------------------------

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if file_path in self.deleted:
            return ReadResult(error=f"File '{file_path}' not found")
        if file_path in self.touched:
            return self.stage.read(file_path, offset, limit)
        return self.base.read(file_path, offset, limit)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = []
        for path in paths:
            if path in self.deleted:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            elif path in self.touched:
                responses.extend(self.stage.download_files([path]))
            else:
                responses.extend(self.base.download_files([path]))
        return responses

    def ls(self, path: str) -> LsResult:
        return self.base.ls(path)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.base.glob(pattern, path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        return self.base.grep(pattern, path, glob, max_count=max_count)

    # --- writes: staged only --------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            target = self._staged_file(file_path)
        except ValueError as exc:
            return WriteResult(error=str(exc))
        current = self._current_text(file_path)
        if current is not None:
            text = restore_endings(current, _to_lf(content))
        else:
            text = _with_ending(_to_lf(content), self._ending_for(file_path))
        self._put(target, text)
        self.deleted.discard(file_path)
        self.touched.add(file_path)
        return WriteResult(path=file_path)

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        current = self._current_text(file_path)
        if current is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        result = perform_string_replacement(
            _to_lf(current), _to_lf(old_string), _to_lf(new_string), replace_all
        )
        if isinstance(result, str):
            return EditResult(error=result)
        new_text, occurrences = result
        self._put(self._staged_file(file_path), restore_endings(current, new_text))
        self.touched.add(file_path)
        return EditResult(path=file_path, occurrences=int(occurrences))

    def delete(self, file_path: str) -> DeleteResult:
        try:
            exists_in_project = self._project_file(file_path).is_file()
        except ValueError as exc:
            return DeleteResult(error=str(exc))
        if self._project_file(file_path).is_dir():
            return DeleteResult(error=f"Error: '{file_path}' is a directory; delete its files")
        if not exists_in_project and file_path not in self.touched:
            return DeleteResult(error=f"Error: '{file_path}' not found")
        self.touched.discard(file_path)
        self._staged_file(file_path).unlink(missing_ok=True)
        if exists_in_project:
            self.deleted.add(file_path)
        return DeleteResult(path=file_path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]

    # --- the batch ------------------------------------------------------------

    def changes(self) -> list[Change]:
        batch = []
        for path in sorted(self.touched | self.deleted):
            before = _read_raw(self._project_file(path))
            after = None if path in self.deleted else _read_raw(self._staged_file(path))
            if before != after:
                batch.append(Change(path, before, after))
        return batch

    # --- helpers --------------------------------------------------------------

    def _current_text(self, file_path: str) -> str | None:
        if file_path in self.deleted:
            return None
        try:
            source = (
                self._staged_file(file_path)
                if file_path in self.touched
                else self._project_file(file_path)
            )
        except ValueError:
            return None
        return _read_raw(source)

    def _ending_for(self, file_path: str) -> str:
        """The line ending for a new file: a neighbour's with the same extension, else LF."""
        folder = self._project_file(file_path).parent
        if folder.is_dir():
            for sibling in sorted(folder.iterdir()):
                if sibling.is_file() and sibling.suffix == Path(file_path).suffix:
                    text = _read_raw(sibling)
                    if text is not None and "\n" in text:
                        return _ending_of(text)
        return "\n"

    @staticmethod
    def _put(target: Path, text: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as f:
            f.write(text)


def apply_changes(project: str | Path, batch: list[Change]) -> tuple[list[str], list[str]]:
    """Write an approved batch to the project.

    A file is only written if it still holds the ``before`` text the user
    approved against; otherwise it is reported as a conflict and left alone.
    Re-applying a batch is safe: files already at ``after`` count as applied.
    Returns (applied paths, conflict paths).
    """
    root = Path(project).resolve()
    applied, conflicts = [], []
    for change in batch:
        target = root / change.path.lstrip("/")
        current = _read_raw(target)
        if current == change.after:
            applied.append(change.path)
        elif current != change.before:
            conflicts.append(change.path)
        elif change.after is None:
            target.unlink()
            applied.append(change.path)
        else:
            StagedBackend._put(target, change.after)
            applied.append(change.path)
    return applied, conflicts


def _read_raw(path: Path) -> str | None:
    """File text with its line endings untouched; None if it doesn't exist."""
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def restore_endings(original: str, new_lf: str) -> str:
    """Give ``new_lf`` (LF-only text) the original file's line endings, line by line.

    Files can mix CRLF and LF. Lines the change didn't touch keep their exact
    original ending; a changed line takes the ending of the line it replaced;
    an inserted line takes the ending of the line before it. Only where there
    is no such line does the file's majority ending apply.
    """
    old_lines = original.splitlines(keepends=True)
    old_text = [line.rstrip("\r\n") for line in old_lines]
    old_end = [line[len(text) :] for line, text in zip(old_lines, old_text, strict=True)]
    new_lines = new_lf.splitlines(keepends=True)
    new_text = [line.rstrip("\n") for line in new_lines]
    has_end = [line.endswith("\n") for line in new_lines]
    majority = _ending_of(original)

    def ending_at(i: int) -> str:
        return old_end[i] if 0 <= i < len(old_end) and old_end[i] else majority

    out = []
    matcher = difflib.SequenceMatcher(a=old_text, b=new_text, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for k, j in enumerate(range(j1, j2)):
            if tag == "equal":
                source = i1 + k
            elif tag == "replace":
                source = min(i1 + k, i2 - 1)
            else:  # insert: follow the line before
                source = i1 - 1 if i1 > 0 else i1
            out.append(new_text[j] + (ending_at(source) if has_end[j] else ""))
    return "".join(out)


def _ending_of(text: str) -> str:
    """The file's majority line ending (LF on a tie or with no line breaks)."""
    crlf = text.count("\r\n")
    return "\r\n" if crlf > text.count("\n") - crlf else "\n"


def _to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _with_ending(text: str, ending: str) -> str:
    return text if ending == "\n" else text.replace("\n", ending)
