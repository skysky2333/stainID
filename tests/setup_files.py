"""Small Word and Excel files for the guided-setup tests (built in memory, no office software needed)."""
from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

DOCUMENT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')


def _paragraph(text: str, split_runs: bool = False) -> str:
    if split_runs and len(text) > 2:  # Word often cuts a word into several runs
        middle = len(text) // 2
        runs = [text[:middle], text[middle:]]
    else:
        runs = [text]
    return "<w:p>" + "".join(f'<w:r><w:t xml:space="preserve">{escape(r)}</w:t></w:r>' for r in runs) + "</w:p>"


def docx(tables: list[tuple[str, list[list[str]]]]) -> bytes:
    """A .docx with a heading paragraph and a table for each (heading, rows); a cell's lines become paragraphs."""
    body = ""
    for heading, rows in tables:
        body += _paragraph(heading) if heading else ""
        body += "<w:tbl>" + "".join("<w:tr>" + "".join("<w:tc>" + "".join(_paragraph(line, True) for line in (cell.split("\n") or [""]))
                                                        + "</w:tc>" for cell in row) + "</w:tr>" for row in rows) + "</w:tbl>"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", DOCUMENT.format(body=body))
    return buffer.getvalue()


def xlsx(sheets: dict[str, list[list[object]]]) -> bytes:
    from openpyxl import Workbook

    book = Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()
