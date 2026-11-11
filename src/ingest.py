"""
src/ingest.py - Ingest arXiv PDFs, extract text with page/section tracking, and diagnose artifacts.

Papers:
  1. Attention Is All You Need: arXiv 1706.03762 v7
  2. Moshi: arXiv 2410.00037 v1
  3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling: arXiv 2507.07955 v2
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pymupdf

# Exact paper catalog specification
PAPERS_METADATA = {
    "attention": {
        "key": "attention",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762v7",
        "pdf_url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "1706.03762v7.pdf",
        "running_header": None,
    },
    "moshi": {
        "key": "moshi",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037v1",
        "pdf_url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "2410.00037v1.pdf",
        "running_header": "Moshi: a speech-text foundation model for real-time dialogue",
    },
    "dynamic_chunking": {
        "key": "dynamic_chunking",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955v2",
        "pdf_url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "2507.07955v2.pdf",
        "running_header": None,
    },
}


@dataclass
class PageRecord:
    paper_key: str
    paper_title: str
    arxiv_id: str
    page_number: int  # 1-indexed
    active_section: str
    sections_found: List[str]
    clean_text: str
    raw_text: str
    char_count: int
    word_count: int


def download_papers(
    dest_dir: Path, force: bool = False
) -> Dict[str, Path]:
    """Download the 3 target arXiv PDFs at their exact specified versions."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    headers = {
        "User-Agent": "RelayResearchPipeline/1.0 (academic research; take-home assignment)"
    }

    for key, meta in PAPERS_METADATA.items():
        pdf_path = dest_dir / meta["filename"]
        paths[key] = pdf_path

        if pdf_path.exists() and pdf_path.stat().st_size > 50000 and not force:
            print(f"[{key}] Already downloaded: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
            continue

        print(f"[{key}] Downloading {meta['arxiv_id']} from {meta['pdf_url']}...")
        with httpx.Client(follow_redirects=True, timeout=90.0, headers=headers) as client:
            resp = client.get(meta["pdf_url"])
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
            print(f"[{key}] Successfully saved to {pdf_path} ({len(resp.content):,} bytes)")

    return paths


def detect_section_headings(
    blocks: List[Any],
) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """
    Detect section headings from PyMuPDF block elements.
    Handles:
      - Numbered multi-line headings: '1\\nIntroduction' or '3.2.1\\nScaled Dot-Product Attention'
      - Numbered single-line headings: '1 Introduction'
      - Structural section names: 'Abstract', 'References', 'Conclusion', 'Acknowledgments'
    Filters out technical metrics like '12.5 Hz' or '1.0 FLOPs'.
    """
    sec_multiline = re.compile(
        r"^(\d+(?:\.\d+)*)\s*\n\s*([A-Z][A-Za-z0-9\s,\-\(\):]{2,70})$"
    )
    sec_singleline = re.compile(
        r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9\s,\-\(\):]{2,70})$"
    )
    structural_pat = re.compile(
        r"^(Abstract|References|Conclusions?|Acknowledgements?|Appendix(?:\s+[A-Z0-9]+)?)$",
        re.IGNORECASE,
    )
    unit_filter = re.compile(r"\b(Hz|kHz|MHz|dB|FLOPs|tokens|sec|ms|bps|kbps)\b")

    detected = []
    for b in blocks:
        # b is (x0, y0, x1, y1, text, block_no, block_type)
        if b[6] != 0:  # skip non-text blocks (images)
            continue
        text = b[4].strip()
        bbox = (b[0], b[1], b[2], b[3])

        # Check structural title
        if structural_pat.match(text):
            detected.append((text, bbox))
            continue

        # Check multi-line block pattern (common in LaTeX / NeurIPS)
        m_multi = sec_multiline.match(text)
        if m_multi:
            num, title = m_multi.group(1), m_multi.group(2).strip()
            if not unit_filter.search(title):
                detected.append((f"{num} {title}", bbox))
                continue

        # Check line by line in block
        for line in text.split("\n"):
            line = line.strip()
            m_single = sec_singleline.match(line)
            if m_single:
                num, title = m_single.group(1), m_single.group(2).strip()
                if not unit_filter.search(title):
                    detected.append((f"{num} {title}", bbox))

    return detected


def clean_page_text(raw_text: str, running_header: Optional[str] = None) -> str:
    """
    Clean typical extraction noise from a page:
      - arXiv margin watermark lines
      - Repeated running headers
      - Standalone page numbers
      - Soft hyphen line wraps ('conversa-\\ntional' -> 'conversational')
    """
    lines = raw_text.split("\n")
    cleaned_lines = []
    arxiv_regex = re.compile(r"arxiv:\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
    page_num_regex = re.compile(r"^\d{1,3}$")

    for line in lines:
        stripped = line.strip()
        # Skip arXiv watermark lines
        if arxiv_regex.search(stripped):
            continue
        # Skip running header if matched
        if running_header and stripped == running_header.strip():
            continue
        # Skip lone page number line
        if page_num_regex.match(stripped):
            continue
        cleaned_lines.append(line)

    joined = "\n".join(cleaned_lines)
    # Fix hyphenation across line wraps: e.g. "conversa-\ntional" -> "conversational"
    # only when both sides are alphabetical words
    joined = re.sub(r"([A-Za-z]{2,})-\n([A-Za-z]{2,})", r"\1\2", joined)
    return joined.strip()


def diagnose_page_anomalies(
    raw_text: str, page_num: int, paper_key: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect extraction hazards that mislead naive retrievers:
      1. Mangled equations (isolated radical symbols, stacked fractions, broken formulas)
      2. Table linearizations (numbers decoupled from headers)
      3. Isolated vertical diagram glyphs / noise
    """
    anomalies: Dict[str, List[Dict[str, Any]]] = {
        "mangled_equations": [],
        "table_linearization": [],
        "diagram_artifacts": [],
    }

    # 1. Equation issues: look for isolated math square root or broken fractions
    if "√" in raw_text or "1\n√" in raw_text or "QKT\n" in raw_text:
        anomalies["mangled_equations"].append({
            "page": page_num,
            "sample": "Detected broken fraction or radical (e.g. QKT / √dk)",
        })

    # Look for single-character lines vertically stacked (typical diagram artifact as in Dynamic Chunking Fig 1)
    single_char_run = 0
    max_single_char_run = 0
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    for line in lines:
        if len(line) == 1:
            single_char_run += 1
            max_single_char_run = max(max_single_char_run, single_char_run)
        else:
            single_char_run = 0

    if max_single_char_run >= 4:
        anomalies["diagram_artifacts"].append({
            "page": page_num,
            "max_consecutive_single_chars": max_single_char_run,
            "sample": "Vertical glyph stacking from diagram/figure labels",
        })

    # 2. Table linearization check
    table_indicators = ["Table 1:", "Table 2:", "Table 3:", "Table 4:", "Table 5:"]
    for ind in table_indicators:
        if ind in raw_text:
            anomalies["table_linearization"].append({
                "page": page_num,
                "table_indicator": ind,
                "sample": f"Table detected on page {page_num}; columnar grid flattened into sequential lines",
            })

    return anomalies


def ingest_all(
    papers_dir: Path, output_json: Path, report_json: Path, force: bool = False
) -> Tuple[List[PageRecord], Dict[str, Any]]:
    """Download, extract, and analyze all 3 target papers."""
    pdf_paths = download_papers(papers_dir, force=force)
    all_records: List[PageRecord] = []
    full_report: Dict[str, Any] = {
        "papers": {},
        "summary": {
            "total_pages": 0,
            "total_characters": 0,
            "total_words": 0,
            "total_sections_detected": 0,
            "total_mangled_equations": 0,
            "total_table_linearizations": 0,
            "total_diagram_artifacts": 0,
        },
    }

    for key, pdf_path in pdf_paths.items():
        meta = PAPERS_METADATA[key]
        doc = pymupdf.open(pdf_path)
        paper_records: List[PageRecord] = []
        current_section = "Header / Frontmatter"
        paper_anomalies: Dict[str, List[Dict[str, Any]]] = {
            "mangled_equations": [],
            "table_linearization": [],
            "diagram_artifacts": [],
            "sections": [],
        }

        print(f"\n--- Extracting {meta['title']} ({len(doc)} pages) ---")

        for p_idx in range(len(doc)):
            page = doc[p_idx]
            page_num = p_idx + 1
            blocks = page.get_text("blocks")
            raw_text = page.get_text("text")

            # Detect section headings on this page
            headings = detect_section_headings(blocks)
            heading_titles = [h[0] for h in headings]

            # Update active section if headings found
            if heading_titles:
                current_section = heading_titles[0]
                for h in heading_titles:
                    paper_anomalies["sections"].append({
                        "page": page_num,
                        "section": h,
                    })

            # Clean page text
            cleaned = clean_page_text(raw_text, running_header=meta["running_header"])
            word_count = len(cleaned.split())
            char_count = len(cleaned)

            record = PageRecord(
                paper_key=key,
                paper_title=meta["title"],
                arxiv_id=meta["arxiv_id"],
                page_number=page_num,
                active_section=current_section,
                sections_found=heading_titles,
                clean_text=cleaned,
                raw_text=raw_text,
                char_count=char_count,
                word_count=word_count,
            )
            paper_records.append(record)
            all_records.append(record)

            # Check anomalies
            anoms = diagnose_page_anomalies(raw_text, page_num, key)
            paper_anomalies["mangled_equations"].extend(anoms["mangled_equations"])
            paper_anomalies["table_linearization"].extend(anoms["table_linearization"])
            paper_anomalies["diagram_artifacts"].extend(anoms["diagram_artifacts"])

        full_report["papers"][key] = {
            "title": meta["title"],
            "arxiv_id": meta["arxiv_id"],
            "page_count": len(doc),
            "total_words": sum(r.word_count for r in paper_records),
            "sections_count": len(paper_anomalies["sections"]),
            "sections_list": paper_anomalies["sections"],
            "anomalies": {
                "mangled_equations_count": len(paper_anomalies["mangled_equations"]),
                "mangled_equations": paper_anomalies["mangled_equations"],
                "table_linearization_count": len(paper_anomalies["table_linearization"]),
                "table_linearization": paper_anomalies["table_linearization"],
                "diagram_artifacts_count": len(paper_anomalies["diagram_artifacts"]),
                "diagram_artifacts": paper_anomalies["diagram_artifacts"],
            },
        }

        # Update totals
        full_report["summary"]["total_pages"] += len(doc)
        full_report["summary"]["total_characters"] += sum(r.char_count for r in paper_records)
        full_report["summary"]["total_words"] += sum(r.word_count for r in paper_records)
        full_report["summary"]["total_sections_detected"] += len(paper_anomalies["sections"])
        full_report["summary"]["total_mangled_equations"] += len(paper_anomalies["mangled_equations"])
        full_report["summary"]["total_table_linearizations"] += len(paper_anomalies["table_linearization"])
        full_report["summary"]["total_diagram_artifacts"] += len(paper_anomalies["diagram_artifacts"])

        print(
            f"  Extracted {len(doc)} pages, {len(paper_anomalies['sections'])} sections, "
            f"{sum(r.word_count for r in paper_records):,} words."
        )

    # Save outputs
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_records], f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_records)} page records to {output_json}")

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"Saved anomaly diagnosis report to {report_json}")

    return all_records, full_report


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers for The Relay")
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("data/papers"),
        help="Directory to store downloaded PDFs",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/papers/extracted_pages.json"),
        help="Output path for extracted page records",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("data/papers/extraction_report.json"),
        help="Output path for extraction anomaly report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    args = parser.parse_args()

    ingest_all(
        papers_dir=args.papers_dir,
        output_json=args.output_json,
        report_json=args.report_json,
        force=args.force,
    )


if __name__ == "__main__":
    main()

