"""
Task 1 — Thu thập văn bản chính sách/quy định về thư viện RMIT.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Các tài liệu được chọn:
    - Quy định và hướng dẫn đặt phòng học tại thư viện RMIT Vietnam.
    - Quy định thẻ cựu sinh viên và quyền truy cập thư viện RMIT Vietnam.
    - Hướng dẫn bản quyền khi sử dụng sách/eBook của thư viện RMIT.

Lưu ý: một số trang trường (vd VinUni, Fulbright) chặn bot crawler mặc định (HTTP 403) —
không phải lỗi của bạn, đó là cấu hình WAF/Cloudflare phía server. Đổi sang trang khác
thay vì cố vượt qua, và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

DOCUMENTS = [
    {
        "url": (
            "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/"
            "documents/pdfs/library/en/study-room-booking-instruction-2025.pdf"
        ),
        "filename": "study-room-booking-instruction-2025.pdf",
    },
    {
        "url": (
            "https://www.rmit.edu.vn/content/dam/rmit/vn/en/"
            "assets-for-production/documents/pdfs/alumni/"
            "alumni-card-policy-june-2020.pdf"
        ),
        "filename": "alumni-card-policy-june-2020.pdf",
    },
    {
        "url": (
            "https://www.rmit.edu.au/content/dam/rmit/documents/"
            "library/copyright/using-chapters-ebooks-students.pdf"
        ),
        "filename": "using-chapters-ebooks-students.pdf",
    },
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RMIT-Library-Document-Collector/1.0)"
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Data directory ready: {DATA_DIR}")


def is_valid_pdf(filepath: Path) -> bool:
    """Kiểm tra nhanh file có chữ ký PDF và dung lượng lớn hơn 1 KB."""
    if not filepath.is_file() or filepath.stat().st_size <= 1024:
        return False

    with filepath.open("rb") as file:
        return file.read(4) == b"%PDF"


def download_file(url: str, filename: str) -> Path:
    """Tải một PDF công khai của RMIT và lưu vào thư mục landing/legal."""
    filepath = DATA_DIR / filename

    if is_valid_pdf(filepath):
        print(f"[SKIP] Valid PDF already exists: {filepath}")
        return filepath

    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()

    content = response.content
    if len(content) <= 1024 or not content.startswith(b"%PDF"):
        content_type = response.headers.get("Content-Type", "không xác định")
        raise ValueError(
            f"URL không trả về PDF hợp lệ: {url} "
            f"(Content-Type: {content_type}, size: {len(content)} bytes)"
        )

    filepath.write_bytes(content)
    print(f"[OK] Downloaded: {filepath} ({len(content):,} bytes)")
    return filepath


def collect_documents() -> list[Path]:
    """Thu thập toàn bộ tài liệu và trả về danh sách file đã có trên máy."""
    setup_directory()
    collected_files = []

    for document in DOCUMENTS:
        try:
            filepath = download_file(document["url"], document["filename"])
            collected_files.append(filepath)
        except (requests.RequestException, ValueError, OSError) as error:
            print(f"[ERROR] Could not download {document['filename']}: {error}")

    return collected_files


if __name__ == "__main__":
    files = collect_documents()
    print(f"\nCompleted: {len(files)}/{len(DOCUMENTS)} valid documents.")
    if len(files) != len(DOCUMENTS):
        raise SystemExit(1)
