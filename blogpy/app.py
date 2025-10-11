# ==================== app.py ====================
# Flask là bếp nấu web; render_template bật HTML; request đọc dữ liệu form
from flask import Flask, render_template, request, redirect, url_for, flash
import re
from collections import Counter

# ----- Mini AI #1: Tạo tóm tắt 1 câu ngắn -----
def make_summary(text: str, max_len: int = 140) -> str:
    # Giải thích siêu dễ:
    # - Lấy câu đầu tiên (tính tới dấu . ! ?)
    # - Nếu quá dài, cắt ngắn + thêm "..."
    first_sentence = re.split(r'(?<=[.!?])\s+', text.strip())
    s = first_sentence[0] if first_sentence else text.strip()
    if len(s) > max_len:
        return s[:max_len].rstrip() + "..."
    return s

# ----- Mini AI #2: Trích tags đơn giản từ nội dung -----
def extract_tags(title: str, content: str, top_k: int = 5):
    # Giải thích siêu dễ:
    # - Tách từ (chỉ giữ chữ, số), chuyển thường.
    # - Bỏ từ vô nghĩa (the, and, là, của,...)
    # - Đếm tần suất → lấy top phổ biến.
    stop = {
        "the","and","a","an","to","of","in","on","for","with","is","are","was","were","it",
        "i","you","we","they","he","she","that","this","these","those",
        "và","là","của","cho","với","trong","một","những","các","đã","đang","sẽ","tôi","bạn"
    }
    text = f"{title} {content}".lower()
    words = re.findall(r"[a-zA-Z0-9]+", text)
    words = [w for w in words if w not in stop and len(w) >= 3]
    # Ưu tiên giữ uniqueness theo thứ tự xuất hiện
    counts = Counter(words)
    # Lấy top theo tần suất, rồi giữ thứ tự ban đầu
    top = [w for w,_ in counts.most_common(50)]
    seen, tags = set(), []
    for w in words:
        if w in top and w not in seen:
            seen.add(w); tags.append(w)
        if len(tags) >= top_k:
            break
    return tags



app = Flask(__name__)
app.secret_key = "dev-secret"  # khóa nhỏ để dùng flash message (thông báo)

# Bộ nhớ tạm trong RAM (mất khi tắt app)
posts = []   # nơi chứa các bài viết
next_id = 1  # id tự tăng cho mỗi bài

@app.route("/")
def home():
    # Hiện tất cả bài viết ra trang chủ
    return render_template("index.html", posts=posts)

@app.route("/posts/new")
def new_post():
    # Trang có form để tạo bài viết
    return render_template("new.html")

@app.route("/posts", methods=["POST"])
def create_post():
    global next_id
    # Lấy dữ liệu người dùng gõ từ form
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()

    # Kiểm tra đơn giản: không để trống
    if not title or not content:
        flash("Please fill in both Title and Content.")
        return redirect(url_for("new_post"))

    # Tạo đối tượng bài viết (id + title + content)
    summary = make_summary(content)
    tags = extract_tags(title, content)

    post = {
        "id": next_id,
        "title": title,
        "content": content,
        "summary": summary,  # <- thêm field
        "tags": tags  # <- thêm field
    }
    posts.append(post)      # thêm vào danh sách
    next_id += 1            # tăng id cho lần sau

    # Sau khi tạo xong → quay về trang chủ để xem
    return redirect(url_for("home"))

# ==================== thêm vào app.py sau create_post ====================

@app.route("/posts/<int:post_id>/edit")
def edit_post(post_id):
    # Tìm bài có id = post_id trong danh sách
    post = next((p for p in posts if p["id"] == post_id), None)
    if not post:
        # Không thấy bài → quay về trang chủ
        flash("Post not found.")
        return redirect(url_for("home"))
    # Hiển thị form edit, truyền dữ liệu bài vào template
    return render_template("edit.html", post=post)

@app.route("/posts/<int:post_id>/update", methods=["POST"])
def update_post(post_id):
    # Lấy dữ liệu mới từ form
    new_title = request.form.get("title", "").strip()
    new_content = request.form.get("content", "").strip()

    # Kiểm tra rỗng
    if not new_title or not new_content:
        flash("Please fill in both Title and Content.")
        return redirect(url_for("edit_post", post_id=post_id))

    # Tìm và cập nhật bài
    for p in posts:
        if p["id"] == post_id:
            p["title"] = new_title
            p["content"] = new_content
            p["summary"] = make_summary(new_content)  # <- cập nhật
            p["tags"] = extract_tags(new_title, new_content)  # <- cập nhật

            break

    # Về trang chủ
    return redirect(url_for("home"))

@app.route("/posts/<int:post_id>/delete", methods=["POST"])
def delete_post(post_id):
    # Xoá bài có id = post_id khỏi danh sách
    # dùng [:] để chỉnh "in-place" (giữ cùng object list)
    posts[:] = [p for p in posts if p["id"] != post_id]
    # Về trang chủ
    return redirect(url_for("home"))
# ==================== end new routes ====================

if __name__ == "__main__":
    app.run(debug=True)
# ==================== end app.py ====================
