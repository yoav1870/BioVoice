import shutil
import base64
from pathlib import Path
import torchaudio

def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".mp3":
        return "audio/mpeg"
    return "application/octet-stream"

def save_spectrogram_player_html(
    audio_path: str,
    spectrogram_png_path: str,
    out_html_path: str | None = None,
    *,
    total_time_sec: float | None = None,
    copy_audio: bool = True,
    embed_image: bool = True,
):
    """
    Creates an HTML file with:
      - <audio controls>
      - spectrogram as a canvas timeline
      - moving playhead synced to audio
      - click-to-seek

    total_time_sec:
      If None, uses audio duration from torchaudio.info().
      If you want "spectrogram time" instead, pass (T-1)*hop_length/sr.
    """
    audio_p = Path(audio_path)
    img_p = Path(spectrogram_png_path)

    assert audio_p.exists(), f"Missing audio: {audio_p}"
    assert img_p.exists(), f"Missing image: {img_p}"

    out_html = Path(out_html_path) if out_html_path else img_p.with_suffix(".html")
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Duration (seconds)
    if total_time_sec is None:
        info = torchaudio.info(str(audio_p))
        total_time_sec = float(info.num_frames) / float(info.sample_rate)

    # Put audio next to the HTML so opening locally works
    audio_src = None
    if copy_audio:
        copied_audio = out_html.parent / audio_p.name
        if copied_audio.resolve() != audio_p.resolve():
            shutil.copy2(audio_p, copied_audio)
        audio_src = copied_audio.name  # relative path
    else:
        # relative path if possible
        try:
            audio_src = str(audio_p.relative_to(out_html.parent))
        except ValueError:
            audio_src = str(audio_p)

    # Embed image as data URL (so HTML is self-contained w.r.t the PNG)
    if embed_image:
        img_bytes = img_p.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        img_src = f"data:{_guess_mime(img_p)};base64,{img_b64}"
    else:
        try:
            img_src = str(img_p.relative_to(out_html.parent))
        except ValueError:
            img_src = str(img_p)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Spectrogram Player</title>
<style>
  body {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    padding: 16px;
    max-width: 1100px;
    margin: 0 auto;
  }}
  .row {{
    display: flex;
    gap: 16px;
    flex-direction: column;
  }}
  canvas {{
    width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 8px;
  }}
  .hint {{
    color: #444;
    font-size: 14px;
  }}
</style>
</head>
<body>
  <div class="row">
  <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
    <audio id="aud" controls preload="metadata" src="{audio_src}"></audio>

    <label style="display:flex; gap:8px; align-items:center; font-size:14px; color:#333;">
      Speed
      <select id="rate" style="padding:4px 6px; border:1px solid #ddd; border-radius:6px;">
        <option value="0.25">0.25×</option>
        <option value="0.5">0.5×</option>
        <option value="0.75">0.75×</option>
        <option value="1" selected>1×</option>
      </select>
    </label>
  </div>

  <div class="hint">Click the spectrogram to seek. The vertical line follows playback.</div>
  <canvas id="cv"></canvas>
  </div>


<script>
(() => {{
  const TOTAL = {total_time_sec:.10f}; // seconds
  const aud = document.getElementById("aud");
  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const img = new Image();
  img.src = "{img_src}";
  const rateSel = document.getElementById("rate");
  aud.playbackRate = parseFloat(rateSel.value);

  rateSel.addEventListener("change", () => {{
    aud.playbackRate = parseFloat(rateSel.value);
  }});

  let naturalW = 0, naturalH = 0;

  function resizeCanvasToImage() {{
    if (!naturalW || !naturalH) return;
    cv.width = naturalW;
    cv.height = naturalH;
    draw();
  }}

  function draw() {{
    if (!naturalW || !naturalH) return;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(img, 0, 0);

    const t = Math.max(0, Math.min(aud.currentTime || 0, TOTAL));
    const x = (t / TOTAL) * cv.width;

    // playhead
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cv.height);
    ctx.stroke();

    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.beginPath();
    ctx.moveTo(x+1, 0);
    ctx.lineTo(x+1, cv.height);
    ctx.stroke();
    ctx.restore();
  }}

  function rafLoop() {{
    draw();
    requestAnimationFrame(rafLoop);
  }}

  img.onload = () => {{
    naturalW = img.naturalWidth;
    naturalH = img.naturalHeight;
    resizeCanvasToImage();
  }};

  // click-to-seek
  cv.addEventListener("click", (e) => {{
    const rect = cv.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width; // 0..1
    aud.currentTime = x * TOTAL;
  }});

  // keep updating
  requestAnimationFrame(rafLoop);
}})();
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    return out_html
