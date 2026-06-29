"""
Animated beating heart for CardioPulse (dark mode).
Pure CSS/SVG — no external dependencies, works on Streamlit Cloud.
Color and beat speed driven by prob.
"""

def heart_html(prob: float, height: int = 320) -> str:
    pct = round(prob * 100)
    
    if prob < 0.35:
        color = "#35d07a"
        glow = "rgba(53,208,122,0.5)"
        label_color = "#35d07a"
    elif prob < 0.65:
        color = "#e6a83a"
        glow = "rgba(230,168,58,0.5)"
        label_color = "#e6a83a"
    else:
        color = "#ff3b40"
        glow = "rgba(255,59,64,0.5)"
        label_color = "#ff3b40"

    # beat speed: fast when high risk
    duration = round(1.4 - 0.7 * prob, 2)

    return f"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html, body {{
    margin: 0; padding: 0;
    background: transparent;
    display: flex; align-items: center; justify-content: center;
    height: {height}px; overflow: hidden;
  }}
  .wrap {{
    position: relative;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: {height}px;
  }}
  .heart-outer {{
    position: relative;
    display: flex; align-items: center; justify-content: center;
  }}
  .glow {{
    position: absolute;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, {glow}, transparent 70%);
    animation: glow-pulse {duration}s ease-in-out infinite;
  }}
  @keyframes glow-pulse {{
    0%   {{ transform: scale(1);   opacity: 0.6; }}
    12%  {{ transform: scale(1.2); opacity: 0.9; }}
    24%  {{ transform: scale(1);   opacity: 0.6; }}
    36%  {{ transform: scale(1.1); opacity: 0.8; }}
    54%  {{ transform: scale(1);   opacity: 0.6; }}
    100% {{ transform: scale(1);   opacity: 0.6; }}
  }}
  .heart-svg {{
    animation: heartbeat {duration}s ease-in-out infinite;
    filter: drop-shadow(0 0 18px {glow});
    z-index: 1;
  }}
  @keyframes heartbeat {{
    0%   {{ transform: scale(1); }}
    12%  {{ transform: scale(1.18); }}
    24%  {{ transform: scale(1); }}
    36%  {{ transform: scale(1.12); }}
    54%  {{ transform: scale(1); }}
    100% {{ transform: scale(1); }}
  }}
  .pct {{
    margin-top: 14px;
    font: 900 52px 'Nunito', system-ui, sans-serif;
    color: #fff;
    text-shadow: 0 2px 14px rgba(0,0,0,.6);
    line-height: 1;
  }}
  .pct small {{ font-size: 22px; }}
</style>
</head><body>
<div class="wrap">
  <div class="heart-outer">
    <div class="glow"></div>
    <svg class="heart-svg" viewBox="0 0 100 90" width="160" height="144" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="hg" cx="50%" cy="40%" r="60%">
          <stop offset="0%" stop-color="{color}" stop-opacity="1"/>
          <stop offset="100%" stop-color="{color}" stop-opacity="0.6"/>
        </radialGradient>
        <filter id="inner-shadow">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feComposite in="SourceGraphic" in2="blur" operator="over"/>
        </filter>
      </defs>
      <!-- Main heart shape -->
      <path d="M50 85 C50 85 5 55 5 28 C5 14 15 5 27 5 C35 5 42 9 50 18 C58 9 65 5 73 5 C85 5 95 14 95 28 C95 55 50 85 50 85Z"
            fill="url(#hg)" stroke="{color}" stroke-width="1.5" opacity="0.95"/>
      <!-- Highlight -->
      <ellipse cx="35" cy="22" rx="10" ry="7" fill="white" opacity="0.18" transform="rotate(-20 35 22)"/>
    </svg>
  </div>
  <div class="pct" style="color:{label_color}">{pct}<small>%</small></div>
</div>
</body></html>
"""
