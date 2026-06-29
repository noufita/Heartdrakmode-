"""
3D anatomical beating heart for CardioPulse (dark mode).
Returns an HTML/Three.js snippet to embed with st.components.v1.html.
The heart beats faster and glows green -> amber -> red as `prob` rises.
"""

def heart_html(prob: float, height: int = 320) -> str:
    return f"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html,body{{margin:0;background:transparent;overflow:hidden;}}
  #wrap{{position:relative;width:100%;height:{height}px;}}
  #glow{{position:absolute;inset:18%;border-radius:50%;filter:blur(34px);opacity:.6;}}
  #pct{{position:absolute;bottom:8px;left:0;right:0;text-align:center;
        font:900 44px Nunito,system-ui,sans-serif;color:#fff;
        text-shadow:0 2px 14px rgba(0,0,0,.6);pointer-events:none;}}
  #pct small{{font-size:20px;}}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body>
<div id="wrap"><div id="glow"></div><div id="canvas"></div>
  <div id="pct">{round(prob*100)}<small>%</small></div></div>
<script>
const PROB = {prob};
const node = document.getElementById('canvas');
const W = node.clientWidth || 320, H = {height};
const lo = new THREE.Color(0x35d07a), mid = new THREE.Color(0xe6a83a), hi = new THREE.Color(0xff3b40);
const tint = PROB < 0.35 ? lo : PROB < 0.65 ? mid : hi;
document.getElementById('glow').style.background =
  'radial-gradient(circle, #'+tint.getHexString()+', transparent 68%)';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 1000);
camera.position.z = 66;
const renderer = new THREE.WebGLRenderer({{antialias:true, alpha:true}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(W, H);
node.appendChild(renderer.domElement);

const BASE = 1.05;
const profile = [[0.3,-13.5],[2.4,-11.2],[4.7,-8],[6.4,-4.2],[7.3,-0.2],[7.5,3.2],[6.9,6.2],[5.5,8.6],[3.5,10],[1.6,10.7],[0.3,10.9]]
  .map(p => new THREE.Vector2(p[0], p[1]));
const bodyGeo = new THREE.LatheGeometry(profile, 90);
const pos = bodyGeo.attributes.position, vec = new THREE.Vector3();
const noise = (x,y,z)=> Math.sin(x*0.55+y*0.4)*0.6 + Math.sin(y*0.7+z*0.5)*0.45 + Math.sin(z*0.85+x*0.6)*0.35 + Math.sin(x*1.3+z*1.1)*0.18;
for (let i=0;i<pos.count;i++){{
  vec.fromBufferAttribute(pos,i);
  const nrm = vec.clone().normalize();
  let d = noise(vec.x, vec.y, vec.z) * 0.5;
  const groove = 1.6 - 0.16*vec.y;
  if (vec.z > 0){{ const gd = vec.x - groove; d -= Math.exp(-(gd*gd)/5.0) * Math.max(0,vec.z/8) * 2.6; }}
  if (vec.z < 0){{ const gd = vec.x + 1.2; d -= Math.exp(-(gd*gd)/7.0) * Math.max(0,-vec.z/9) * 1.2; }}
  vec.addScaledVector(nrm, d);
  pos.setXYZ(i, vec.x, vec.y, vec.z);
}}
bodyGeo.computeVertexNormals();
const mat = new THREE.MeshStandardMaterial({{color:0x911d1d, roughness:0.5, metalness:0.06, emissive:new THREE.Color(0x5e0f0f), emissiveIntensity:0.5}});
const body = new THREE.Mesh(bodyGeo, mat);
const group = new THREE.Group(); group.add(body);

function tube(pts, r, col, emi){{
  const curve = new THREE.CatmullRomCurve3(pts.map(p=>new THREE.Vector3(p[0],p[1],p[2])));
  const g = new THREE.TubeGeometry(curve, 48, r, 18, false);
  const m = new THREE.MeshStandardMaterial({{color:col, roughness:0.55, metalness:0.05, emissive:new THREE.Color(emi||0x140404), emissiveIntensity:0.22}});
  return new THREE.Mesh(g, m);
}}
group.add(tube([[1,9,0.6],[1.6,12.6,0],[0,16.2,-2.6],[-4.6,16.6,-3.6],[-6.8,12.4,-3.6]], 1.95, 0xc25a44));
group.add(tube([[-2.4,8.6,2.1],[-3.1,12,1.6],[-1.4,15.6,0.6],[1.2,17.2,-0.6]], 1.7, 0xb44a58));
group.add(tube([[4.3,8,-1.5],[4.7,12,-1.6],[4.8,15.6,-1.7]], 1.45, 0x9c5a6e));
group.add(tube([[1.7,7.6,7.1],[1.9,3,7.7],[1.4,-1.6,7.3],[0.9,-6.2,5.7],[0.5,-9.8,3.4]], 0.5, 0xd23a40, 0x340606));
group.add(tube([[1.7,7.6,7.1],[5.6,6.6,4.6],[7.5,3,0.6],[6.6,-0.6,-3.6]], 0.5, 0xd23a40, 0x340606));
group.position.y = -1.2; group.scale.set(BASE,BASE,BASE);
scene.add(group);

scene.add(new THREE.AmbientLight(0x4a2626, 1.05));
const key = new THREE.PointLight(0xffd8c8, 1.3); key.position.set(36,40,52); scene.add(key);
const rim = new THREE.PointLight(0xff5a4a, 0.95); rim.position.set(-44,-10,26); scene.add(rim);
const top = new THREE.DirectionalLight(0xffffff, 0.4); top.position.set(0,45,18); scene.add(top);
mat.emissive.copy(tint).multiplyScalar(0.6);

const clock = new THREE.Clock();
(function loop(){{
  const t = clock.getElapsedTime();
  const period = 1.2 - 0.5*PROB;
  const ph = (t % period)/period;
  const beat = Math.exp(-Math.pow((ph-0.12)/0.045,2)) + 0.55*Math.exp(-Math.pow((ph-0.34)/0.07,2));
  const sc = BASE*(1+0.06*beat);
  group.scale.set(sc, sc*(1-0.025*beat), sc);
  mat.emissiveIntensity = 0.42 + 0.45*beat;
  group.rotation.y += 0.005;
  group.rotation.z = 0.12 + Math.sin(t*0.5)*0.03;
  group.rotation.x = 0.08;
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}})();
</script></body></html>
"""
