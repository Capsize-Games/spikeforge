// Shared GLSL primitives. All scene detail is procedural; no image textures.
export const vertex = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const common = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform vec2 uResolution;
uniform vec2 uPointer;
uniform float uTime;
uniform float uEncoding;
const float PI = 3.14159265359;
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),
    mix(hash(i+vec2(0,1)),hash(i+1.0),f.x),f.y);
}
float fbm(vec2 p) {
  float sum = 0.0, amp = .5;
  mat2 turn = mat2(.8,-.6,.6,.8);
  for(int i=0;i<6;i++) {
    sum += noise(p)*amp;
    p = turn*p*2.03+vec2(5.7,3.2);
    amp *= .5;
  }
  return sum;
}
float segment(vec2 p, vec2 a, vec2 b) {
  vec2 d = b-a;
  return length(p-a-d*clamp(dot(p-a,d)/dot(d,d),0.0,1.0));
}
// A white-hot filament core, colored near glow, and broad atmospheric scatter.
vec3 emission(float d, vec3 tint, float strength) {
  float aa = .65/uResolution.y;
  return strength*(mix(tint,vec3(1.0,.85,.67),.58)*exp(-d*d/(aa*aa))*3.0
    +tint*(exp(-d*650.0)*3.8+exp(-d*115.0)*.8+exp(-d*29.0)*.1));
}
vec3 starLayer(vec2 p, float scale, float seed) {
  vec2 cell = p*scale;
  vec2 id = floor(cell), f = fract(cell)-.5;
  float h = hash(id+seed);
  vec2 pos = vec2(hash(id+1.2),hash(id+8.7))*.65-.325;
  float d = length(f-pos);
  float aa = max(.004, scale/uResolution.y*.38);
  float small = exp(-d*d/(aa*aa)) * step(.975,h);
  float bright = step(.998,h);
  float core = exp(-d*48.0)*bright*1.6;
  float halo = .009/(d*d+.005)*bright;
  float rays = exp(-abs(f.x-pos.x)*180.0)*exp(-abs(f.y-pos.y)*15.0);
  rays += exp(-abs(f.y-pos.y)*180.0)*exp(-abs(f.x-pos.x)*15.0);
  float twinkle = .9+.1*sin(uTime*.22+h*50.0);
  return mix(vec3(.19,.35,1.0),vec3(.85,.93,1.0),hash(id+22.0))*
    (small*.95+core+halo*.11+rays*bright*.28)*twinkle;
}
vec3 nebulaField(vec2 p, float richness) {
  vec2 q = p*3.0+vec2(4.2,7.4)+uPointer*.008;
  q += vec2(uTime*.0032,-uTime*.0017);
  vec2 warp = vec2(fbm(q),fbm(q+vec2(12.3,2.1)));
  float field = fbm(q*1.7+warp*3.1);
  float dust = fbm(q*4.0+warp*4.0);
  float tendril = pow(max(0.0,1.0-abs(field-.52)*4.8),6.0);
  float clouds = pow(max(0.0,field-.30)*1.75,3.2)*richness;
  float fine = fbm(q*14.0+warp*3.0);
  float knots = pow(max(0.0,fine-.38)*3.0,2.0);
  vec3 color = vec3(.0007,.0012,.004);
  color += vec3(.006,.035,.28)*clouds*(.15+knots*2.8);
  color += vec3(.025,.13,.75)*tendril*pow(dust,5.0)*richness*(.12+knots)*.55;
  float rose = smoothstep(.48,.78,fbm(q*.6+vec2(4,12)));
  color += vec3(.48,.004,.20)*clouds*rose*knots;
  color *= .5+.5*smoothstep(.18,.62,dust);
  return color;
}
vec3 cosmos(vec2 p, float richness) {
  vec3 color=nebulaField(p,richness);
  vec2 drift=vec2(uTime*.00065,uTime*.00022);
  color += starLayer(p+drift,105.0,7.0)+starLayer(p+6.0+drift*1.5,42.0,3.0)+starLayer(p+3.2+drift*2.0,21.0,12.0)*.8;
  return color;
}
void outputColor(vec3 color) {
  fragColor = vec4(max(color,0.0)/uEncoding,1.0);
}
`;

export const bloom = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uTexture;
uniform vec2 uDirection;
uniform float uThreshold;
uniform float uEncoding;
vec3 sampleLight(vec2 p) {
  vec3 c = texture(uTexture,p).rgb;
  return max(c-vec3(uThreshold/uEncoding),0.0);
}
void main() {
  vec3 c = sampleLight(vUv)*.227027;
  c += sampleLight(vUv+uDirection*1.384615)*.316216;
  c += sampleLight(vUv-uDirection*1.384615)*.316216;
  c += sampleLight(vUv+uDirection*3.230769)*.070270;
  c += sampleLight(vUv-uDirection*3.230769)*.070270;
  fragColor = vec4(c,1.0);
}`;

export const composite = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uTexture;
uniform sampler2D uBloom;
uniform sampler2D uWideBloom;
uniform float uEncoding;
void main() {
  vec3 c = (texture(uTexture,vUv).rgb+texture(uBloom,vUv).rgb*.65+
    texture(uWideBloom,vUv).rgb*.75)*uEncoding;
  // ACES filmic response preserves bright cores without clipping colored bloom.
  c = clamp((c*(2.51*c+.03))/(c*(2.43*c+.59)+.14),0.0,1.0);
  c = pow(c,vec3(1.0/2.2));
  float grain = fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453);
  fragColor = vec4(c+(grain-.5)/255.0,1.0);
}`;
