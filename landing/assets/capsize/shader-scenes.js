import { common } from './shader-library.js?v=20260913g';
import { extensions } from './shader-extensions.js?v=20260913g';

const universe = `
void main() {
  float aspect = uResolution.x/uResolution.y;
  bool mobile = aspect<1.25;
  vec2 p = (vUv-.5)*vec2(aspect,1);
  vec3 color = cosmos(p,3.8);
  // The quiet left side carries the typography; the nebula opens on the right.
  float reveal = smoothstep(.22,.66,vUv.x);
  if(mobile) reveal = 1.0-smoothstep(.47,.88,vUv.y);
  color *= mix(.035,1.0,reveal);
  vec2 nebula=p+vec2(0,uTime*.0004);
  float vein=nebula.y-nebula.x*.65+.22+(fbm(nebula*3.0+7.0)-.5)*.42;
  float dust=pow(fbm(nebula*80.0),3.0);
  color+=vec3(.8,.008,.34)*exp(-abs(vein)*55.0)*dust*reveal*.5;
  vec2 origin = mobile ? vec2(.55,.35) : vec2(.655,.51);
  vec2 q = (vUv-origin)*vec2(aspect,1)-uPointer*.003;
  float scale = mobile ? .79 : 1.0;
  q /= scale;
  q.y -= sin(uTime*.65)*.008;
  vec2 a = vec2(-.184,.15), b = vec2(.184,.15), c = vec2(0,-.16);
  float edge = min(segment(q,a,b), min(segment(q,b,c),segment(q,c,a)));
  float breathing = .91+.09*sin(uTime*.8);
  color += emission(edge,vec3(1.0,.035,.001),breathing);
  color += vec3(1.0,.028,.001)*exp(-edge*100.0)*1.5;
  float path=mod(uTime*.24,3.0);
  vec2 spark=path<1.0 ? mix(a,b,path) : path<2.0 ? mix(b,c,path-1.0) : mix(c,a,path-2.0);
  color+=emission(length(q-spark),vec3(1,.22,.03),.38);
  // Fine interference filaments emphasize suspension, not a reflective floor.
  float vertical = exp(-abs(q.x)*1200.0)*exp(-abs(q.y)*1.9);
  color += vec3(.5,.027,.01)*vertical*.32;
  for(int i=0;i<3;i++) {
    float fi=float(i);
    vec2 pin = vec2((fi-1.0)*.12,.15);
    float d=length(q-pin);
    color += vec3(1.0,.14,.028)*exp(-d*220.0)*.9;
  }
  outputColor(color);
}`;

const uwuchat = `
void main() {
  float aspect=uResolution.x/uResolution.y;
  vec2 p=(vUv-.5)*vec2(aspect,1);
  vec3 color=cosmos(p+vec2(12,4),1.5)*.6;
  float t=uTime*.12;
  for(int i=0;i<14;i++) {
    float f=float(i), phase=f*.19;
    float envelope=.65+.35*cos(p.x*1.7);
    float wave=sin(p.x*6.3+t+phase)*.145*envelope;
    wave+=sin(p.x*10.0-t*.6+phase*2.0)*.035;
    float slope=cos(p.x*6.3+t+phase)*.91;
    float d=abs(p.y-wave+.02)/(sqrt(1.0+slope*slope));
    vec3 tint=mix(vec3(.015,.19,1.0),vec3(1.0,.008,.65),f/13.0);
    float energy=.022+.035*pow(.5+.5*sin(p.x*7.0-t*2.0+phase),8.0);
    color+=emission(d,tint,energy);
    color+=tint*exp(-d*180.0)*energy*1.5;
  }
  for(int i=0;i<3;i++) {
    float x=.35+float(i)*.40+.028*sin(t+float(i));
    float y=sin(x*6.3+t+.6)*.145+sin(x*10.0-t*.6+1.2)*.035-.02;
    float d=length(p-vec2(x,y));
    color+=emission(d,vec3(.8,.035,1.0),.3);
  }
  outputColor(color);
}`;

const spikeforge = `
void main() {
  float aspect=uResolution.x/uResolution.y;
  vec2 p=(vUv-vec2(.77,.49))*vec2(aspect,1);
  vec3 color=cosmos(p+vec2(3,16),2.0)*.8;
  float t=uTime*.14;
  for(int i=0;i<10;i++) {
    float fi=float(i), angle=fi*2.39996;
    vec2 dir=vec2(cos(angle),sin(angle));
    vec2 side=vec2(-dir.y,dir.x);
    vec2 last=vec2(0);
    for(int j=1;j<6;j++) {
      float fj=float(j), r=fj*.15;
      vec2 next=dir*r+side*(noise(vec2(fi*7.0,fj))-.5)*r*.75;
      next+=side*sin(t*2.0+fi+fj)*.009;
      float d=segment(p,last,next);
      vec3 tint=mix(vec3(.015,.36,1),vec3(.7,.01,1),hash(vec2(fi,fj)));
      float pulse=pow(.5+.5*sin(r*8.0-t*2.0+fi),12.0);
      color+=emission(d,tint,.055+pulse*.12);
      vec2 fork=next+side*(hash(vec2(fi,fj+8.0))-.5)*.22+dir*.09;
      color+=emission(segment(p,next,fork),tint,.025);
      color+=tint*exp(-length(p-next)*260.0)*pulse*2.5;
      float travel=fract(uTime*.20+fi*.137)*5.0;
      float part=travel-(fj-1.0);
      float gate=smoothstep(0.0,.12,part)*(1.0-smoothstep(.85,1.0,part));
      vec2 signal=mix(last,next,clamp(part,0.0,1.0));
      color+=emission(length(p-signal),tint,gate*.34);
      last=next;
    }
  }
  float core=length(p);
  color+=emission(core,vec3(.3,.03,1),.72);
  color+=vec3(.07,.03,.26)*exp(-core*18.0);
  outputColor(color);
}`;

const luna = `
void main() {
  float aspect=uResolution.x/uResolution.y;
  vec2 p=(vUv-vec2(.74,.51))*vec2(aspect,1);
  vec3 color=cosmos(p+vec2(22,8),1.6)*.55;
  float radius=.365, d=length(p);
  float rim=abs(d-radius);
  float aa=1.0/uResolution.y;
  if(d<radius) {
    vec3 normal=vec3(p/radius,sqrt(max(0.0,1.0-dot(p,p)/(radius*radius))));
    vec2 terrain=vec2(atan(normal.x,normal.z)+uTime*.012,asin(normal.y))*7.0;
    float surface=fbm(terrain+vec2(0,4.0));
    float rough=noise(terrain*37.0);
    float light=pow(max(0.0,dot(normal,normalize(vec3(1.0,.22,-.15)))),3.0);
    float craters=fbm(terrain*9.0);
    color=vec3(.0005,.0008,.002)*(1.0+surface*2.0);
    color+=vec3(.22,.035,.012)*light*(surface*.8+rough*.18+craters*.35);
  }
  float litSide=smoothstep(-.24,.30,p.x+p.y*.22);
  float outside=smoothstep(radius-aa,radius+aa,d);
  color+=emission(rim,vec3(1.0,.055,.006),litSide*.40);
  color+=vec3(.4,.016,.002)*exp(-rim*38.0)*outside*litSide;
  float glintX=.10+sin(uTime*.38)*.038;
  vec2 glint=vec2(glintX,sqrt(radius*radius-glintX*glintX));
  float corona=.88+.12*sin(atan(p.y,p.x)*5.0-uTime*.7);
  color+=vec3(.7,.04,.004)*exp(-rim*80.0)*outside*litSide*corona*.45;
  float orbitAngle=uTime*.20+.7;
  vec2 beacon=vec2(cos(orbitAngle),sin(orbitAngle))*(radius+.01);
  color+=emission(length(p-beacon),vec3(1,.16,.025),.23);
  color+=emission(length(p-glint),vec3(1,.13,.01),.48);
  color+=vec3(.85,.05,.008)*exp(-abs(p.x-glint.x)*1100.0)*exp(-abs(p.y-glint.y)*5.0)*.28;
  outputColor(color);
}`;

export const scenes = Object.fromEntries(Object.entries({
  universe, uwuchat, spikeforge, luna, ...extensions
}).map(([name, source]) => [name, common+source]));

// A tessellated 3D sheet follows a gravitational height field. UV derivatives
// keep the grid fine at every distance, without a horizon singularity.
export const gridVertex = `#version 300 es
precision highp float;
uniform vec2 uResolution;
uniform vec2 uPointer;
uniform float uTime;
out vec2 worldXZ;
out float depth;
void main() {
  int cell=gl_VertexID/6, corner=gl_VertexID%6;
  vec2 offsets[6]=vec2[6](vec2(0,0),vec2(1,0),vec2(0,1),vec2(0,1),vec2(1,0),vec2(1,1));
  vec2 uv=(vec2(float(cell%160),float(cell/160))+offsets[corner])/160.0;
  vec2 xz=vec2((uv.x-.5)*18.0,uv.y*13.0-3.8);
  float r=length(xz);
  float y=-1.12*exp(-r*r*.44)+.043*sin(r*2.0-uTime*.65)*exp(-r*.24);
  float z=5.8+xz.y*.968-y*.25;
  vec2 projected=vec2(xz.x,y*.968+xz.y*.25)*1.32/z;
  float aspect=uResolution.x/uResolution.y;
  vec2 center=aspect<1.25 ? vec2(.55,.30) : vec2(.655,.39);
  float scale=aspect<1.25 ? .73:1.0;
  vec2 ndc=(center-.5)*2.0+projected*vec2(2.0/aspect,2.0)*scale;
  ndc+=uPointer*.004;
  gl_Position=vec4(ndc*z,0.0,z);
  worldXZ=xz; depth=z;
}`;

export const gridFragment = `#version 300 es
precision highp float;
in vec2 worldXZ; in float depth; out vec4 fragColor;
uniform float uTime;
uniform float uEncoding;
uniform vec2 uResolution;
void main() {
  vec2 coord=worldXZ/.36;
  vec2 derivative=max(fwidth(coord),vec2(.0001));
  vec2 d=abs(fract(coord-.5)-.5)/derivative;
  float line=1.0-smoothstep(.35,.95,min(d.x,d.y));
  float r=length(worldXZ);
  float ring=abs(r-2.22);
  float sweep=pow(.5+.5*sin(atan(worldXZ.y,worldXZ.x)*2.0-uTime*.48),8.0);
  float screenFade=uResolution.x/uResolution.y<1.25 ? 1.0 : smoothstep(.22,.64,gl_FragCoord.x/uResolution.x);
  float fade=exp(-r*.24)*smoothstep(1.3,3.0,depth)*screenFade;
  vec3 color=vec3(.006,.095,.43)*line*fade*(.8+1.6*exp(-abs(r-1.4)*2.0));
  color+=vec3(.008,.18,1.0)*exp(-abs(r-1.3)*65.0)*(.3+sweep*.35);
  color+=vec3(.25,.045,.008)*exp(-ring*80.0)*screenFade;
  color+=vec3(.04,.15,.45)*exp(-abs(r-1.3)*10.0)*.10;
  float ripple=exp(-pow(r-mod(uTime*.28,4.8),2.0)*35.0);
  color+=vec3(.02,.12,.42)*line*ripple*fade*.5;
  fragColor=vec4(color/uEncoding,0.0);
}`;
