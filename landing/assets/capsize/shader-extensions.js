// Additional homepage scenes use the shared noise, emission, and HDR pipeline.
// Their visual structures deliberately differ from the hero's gravity sheet.
const arcade = `
void main() {
  float aspect=uResolution.x/uResolution.y;
  vec2 p=(vUv-vec2(.74,.52))*vec2(aspect,1);
  vec3 color=cosmos(p+vec2(14,3),.9)*.28;
  float t=uTime*.10;
  float angle=.23+sin(t*.24)*.055;
  mat2 rotate=mat2(cos(angle),-sin(angle),sin(angle),cos(angle));
  vec2 q=rotate*p;
  for(int i=0;i<10;i++) {
    float layer=fract(float(i)/10.0+t*.13);
    float radius=.04+layer*layer*.86;
    vec2 b=abs(q)-vec2(radius,radius*.69);
    float frame=abs(max(b.x,b.y));
    float fade=sin(layer*PI)*.24;
    vec3 tint=mix(vec3(.14,.015,1),vec3(1,.08,.008),layer);
    color+=emission(frame,tint,fade);
  }
  float core=length(q);
  color+=vec3(.3,.035,.04)*exp(-core*28.0);
  outputColor(color);
}`;

const art = `
void main() {
  float aspect=uResolution.x/uResolution.y;
  vec2 p=(vUv-.5)*vec2(aspect,1);
  float t=uTime*.09;
  vec2 q=p*2.0;
  q+=vec2(sin(q.y*2.1+t),cos(q.x*2.4-t))*.3;
  float f=fbm(q+vec2(t*.08,4.0));
  float contour=abs(fract(f*16.0-t*.06)-.5);
  vec3 tint=mix(vec3(.008,.43,.70),vec3(.66,.015,.34),smoothstep(.25,.72,f));
  vec3 color=vec3(.001,.005,.009)+tint*pow(f,4.0)*.2;
  float sweep=.65+.35*sin(p.x*3.0+p.y*2.0-t*.8);
  color+=tint*exp(-contour*55.0)*1.25*sweep;
  color+=vec3(.08,.48,.7)*exp(-contour*180.0)*.62;
  float focus=exp(-length(p-vec2(.42,.1))*2.0);
  color*=.4+focus;
  outputColor(color);
}`;

const radio = `
void main() {
  float aspect=uResolution.x/uResolution.y;
  vec2 p=(vUv-.5)*vec2(aspect,1);
  vec3 color=vec3(.002,.004,.004);
  float time=uTime*.55;
  vec2 q=p-vec2(aspect*.19,0);
  float x=floor(q.x*55.0)/55.0;
  float envelope=.04+.13*pow(.5+.5*sin(x*8.0-time),2.0);
  envelope+=.075*noise(vec2(x*14.0,time*.65));
  float bars=(1.0-smoothstep(.22,.38,abs(fract(q.x*55.0)-.5)));
  float level=1.0-smoothstep(envelope,envelope+.006,abs(q.y));
  float segments=1.0-smoothstep(.32,.48,abs(fract(q.y*100.0)-.5));
  vec3 green=vec3(.035,.52,.22), amber=vec3(.8,.30,.018);
  vec3 tint=mix(green,amber,smoothstep(.10,.24,abs(q.y)));
  color+=tint*bars*level*segments*.8;
  color+=tint*exp(-max(0.0,abs(q.y)-envelope)*40.0)*bars*.018;
  float baseline=exp(-abs(q.y+.31)*750.0);
  float ticks=exp(-abs(fract(q.x*11.0)-.5)*90.0);
  color+=vec3(.08,.20,.13)*(baseline+ticks*exp(-abs(q.y+.31)*80.0))*.25;
  outputColor(color);
}`;

const navigation = `
void main() {
  vec2 p=vec2(vUv.x*3.2,vUv.y*.25);
  vec3 color=nebulaField(p+vec2(4.0,9.0),1.6);
  color*=.75+.25*sin(vUv.x*5.0+uTime*.07);
  outputColor(color);
}`;

export const extensions = { arcade, art, radio, navigation };
