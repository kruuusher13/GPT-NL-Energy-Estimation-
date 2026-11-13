import numpy as np, wave
sr=48000; T=30.0; n=int(sr*T); t=np.arange(n)/sr; rng=np.random.default_rng(7)
def lp(x,fc):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/sr); return np.fft.irfft(X/(1+(f/fc)**4),len(x))
def hp(x,fc):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/sr); r=(f/fc)**4; return np.fft.irfft(X*r/(1+r),len(x))
m2f=lambda m:440*2**((m-69)/12)
secs=[(0,6.0,[50,57,62,65],900),(5.6,13.8,[46,53,58,62],1100),(13.4,16.9,[43,50,55,58,62],1300),(16.4,25.4,[41,48,53,57,60],1500),(25.0,30,[38,50,57,64,65],1200)]
L=np.zeros(n);R=np.zeros(n)
for a,b,notes,fc in secs:
    e=np.clip((t-a)/1.2,0,1)*np.clip((b-t)/1.2,0,1); e=e*e*(3-2*e)
    for ch,det in ((L,1.0012),(R,0.9988)):
        v=np.zeros(n)
        for m in notes:
            f0=m2f(m)*det
            for k in range(1,6): v+=np.sin(2*np.pi*f0*k*t+k)/k
        ch+=lp(v,fc)*e*0.05
def hit(x,at,amp,f1=48,f2=110,dec=7):
    i=int(at*sr); tt=np.arange(n-i)/sr
    x[i:]+=amp*np.sin(2*np.pi*(f1*tt+(f2/30)*(1-np.exp(-tt*30))))*np.exp(-tt*dec)
kick=np.zeros(n)
for tb in np.arange(2.0,25.1,0.625): hit(kick,tb,0.32 if tb<16.6 else 0.42)
hat=np.zeros(n); nz=rng.standard_normal(n)
for tb in np.arange(13.6,25.1,0.3125):
    i=int(tb*sr); w=min(int(0.06*sr),n-i); hat[i:i+w]+=nz[i:i+w]*np.exp(-np.arange(w)/sr*70)*0.05
hat=hp(hat,6000)
r=np.clip((t-22.8)/2.4,0,1)*(t<25.2); riser=hp(rng.standard_normal(n),2500)*r**2.2*0.09+np.sin(2*np.pi*(200*t+300*np.clip(t-22.8,0,2.4)**2))*r**2*0.03
boom=np.zeros(n); hit(boom,25.2,0.75,f1=34,f2=60,dec=1.3)
i=int(25.2*sr); bn=np.zeros(n); bn[i:]=lp(rng.standard_normal(n-i),400)*np.exp(-np.arange(n-i)/sr*3)*0.5; boom+=bn
dry=0*kick
ir=rng.standard_normal(int(1.8*sr))*np.exp(-np.arange(int(1.8*sr))/sr*3.2); ir=lp(ir,3000); ir/=np.abs(ir).sum()**0.5*8
def conv(x):
    N=1<<int(np.ceil(np.log2(len(x)+len(ir)))); return np.fft.irfft(np.fft.rfft(x,N)*np.fft.rfft(ir,N),N)[:len(x)]
outs=[]
for ch in (L,R):
    s=ch+dry; s=s+0.35*conv(s)
    outs.append(s)
st=np.stack(outs,1); st*=np.clip(t/0.4,0,1)[:,None]*np.clip((30-t)/1.4,0,1)[:,None]
st=st/np.abs(st).max()*0.28
pcm=(st*32767).astype(np.int16)
w=wave.open("music.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes()); w.close()
print("music ok")
