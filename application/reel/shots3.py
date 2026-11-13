import subprocess, os, glob, shutil, json
from PIL import Image
FN=os.path.abspath("fonts")+"/"
E=["-c:v","libx264","-preset","fast","-crf","14","-an","-pix_fmt","yuv420p"]
PX,PY,PW,PH=96,118,1152,720
def run(a):
    r=subprocess.run(["ffmpeg","-v","error","-y"]+a,capture_output=True,text=True)
    if r.returncode: print(r.stderr[-2500:]); raise SystemExit("ffmpeg failed: "+" ".join(a[-3:]))
def compose(panel,layer,out,dur):
    # panel: 1152x720 clip -> place on paper canvas, then the layout layer on top
    run(["-f","lavfi","-i",f"color=c=0xf6f7f8:s=1920x1080:r=24:d={dur}","-i",panel,"-loop","1","-t",str(dur),"-i",layer,"-filter_complex",
         f"[0][1]overlay={PX}:{PY}:shortest=1[a];[a][2]overlay=0:0,format=yuv420p","-t",str(dur)]+E+[out])
FIT_BG={"white":"0xffffff","black":"0x0b0e12"}
def video_panel(src,ss,dur,out,crop=None):
    vf=(f"crop={crop}," if crop else "")+f"scale={PW}:{PH}:force_original_aspect_ratio=increase:flags=lanczos,crop={PW}:{PH},fps=24,setsar=1,format=yuv420p"
    run(["-ss",str(ss),"-t",str(dur),"-i",src,"-vf",vf]+E+[out])
def frames_panel(frames,dur,out,box=None,bg="white",hold_last=0.0):
    d="seq_"+out; shutil.rmtree(d,ignore_errors=True); os.makedirs(d)
    for i,f in enumerate(frames):
        im=Image.open(f).convert("RGB")
        if box:
            x,y,w,h=box; w=min(w,im.width-x); h=min(h,im.height-y); im=im.crop((x,y,x+w,y+h))
        im.save(f"{d}/{i:04d}.png")
    W=max(Image.open(f"{d}/{i:04d}.png").size[0] for i in range(len(frames))); H=max(Image.open(f"{d}/{i:04d}.png").size[1] for i in range(len(frames)))
    for i in range(len(frames)):
        im=Image.open(f"{d}/{i:04d}.png"); c=Image.new("RGB",(W,H),"white" if bg=="white" else (11,14,18)); c.paste(im,(0,0)); c.save(f"{d}/{i:04d}.png")
    rate=len(frames)/dur
    vf=(f"scale={PW-80}:{PH-60}:force_original_aspect_ratio=decrease:flags=lanczos,pad={PW}:{PH}:(ow-iw)/2:(oh-ih)/2:color={FIT_BG[bg]},fps=24,setsar=1,format=yuv420p")
    run(["-framerate",f"{rate:.5f}","-i",f"{d}/%04d.png","-vf",vf,"-t",str(dur)]+E+[out])
def still_panel(img,dur,out,bg="black",zoom=0.0009):
    n=int(dur*24)+1
    run(["-i",img,"-vf",f"scale={(PW-40)*3}:-2:flags=lanczos,zoompan=z='1+{zoom}*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={PW-40}x{int((PW-40)*Image.open(img).height/Image.open(img).width)//2*2}:fps=24,pad={PW}:{PH}:(ow-iw)/2:(oh-ih)/2:color={FIT_BG[bg]},setsar=1,format=yuv420p","-frames:v",str(n-1)]+E+[out])
BOX=json.load(open("cap/boxes.json"))
S=lambda p: sorted(glob.glob(p))
os.makedirs("p",exist_ok=True)
plan=[]
# title
run(["-loop","1","-t","2.8","-i","lay/title.png","-vf","fps=24,format=yuv420p"]+E+["x00.mp4"]); plan.append("x00")
video_panel("sat.mp4",0.2,2.0,"p/h1.mp4"); compose("p/h1.mp4","lay/h1.png","x01.mp4",2.0); plan.append("x01")
frames_panel(S("cap/fpgaA_*.png")+S("cap/fpgaB_*.png"),3.0,"p/f2.mp4",box=BOX["cap3"]); compose("p/f2.mp4","lay/f2.png","x02.mp4",3.0); plan.append("x02")
video_panel("rov.mp4",0.3,1.6,"p/h3.mp4"); compose("p/h3.mp4","lay/h3.png","x03.mp4",1.6); plan.append("x03")
video_panel("m2.mp4",5.5,1.8,"p/f4.mp4",crop="1590:894:330:100"); compose("p/f4.mp4","lay/f4.png","x04.mp4",1.8); plan.append("x04")
frames_panel(list(reversed(S("cap/pf_*.png"))),2.6,"p/f5.mp4",box=BOX["cap1"]); compose("p/f5.mp4","lay/f5.png","x05.mp4",2.6); plan.append("x05")
x,y,w,h=BOX["cap2"]; frames_panel(S("cap/chess_*.png"),3.3,"p/f6.mp4",box=(x,y,w,min(h,600))); compose("p/f6.mp4","lay/f6.png","x06.mp4",3.3); plan.append("x06")
frames_panel(["cap/g_Torxflow.png","cap/g_ConcertLab.png"],3.0,"p/f7.mp4"); compose("p/f7.mp4","lay/f7.png","x07.mp4",3.0); plan.append("x07")
video_panel("hall.mp4",0.3,1.5,"p/h8.mp4"); compose("p/h8.mp4","lay/h8.png","x08.mp4",1.5); plan.append("x08")
frames_panel(S("cap/energy_*.png"),2.4,"p/f9.mp4",box=BOX["cap0"]); compose("p/f9.mp4","lay/f9.png","x09.mp4",2.4); plan.append("x09")
# terminal, light
cmd="gptnl-energy forecast --n 400000"
lines=[("Energy forecast — 400,000 docs | corpus=american_stories (~1,798 chars/doc)","0x68727a"),
("  data_splitting                17.4 kJ","0x000000"),("  string_normalization         175.2 kJ","0x000000"),("  heuristic_filtering          624.8 kJ","0x000000"),
("  toxic_language_detection      60.8 kJ","0x000000"),("  deduplication                125.1 kJ","0x000000"),("  TOTAL PIPELINE                1.00 MJ","0x2a78d6"),
("  =  0.33 kWh (PUE 1.2)  =  EUR 0.10  =  0.10 kg CO2","0x2a78d6")]
os.makedirs("txt",exist_ok=True); k=[0]; f=[]
def tdt(t,x,y,col,en,size=24):
    k[0]+=1; fn=f"txt/t{k[0]}.txt"; open(fn,"w").write(t)
    f.append(f"drawtext=fontfile={FN}m.ttf:textfile={fn}:x={x}:y={y}:fontsize={size}:fontcolor={col}:expansion=none:enable='{en}'")
tdt("$",60,90,"0x1baf7a","gte(t,0)")
for i in range(1,len(cmd)+1):
    a=0.08+i*0.02; tdt(cmd[:i]+("▌" if i<len(cmd) else ""),88,90,"0x000000",f"between(t,{a:.3f},{a+0.02:.3f})" if i<len(cmd) else f"gte(t,{a:.3f})")
for i,(s,c) in enumerate(lines): tdt(s,60,170+52*i,c,f"gte(t,{0.85+0.07*i:.2f})")
run(["-f","lavfi","-i",f"color=c=0xffffff:s={PW}x{PH}:d=1.8:r=24","-vf",",".join(f)+",format=yuv420p","-t","1.8"]+E+["p/f10.mp4"])
compose("p/f10.mp4","lay/f10.png","x10.mp4",1.8); plan.append("x10")
still_panel("mon.png",1.9,"p/f11.mp4"); compose("p/f11.mp4","lay/f11.png","x11.mp4",1.9); plan.append("x11")
run(["-loop","1","-t","2.3","-i","lay/refs.png","-vf","fps=24,format=yuv420p"]+E+["x12.mp4"]); plan.append("x12")
open("list.txt","w").write("".join(f"file {x}.mp4\n" for x in plan))
run(["-f","concat","-safe","0","-i","list.txt","-c","copy","base.mp4"])
print("shots ok")
